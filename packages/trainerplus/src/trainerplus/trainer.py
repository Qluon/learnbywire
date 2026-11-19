"""Thin HF Trainer orchestration with a separate public control-session adapter."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .artifacts import (TrainResult, VERSION, checkpoint_record, clock, identity,
    isolated_caches, model_identity, new_run, preserve_random_state, save, sha_file, verify_checkpoint, versions)
from .config import DataConfig, TrainingConfig, positive
from .data import TokenCollator, TokenDataset, prepare_dataset
from .integration import load_optimizerplus


class TrainerPlus:
    def __init__(self, *, model, optimizer, train_dataset, task="cpt", control=None,
                 tokenizer=None, eval_dataset=None, args=None, data_config=None,
                 control_options=None):
        self.model, self.optimizer = model, optimizer
        self.args = args or TrainingConfig()
        self.data_config = data_config or DataConfig()
        if not isinstance(self.args, TrainingConfig) or not isinstance(self.data_config, DataConfig):
            raise TypeError("Use TrainingConfig and DataConfig")
        self.task, self.control, self.tokenizer = task, control, tokenizer
        self.control_options = dict(control_options or {})
        if control is not None:
            # Record the effective reduction in manifests and checkpoint identity.
            self.control_options.setdefault("loss_reduction", "token_mean")
        json.dumps(self.control_options, allow_nan=False)
        if control is None and self.control_options:
            raise ValueError("control_options require an explicit control")
        self.train_dataset = prepare_dataset(train_dataset, task=task, tokenizer=tokenizer, config=self.data_config)
        self.eval_dataset = None if eval_dataset is None else prepare_dataset(
            eval_dataset, task=task, tokenizer=tokenizer, config=self.data_config)
        self.collator = TokenCollator(tokenizer.pad_token_id if tokenizer is not None else 0)
        self.run_dir = None
        self.committed_steps = 0
        self.last_checkpoint = None
        self._used = False
        self._active = False
        self._hf_trainer = None
        self._validate()

    def _validate(self):
        params = list(self.model.parameters())
        devices = {p.device for p in params}
        if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
            raise ValueError("This release supports a model on one CPU or CUDA device")
        self.device = next(iter(devices))
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Use a single process; distributed orchestration is not qualified yet")
        if self.device.type == "cuda" and torch.cuda.device_count() != 1:
            raise ValueError("Select one visible GPU with CUDA_VISIBLE_DEVICES before launching")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise ValueError("Distributed TrainerPlus orchestration is not qualified yet")
        trainable = {id(p) for p in params if p.requires_grad}
        owned = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if not trainable or set(owned) != trainable or len(owned) != len(set(owned)):
            raise ValueError("Optimizer must own exactly the trainable parameters, once each")
        if hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
            raise ValueError("Supply an unscheduled optimizer; TrainingConfig owns the common schedule")
        cfg = getattr(self.model, "config", None)
        if getattr(cfg, "is_encoder_decoder", False):
            raise ValueError("This release supports causal language models")
        if self.args.eval_steps is not None and self.eval_dataset is None:
            raise ValueError("eval_steps requires eval_dataset")
        if self.control is not None:
            if not callable(self.control):
                raise TypeError("Use control=OptimizerPlus (the imported class), or control=None")
            if not self.train_dataset.fixed_length:
                raise ValueError("OptimizerPlus currently requires fixed padding; native training supports dynamic padding")
            if self.args.mixed_precision != "bf16" or self.args.native_compile or self.args.max_grad_norm is not None:
                raise ValueError("OptimizerPlus requires BF16 and owns compilation/clipping")
            reserved = {"model", "train_loader", "total_steps", "examples_per_update", "microbatch_size",
                        "resume_from", "workload_id", "lr_schedule", "warmup_ratio"}
            if reserved.intersection(self.control_options):
                raise ValueError("Shared batching, scheduling and recovery are owned by TrainerPlus settings")
            reduction = self.control_options.get("loss_reduction")
            if not isinstance(reduction, str) or reduction not in {"token_mean", "microbatch_mean"}:
                raise ValueError("control_options.loss_reduction must be token_mean or microbatch_mean")

    def _prepare_order(self):
        cfg = self.args
        group = cfg.effective_batch_size
        remainder = len(self.train_dataset) % group
        if remainder and not cfg.drop_last:
            raise ValueError("Incomplete update group; choose drop_last=True or a divisible effective batch")
        usable = len(self.train_dataset)-remainder
        available = usable//group * cfg.num_epochs
        steps = cfg.max_steps if cfg.max_steps is not None else available
        if steps < 1 or steps > available:
            raise ValueError("Update budget exceeds the examples available in the configured epochs")
        generator = torch.Generator().manual_seed(cfg.seed)
        order = []
        for _ in range(cfg.num_epochs):
            indices = torch.randperm(len(self.train_dataset), generator=generator).tolist() if cfg.shuffle else list(range(len(self.train_dataset)))
            order.extend(indices[:min(usable, steps*group-len(order))])
            if len(order) == steps*group:
                break
        self._ordered = self.train_dataset.ordered(order)
        self._total_steps = steps
        return order, remainder

    def _contract(self, order, remainder):
        initial = model_identity(self.model)
        self._initial_hash = initial.pop("initial_trainable_sha256")
        settings = asdict(self.args)
        for name in ("output_dir", "save_model", "logging_steps", "save_steps"):
            settings.pop(name)
        groups = [{k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                   for k, v in group.items() if k != "params"} for group in self.optimizer.param_groups]
        return dict(schema=1, model=initial, task=self.task, settings=settings,
            data_config=asdict(self.data_config), dataset_sha256=self.train_dataset.fingerprint(),
            evaluation_sha256=self.eval_dataset.fingerprint() if self.eval_dataset else None,
            order_sha256=identity(order), total_steps=self._total_steps, dropped_rows_per_epoch=remainder,
            optimizer_class=f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            optimizer_groups=groups, control="OptimizerPlus" if self.control is not None else None,
            control_version=load_optimizerplus().__version__ if self.control is not None else None,
            control_options=self.control_options, versions=versions(),
            harness_source={p.name:sha_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})

    def _emit(self, event, **values):
        record = dict(event=event, committed_steps=self.committed_steps, **values)
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        save(self.run_dir / "status.json", dict(status="running", committed_steps=self.committed_steps,
                                                total_steps=self._total_steps))

    def _after_update(self, step, *, loss=None, startup=False):
        before = self.committed_steps
        if type(step) is not int or not before <= step <= self._total_steps or (step == before and not startup):
            raise RuntimeError("Invalid committed update progress")
        if loss is not None and not math.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite training loss")
        self.committed_steps = step
        if startup or step == self._total_steps or step//self.args.logging_steps > before//self.args.logging_steps:
            self._emit("startup" if startup else "update", previous_committed_steps=before, loss=loss)
        if self.args.eval_steps and step//self.args.eval_steps > before//self.args.eval_steps:
            self._emit("evaluation", metrics=self._evaluate(self.eval_dataset),
                       execution="active_control" if self.control is not None else "hf_native")

    def _native(self, resume, stop):
        # Public Trainer owns forward/backward, accumulation and optimizer steps.
        # Only documented callbacks, dataset/collator and optimizer arguments are used.
        from transformers import Trainer, TrainerCallback, TrainingArguments
        owner = self

        class Observe(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                owner._after_update(int(state.global_step))
                if stop is not None and state.global_step >= stop:
                    control.should_training_stop = True
                    control.should_save = True
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    owner._emit("hf_log", values=logs)

            def on_save(self, args, state, control, **kwargs):
                path = Path(args.output_dir)/f"checkpoint-{state.global_step}"
                checkpoint_record(path, contract=owner._workload, step=int(state.global_step), control=None)
                owner.last_checkpoint = path

        cfg = self.args
        hf_args = TrainingArguments(output_dir=str(self.run_dir/"hf"), use_cpu=self.device.type == "cpu",
            per_device_train_batch_size=cfg.microbatch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=self._total_steps, num_train_epochs=1, seed=cfg.seed, data_seed=cfg.seed,
            train_sampling_strategy="sequential", dataloader_num_workers=0, dataloader_pin_memory=False,
            bf16=cfg.mixed_precision == "bf16", torch_compile=cfg.native_compile,
            max_grad_norm=cfg.max_grad_norm or 0.0, lr_scheduler_type=cfg.lr_schedule,
            warmup_steps=math.ceil(self._total_steps*cfg.warmup_ratio),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            logging_steps=cfg.logging_steps, logging_nan_inf_filter=False, disable_tqdm=True,
            save_strategy="steps" if cfg.save_steps else "no", save_steps=cfg.save_steps or 500,
            report_to=[], remove_unused_columns=False)
        self._hf_trainer = Trainer(model=self.model, args=hf_args, train_dataset=self._ordered,
            data_collator=self.collator, optimizers=(self.optimizer, None), callbacks=[Observe()])
        if cfg.gradient_accumulation_steps > 1 and not self._hf_trainer.model_accepts_loss_kwargs:
            raise ValueError("For matched token-weighted accumulation, the native model must accept HF loss kwargs")
        save(self.run_dir/"hf_training_arguments.json", hf_args.to_dict())
        self._hf_trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if self.committed_steps != self._hf_trainer.state.global_step:
            raise RuntimeError("HF Trainer and wrapper progress disagree")
        self.model = self._hf_trainer.accelerator.unwrap_model(self._hf_trainer.model_wrapped,
            keep_fp32_wrapper=False, keep_torch_compile=False)

    def _validate_control_device(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("OptimizerPlus 1.2.2 Run+ requires CUDA")
        if getattr(self.model.config, "use_cache", False):
            raise ValueError("Set model.config.use_cache=False")
        if getattr(self.model, "is_loaded_in_4bit", False) or getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Quantized control integration needs separate qualification")
        if any(p.requires_grad and p.dtype == torch.float16 for p in self.model.parameters()):
            raise ValueError("Trainable FP16 parameters are not supported by this integration")

    def _controlled(self, resume, stop):
        self._validate_control_device()
        module = load_optimizerplus()
        if self.control is not module.OptimizerPlus:
            raise ValueError("This adapter requires the imported OptimizerPlus 1.2.2 control class")
        loader = module.TokenBatches(**self._ordered.tensors(), examples_per_update=self.args.effective_batch_size,
                                     microbatch_size=self.args.microbatch_size,
                                     pad_token_id=self.collator.pad_token_id)
        options = dict(self.control_options, lr_schedule=None if self.args.lr_schedule == "constant" else "cosine",
                       warmup_ratio=self.args.warmup_ratio)
        if resume:
            options.update(resume_from=resume, workload_id=identity(self._workload))
        session = self.control(self.optimizer, model=self.model, train_loader=loader,
                               total_steps=self._total_steps, **options)
        last_saved_step = self.committed_steps

        def save_boundary(force=False):
            nonlocal last_saved_step
            step = self.committed_steps
            due = self.args.save_steps and step//self.args.save_steps > last_saved_step//self.args.save_steps
            if step < self._total_steps and step != last_saved_step and (force or due):
                path = self.run_dir/f"checkpoint-{step}"
                session.save_checkpoint(path, workload_id=identity(self._workload))
                checkpoint_record(path, contract=self._workload, step=step, control="OptimizerPlus")
                self.last_checkpoint, last_saved_step = path, step

        try:
            self._emit("control_runtime", summary=session.runtime_summary)
            self._after_update(int(session.committed_steps), startup=True)
            save_boundary(force=stop is not None and self.committed_steps >= stop)
            if stop is not None and self.committed_steps >= stop:
                return
            for batch in session:
                loss = session.train_step(batch)
                self._after_update(int(session.committed_steps), loss=float(loss))
                stopping = stop is not None and self.committed_steps >= stop
                save_boundary(force=stopping)
                if stopping:
                    break
        finally:
            session.close()

    def train(self, *, resume_from_checkpoint=None, stop_after_steps=None):
        if self._used:
            raise RuntimeError("TrainerPlus instances are single-use; create a new one for another run or resume")
        positive("stop_after_steps", stop_after_steps, optional=True)
        self._validate()
        order, remainder = self._prepare_order()
        self._used = True
        self.run_dir = new_run(self.args.output_dir)
        started = clock()
        self._active = True
        start_step = 0
        try:
            if self.control is not None:
                self._validate_control_device()
            self._workload = self._contract(order, remainder)
            resume = None
            if resume_from_checkpoint:
                resume, start_step = verify_checkpoint(resume_from_checkpoint, self._workload)
                self.committed_steps = start_step
            if stop_after_steps is not None and not start_step < stop_after_steps < self._total_steps:
                raise ValueError("stop_after_steps must be after the resumed step and before the original total budget")
            save(self.run_dir/"manifest.json", dict(contract=self._workload, workload_sha256=identity(self._workload),
                initial_trainable_sha256=self._initial_hash, resume_from=str(resume) if resume else None,
                data_preparation=self.train_dataset.metadata,
                timing="training_seconds includes engine/session setup, probes, periodic evaluation and saves; workflow additionally includes manifest, final evaluation/export; earlier model loading/tokenization/order materialization excluded"))
            self._emit("begin")
            from transformers import set_seed
            set_seed(self.args.seed)
            self.model.train()
            self._sync()
            training_started = clock()
            with isolated_caches(self.run_dir):
                if self.control is None:
                    self._native(resume, stop_after_steps)
                else:
                    self._controlled(resume, stop_after_steps)
            self._sync()
            elapsed = clock()-training_started
            complete = self.committed_steps == self._total_steps
            if not complete and (stop_after_steps is None or self.committed_steps < stop_after_steps):
                raise RuntimeError("Training ended before the declared update budget")
            if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters() if p.requires_grad):
                raise FloatingPointError("Nonfinite trained parameters")
            group = self.args.effective_batch_size
            metrics = self._ordered.counts(start_step*group, self.committed_steps*group)
            metrics.update(training_seconds=elapsed, input_tokens_per_second=metrics["input_tokens"]/elapsed,
                supervised_tokens_per_second=metrics["supervised_tokens"]/elapsed,
                updates_this_run=self.committed_steps-start_step)
            if self.eval_dataset is not None:
                metrics["evaluation"] = self._evaluate(self.eval_dataset)
            model_path = None
            if complete and self.args.save_model:
                model_path = self.run_dir/"model"
                if callable(getattr(self.model, "save_pretrained", None)):
                    self.model.save_pretrained(model_path, safe_serialization=True)
                else:
                    model_path.mkdir()
                    torch.save(self.model.state_dict(), model_path/"model_state.pt")
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(model_path)
            self._sync()
            metrics["workflow_seconds"] = clock()-started
            status = "complete" if complete else "stopped"
            save(self.run_dir/"result.json", dict(status=status, committed_steps=self.committed_steps,
                metrics=metrics, model_path=str(model_path) if model_path else None,
                checkpoint=str(self.last_checkpoint) if self.last_checkpoint else None))
            save(self.run_dir/"status.json", dict(status=status, committed_steps=self.committed_steps))
            return TrainResult(status, self.committed_steps, self.run_dir, metrics, model_path, self.last_checkpoint)
        except BaseException as error:
            save(self.run_dir/"status.json", dict(status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                committed_steps=self.committed_steps, error=f"{type(error).__name__}: {error}"))
            raise
        finally:
            self._active = False

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def evaluate(self, dataset=None):
        if self._active:
            raise RuntimeError("Use eval_steps for evaluation at completed update boundaries")
        data = self.eval_dataset if dataset is None else prepare_dataset(dataset, task=self.task,
            tokenizer=self.tokenizer, config=self.data_config)
        return self._evaluate(data)

    def _evaluate(self, dataset):
        if dataset is None:
            raise ValueError("Provide eval_dataset")
        modes = [(module, module.training) for module in self.model.modules()]
        nll, count, correct = 0.0, 0, 0
        try:
            with preserve_random_state(self.device), torch.inference_mode():
                self.model.eval()
                for start in range(0, len(dataset), self.args.microbatch_size):
                    batch = self.collator([dataset[i] for i in range(start, min(start+self.args.microbatch_size, len(dataset)))])
                    batch = {k:v.to(self.device) for k,v in batch.items()}
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.args.mixed_precision == "bf16"):
                        output = self.model(**batch)
                    logits = (output["logits"] if isinstance(output, dict) else output.logits)[:, :-1].float()
                    target = batch["labels"][:, 1:]
                    valid = target.ne(-100)
                    nll += float(F.cross_entropy(logits[valid], target[valid], reduction="sum"))
                    count += int(valid.sum())
                    correct += int(logits.argmax(-1)[valid].eq(target[valid]).sum())
            loss = nll/count
            perplexity = math.exp(loss)
            if not math.isfinite(perplexity):
                raise FloatingPointError("Nonfinite evaluation")
            return dict(loss=loss, perplexity=perplexity, next_token_accuracy=correct/count,
                        supervised_tokens=count, loss_reduction="token_mean")
        finally:
            for module, mode in modes:
                module.training = mode
