"""Train a causal LM with your own model and data. Importing never starts training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Hugging Face model ID or local directory')
    parser.add_argument('--revision', help='Optional model commit/revision for reproducibility')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--data', required=True, help='Your JSONL, JSON or CSV training file')
    parser.add_argument('--validation', help='Separate held-out data in the same format')
    parser.add_argument('--control', choices=['optimizerplus', 'native'], default='optimizerplus')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--task', choices=['cpt', 'sft'], default='cpt')
    parser.add_argument('--sft-loss', choices=['auto', 'completion', 'assistant', 'all'], default='auto')
    parser.add_argument('--output', default='runs/first-training')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, help='Optional update budget within the available epochs')
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', default='all-linear', help='all-linear, or comma-separated module names')
    parser.add_argument('--truncate', action='store_true', help='Allow cutting long records; otherwise fail clearly')
    parser.add_argument('--drop-last', action='store_true', help='Discard the incomplete final effective batch')
    parser.add_argument('--native-compile', action='store_true', help='Public HF torch.compile on the native path')
    args = parser.parse_args(argv)
    if args.control == 'optimizerplus' and (args.device != 'cuda' or args.native_compile):
        parser.error('OptimizerPlus requires CUDA and manages compilation; omit --native-compile')
    if args.control == 'optimizerplus' and args.steps is not None and args.steps < 50:
        parser.error('Automatic OP control currently requires at least 50 updates; use --steps 50 or a full epoch')
    if args.lora_rank < 1:
        parser.error('This LoRA example needs a positive --lora-rank')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model
    from trainerplus import TrainerPlus, TrainingConfig, DataConfig, prepare_dataset
    from trainerplus.artifacts import save, sha_file

    # Keep the exact outer script and input identity with this new run.
    launch = Path(args.output) / ('launch_' + uuid.uuid4().hex[:12])
    launch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, launch/'first_run.py')
    save(launch/'launch.json', dict(arguments=vars(args), script_sha256=sha_file(__file__),
        data_sha256=sha_file(args.data),
        validation_sha256=sha_file(args.validation) if args.validation else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    set_seed(args.seed)
    load = dict(local_files_only=args.local_files_only)
    if args.revision:
        load['revision'] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('This example needs a tokenizer with a pad token or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    data_config = DataConfig(max_length=args.length, padding='fixed',
        truncation='truncate' if args.truncate else 'error', sft_loss=args.sft_loss)
    train_data = prepare_dataset(args.data, task=args.task, tokenizer=tokenizer, config=data_config)
    validation = None if args.validation is None else prepare_dataset(
        args.validation, task=args.task, tokenizer=tokenizer, config=data_config)
    config = TrainingConfig(output_dir=launch, num_epochs=args.epochs, max_steps=args.steps,
        microbatch_size=args.microbatch, gradient_accumulation_steps=args.accumulation,
        mixed_precision='bf16' if args.device == 'cuda' else 'no', seed=args.seed,
        lr_schedule=args.schedule, warmup_ratio=args.warmup_ratio,
        max_grad_norm=None, native_compile=args.native_compile, drop_last=args.drop_last)
    batch = config.effective_batch_size
    if len(train_data) % batch and not args.drop_last:
        raise ValueError(f'{len(train_data)} prepared rows do not divide by effective batch {batch}; '
                         'adjust batch sizes or explicitly choose --drop-last')
    available = (len(train_data)//batch) * args.epochs
    steps = args.steps if args.steps is not None else available
    if not 1 <= steps <= available:
        raise ValueError(f'Update budget {steps} exceeds the {available} updates available; adjust data or epochs')
    if args.control == 'optimizerplus' and steps < 50:
        raise ValueError(f'Automatic OP control needs at least 50 updates; this dataset/config gives {steps}')
    print(f'{len(train_data)} prepared rows; effective batch {batch}; {steps} planned updates', flush=True)

    dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
        attn_implementation='sdpa', **load)
    model.config.use_cache = False
    targets = 'all-linear' if args.lora_targets == 'all-linear' else args.lora_targets.split(',')
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=0.0, bias='none', target_modules=targets)).to(args.device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float().contiguous()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
        fused=args.device == 'cuda', foreach=False)
    control, options = None, None
    if args.control == 'optimizerplus':
        from optimizerplus import OptimizerPlus
        control = OptimizerPlus
        # General serial execution handles this example's padding and token weighting.
        options = dict(audition='checkpoint_continue', acceleration='auto',
                       loss_reduction='token_mean', microbatch_execution='serial')
    trainer = TrainerPlus(model=model, optimizer=optimizer, tokenizer=tokenizer,
        train_dataset=train_data, eval_dataset=validation, task=args.task,
        control=control, args=config, data_config=data_config, control_options=options)
    result = trainer.train()
    print(json.dumps(dict(status=result.status, updates=result.committed_steps,
        artifacts=str(result.run_dir), saved_model=str(result.model_path), metrics=result.metrics), indent=2))
    return result


if __name__ == '__main__':
    main()
