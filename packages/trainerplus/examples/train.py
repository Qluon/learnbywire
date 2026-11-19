"""Local CPT/SFT example. Launch explicitly; importing this file never trains."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Transformers model/tokenizer directory')
    parser.add_argument('--data', required=True, help='Local JSON, JSONL or CSV')
    parser.add_argument('--validation')
    parser.add_argument('--task', choices=['cpt', 'sft'], required=True)
    parser.add_argument('--control', choices=['native', 'optimizerplus'], default='native')
    parser.add_argument('--control-loss-reduction', choices=['token_mean', 'microbatch_mean'],
                        help='OptimizerPlus training loss weighting; defaults to token_mean')
    parser.add_argument('--microbatch-execution', choices=['serial', 'parallel'],
                        help='OptimizerPlus execution; parallel requires dense structural LoRA, microbatch 1 and accumulation 2')
    parser.add_argument('--microbatches-per-replay', type=int,
                        help='OptimizerPlus graph grouping; omit to retain automatic paired execution')
    parser.add_argument('--cuda-alloc-conf',
                        help='PyTorch allocator setting applied before CUDA initialization, e.g. expandable_segments:True')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', default='runs/trainerplus')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--padding', choices=['fixed', 'dynamic'], default='fixed')
    parser.add_argument('--truncate', action='store_true', help='Explicitly permit truncation')
    parser.add_argument('--pack', action='store_true', help='Pack CPT documents with EOS boundaries')
    parser.add_argument('--sft-loss', choices=['auto', 'all', 'completion', 'assistant'], default='auto')
    parser.add_argument('--pad-with-eos', action='store_true', help='Explicitly use EOS as the pad token')
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-steps', type=int)
    parser.add_argument('--save-steps', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--lora-rank', type=int, default=0)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', help='Comma-separated module names, explicitly chosen for this model')
    opts = parser.parse_args()
    if opts.control == 'optimizerplus' and opts.device != 'cuda':
        parser.error('OptimizerPlus 1.2.2 requires CUDA')
    if opts.control_loss_reduction is not None and opts.control != 'optimizerplus':
        parser.error('--control-loss-reduction requires --control optimizerplus')
    if opts.microbatch_execution is not None and opts.control != 'optimizerplus':
        parser.error('--microbatch-execution requires --control optimizerplus')
    if opts.microbatch_execution == 'parallel' and not (
            opts.microbatch == 1 and opts.accumulation == 2 and opts.lora_rank > 0
            and opts.padding == 'fixed' and opts.control_loss_reduction == 'microbatch_mean'
            and opts.microbatches_per_replay is None):
        parser.error('parallel requires LoRA, fixed padding, microbatch 1, accumulation 2, microbatch_mean, and no replay-group override')
    if opts.microbatches_per_replay is not None:
        if opts.control != 'optimizerplus' or opts.microbatches_per_replay < 1:
            parser.error('--microbatches-per-replay requires OptimizerPlus and a positive integer')
    if opts.cuda_alloc_conf is not None:
        if opts.device != 'cuda' or not opts.cuda_alloc_conf.strip():
            parser.error('--cuda-alloc-conf requires CUDA and a nonempty configuration')
        loaded_torch = sys.modules.get('torch')
        if loaded_torch is not None and loaded_torch.cuda.is_initialized():
            parser.error('Set --cuda-alloc-conf in a fresh process before CUDA initialization')
        os.environ['PYTORCH_ALLOC_CONF'] = opts.cuda_alloc_conf
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    if opts.lora_rank and not opts.lora_targets:
        parser.error('--lora-rank requires model-appropriate --lora-targets')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trainerplus import TrainerPlus, TrainingConfig, DataConfig
    from trainerplus.artifacts import save, sha_file

    # Preserve the outer script/config before model loading or training begins.
    launch = Path(opts.output)/('launch_'+uuid.uuid4().hex[:12])
    launch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, launch/'train.py')
    save(launch/'launch.json', dict(arguments=vars(opts), script_sha256=sha_file(__file__),
        cuda_allocator_configuration=os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
        train_data_sha256=sha_file(opts.data),
        validation_data_sha256=sha_file(opts.validation) if opts.validation else None))
    set_seed(opts.seed)
    tokenizer = AutoTokenizer.from_pretrained(opts.model, local_files_only=True)
    if opts.pad_with_eos:
        tokenizer.pad_token = tokenizer.eos_token
    precision = torch.bfloat16 if opts.device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(opts.model, local_files_only=True, dtype=precision)
    model.config.use_cache = False
    if opts.lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=opts.lora_rank, lora_alpha=opts.lora_alpha,
            lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
            target_modules=opts.lora_targets.split(',')))
    model.to(opts.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=opts.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
        fused=opts.device == 'cuda', foreach=False)
    control = None
    if opts.control == 'optimizerplus':
        from optimizerplus import OptimizerPlus
        control = OptimizerPlus
    control_options = {}
    if opts.microbatch_execution is not None:
        control_options['microbatch_execution'] = opts.microbatch_execution
    if opts.control_loss_reduction is not None:
        control_options['loss_reduction'] = opts.control_loss_reduction
    if opts.microbatches_per_replay is not None:
        control_options['microbatches_per_replay'] = opts.microbatches_per_replay
    trainer = TrainerPlus(model=model, optimizer=optimizer, tokenizer=tokenizer,
        task=opts.task, control=control, train_dataset=opts.data, eval_dataset=opts.validation,
        control_options=control_options or None,
        args=TrainingConfig(output_dir=launch, num_epochs=opts.epochs, max_steps=opts.max_steps,
            microbatch_size=opts.microbatch, gradient_accumulation_steps=opts.accumulation,
            mixed_precision='bf16' if opts.device == 'cuda' else 'no', seed=opts.seed,
            lr_schedule=opts.schedule, warmup_ratio=opts.warmup_ratio, drop_last=opts.drop_last,
            eval_steps=opts.eval_steps, save_steps=opts.save_steps),
        data_config=DataConfig(max_length=opts.length, padding=opts.padding,
            truncation='truncate' if opts.truncate else 'error', packing=opts.pack, sft_loss=opts.sft_loss))
    result = trainer.train(resume_from_checkpoint=opts.resume, stop_after_steps=opts.stop_after_steps)
    print(f'{result.status}: {result.committed_steps} committed updates; artifacts: {result.run_dir}')


if __name__ == '__main__':
    main()
