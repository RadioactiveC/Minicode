"""Train LoRA on MiniMind for MinCode's 3 coding tools.

Usage:
    python scripts/train_lora.py

This script wraps minimind's LoRA training pipeline with MinCode-specific defaults.
Base weight: full_sft_768.pth (dense, 64M params)
Data: dataset/mincode_sft.jsonl (272 samples)
Output: out/lora_mincode_768.pth (~779KB LoRA weights)
"""

import os
import sys
import time
import warnings
import argparse

import torch
from torch import optim
from contextlib import nullcontext

# Add minimind to path
MINIMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../minimind'))
sys.path.insert(0, MINIMIND_DIR)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, save_lora, merge_lora
from dataset.lm_dataset import SFTDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_lr(current_step, total_steps, lr):
    """Cosine learning rate schedule with warmup."""
    warmup_steps = int(total_steps * 0.1)
    if current_step < warmup_steps:
        return lr * current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))


def train():
    parser = argparse.ArgumentParser(description="Train LoRA for MinCode")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(PROJECT_DIR, "dataset/mincode_sft.jsonl"))
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "out"))
    parser.add_argument("--minimind_weights", type=str,
                        default=os.path.join(MINIMIND_DIR, "out"))
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base weights after training")
    parser.add_argument("--resume_lora", type=str, default="",
                        help="Path to existing LoRA weights to resume training from")
    parser.add_argument("--output_name", type=str, default="mincode",
                        help="Output name prefix (e.g. 'mincode_v1' -> lora_mincode_v1_768.pth)")
    args = parser.parse_args()

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Model config (minimind-3 dense defaults)
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)

    # Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MINIMIND_DIR, "model"))
    model = MiniMindForCausalLM(lm_config)
    weight_path = os.path.join(args.minimind_weights, "full_sft_768.pth")
    print(f"Loading base weights: {weight_path}")
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)

    # Apply LoRA (rank=16, only on square Linear layers)
    apply_lora(model)

    # Resume from existing LoRA weights if specified
    if args.resume_lora:
        print(f"Loading existing LoRA weights: {args.resume_lora}")
        lora_state = torch.load(args.resume_lora, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print(f"  Loaded {len(lora_state)} LoRA tensors")

    # Stats
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
    print(f"Total params: {total_params / 1e6:.3f}M")
    print(f"LoRA params:  {lora_params_count / 1e6:.3f}M ({lora_params_count / total_params * 100:.2f}%)")

    # Freeze non-LoRA params
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # Dataset
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"Training samples: {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    # Optimizer + scaler
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    # MPS doesn't support GradScaler, CPU doesn't need it
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = nullcontext() if not use_amp else torch.cuda.amp.autocast(dtype=torch.float16)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    lora_save_path = os.path.join(args.save_dir, f"lora_{args.output_name}_768.pth")
    total_steps = args.epochs * len(loader)
    global_step = 0

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps/epoch: {len(loader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Save to: {lora_save_path}")
    print()

    model.train()
    best_loss = float('inf')
    loss_log = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        start_time = time.time()

        for step, (input_ids, labels) in enumerate(loader, 1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            global_step += 1

            lr = get_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss
                if res.aux_loss is not None:
                    loss = loss + res.aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item()
            epoch_loss += current_loss
            epoch_steps += 1

            if step % args.log_interval == 0 or step == len(loader):
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                eta = elapsed / step * (len(loader) - step)
                print(f"  Epoch {epoch+1}/{args.epochs} [{step}/{len(loader)}] "
                      f"loss={current_loss:.4f} avg={avg_loss:.4f} lr={lr:.6f} "
                      f"eta={eta:.0f}s")

        avg_epoch_loss = epoch_loss / epoch_steps
        loss_log.append(avg_epoch_loss)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} done. avg_loss={avg_epoch_loss:.4f} time={elapsed:.1f}s")

        # Save best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.eval()
            save_lora(model, lora_save_path)
            print(f"  -> Saved best LoRA to {lora_save_path}")
            model.train()

    # Final save (always)
    model.eval()
    final_path = os.path.join(args.save_dir, f"lora_{args.output_name}_768_final.pth")
    save_lora(model, final_path)
    print(f"\nTraining complete. Final LoRA saved to {final_path}")
    print(f"Best LoRA (loss={best_loss:.4f}) saved to {lora_save_path}")
    print(f"Loss curve: {[f'{l:.4f}' for l in loss_log]}")

    # Merge if requested
    if args.merge:
        merged_path = os.path.join(args.save_dir, f"{args.output_name}_768.pth")
        print(f"\nMerging LoRA into base weights -> {merged_path}")
        # Reload base model for clean merge
        base_model = MiniMindForCausalLM(lm_config)
        base_model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        apply_lora(base_model)
        merge_lora(base_model, lora_save_path, merged_path)
        print(f"Merged model saved to {merged_path}")
        merged_size = os.path.getsize(merged_path) / 1e6
        print(f"Merged model size: {merged_size:.1f}MB")


if __name__ == "__main__":
    train()
