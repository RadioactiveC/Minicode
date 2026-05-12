"""MinCode Agentic RL Training (GRPO/CISPO).

Adapted from MiniMind's train_agent.py for MinCode's 3-tool setup.
Full-parameter training with reference model and mock tool execution.

Usage:
    python scripts/train_rl.py --from_weight mincode_v1    # RL on v1 SFT
    python scripts/train_rl.py --from_weight mincode_v2    # RL on v2 SFT
"""

import os
import sys
import re
import gc
import json
import math
import random
import argparse
import warnings

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

MINIMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../minimind'))
sys.path.insert(0, MINIMIND_DIR)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ================================ Tools & Mock Execution ================================

TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "列出目录中的文件和子目录", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}}, "required": []}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "写入文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "写入内容"}}, "required": ["path", "content"]}}},
]

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

# Mock execution results
MOCK_RESULTS = {
    "list_files": lambda args: {
        "files": ["README.md", "main.py", "setup.py", "requirements.txt", ".gitignore"],
        "path": args.get("path", ".")
    },
    "read_file": lambda args: {
        "content": f"# File: {args.get('path', 'unknown')}\n\nSample file content.",
        "path": args.get("path", "unknown")
    },
    "write_file": lambda args: {
        "status": "ok",
        "path": args.get("path", "unknown"),
        "bytes_written": len(args.get("content", ""))
    },
}

# Argument validation
CHECK_ARGS = {
    "list_files": lambda a: True,  # path is optional
    "read_file": lambda a: bool(a.get("path")),
    "write_file": lambda a: bool(a.get("path")) and a.get("content") is not None,
}


# ================================ Parsing & Execution ================================

def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        return fn(args)
    except Exception:
        return None


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# ================================ Dataset ================================

class MinCodeRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
                del message["tools"]  # Remove so it doesn't interfere with chat template
            messages.append(message)
        # Drop the last empty assistant message (placeholder for generation)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        indices = list(self.sampler) if hasattr(self.sampler, '__iter__') else self.sampler
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        for batch in batches[self.skip_batches:]:
            yield batch

    def __len__(self):
        indices = list(self.sampler) if hasattr(self.sampler, '__iter__') else self.sampler
        total = (len(indices) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)


# ================================ Per-token log-probs ================================

def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    """Compute per-token log probabilities for the last n_keep tokens."""
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    # logits shape: [B, total-1, vocab]
    # We only need logits at positions that predict the last n_keep tokens,
    # i.e., logits[:, -n_keep:, :] predicting input_ids[:, -n_keep:]
    logits = logits[:, -n_keep:, :]  # [B, n_keep, vocab]
    target_ids = input_ids[:, -n_keep:]  # [B, n_keep]
    per_token_logps = torch.gather(
        logits.log_softmax(dim=-1), 2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, n_keep]
    return per_token_logps


# ================================ Rollout Engine (simplified) ================================

def rollout_single(model, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, device="cpu"):
    """Multi-turn rollout: generate -> parse tool calls -> mock execute -> continue."""
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    unfinished = False

    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=False
        )
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()

        if prompt_ids is None:
            prompt_ids = context_ids

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][len(context_ids):].tolist()
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Compute log-probs for new tokens
        full_mask = (output_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            logps = compute_per_token_logps(model, output_ids, len(new_ids), attention_mask=full_mask)
        new_logps = logps[0].tolist()

        # Filter pad/eos
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]

        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)

        # Check for tool calls
        calls = parse_tool_calls(new_text)
        if not calls:
            break

        unfinished = (turn == max_turns - 1)
        messages.append({"role": "assistant", "content": new_text})

        for call in calls:
            name = call.get("name", "")
            raw = call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}'
            result_str = result_str[:2048]
            messages.append({"role": "tool", "content": result_str})

        # Encode the observation (tool results) — these tokens are NOT trained on
        observe_context = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not unfinished,
            tools=tools, open_thinking=False
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished


def rollout_batch(model, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, device="cpu"):
    all_completions = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []

    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = \
                rollout_single(model, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, device)
            all_completions.append(completion)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)

    return all_completions, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished


# ================================ Reward Calculation ================================

def match_args_score(pred_args, gt_args):
    """Score argument value match against ground truth.

    Returns:
      1.0 — all gt keys present and values match
      0.5 — partial match (some keys match)
      0.0 — no match
    """
    if not gt_args:
        return 1.0  # No gt args to check, auto pass
    if not isinstance(pred_args, dict):
        return 0.0

    matched = 0
    total = len(gt_args)
    for key, gt_val in gt_args.items():
        pred_val = pred_args.get(key)
        if pred_val is None:
            continue
        # Normalize to string for comparison
        if str(pred_val).strip().lower() == str(gt_val).strip().lower():
            matched += 1
        elif str(gt_val).strip().lower() in str(pred_val).strip().lower():
            matched += 0.5  # Partial: gt value contained in prediction
    return matched / total if total > 0 else 1.0


def calculate_rewards(completions, gt_batch, num_gen, turn_outputs_batch, unfinished_batch, device="cpu"):
    """MinCode-specific reward function.

    Reward components:
      - Tool name correct: +2.0
      - JSON arguments valid: +1.0
      - Required args present: +1.0
      - Argument values match gt: +1.0 (full) / +0.5 (partial)
      - No false trigger (no_tool correct): +2.0
      - Wrong tool / false trigger: -1.0
      - Repetition penalty: up to -0.5
      - Unfinished penalty: -0.5
    """
    rewards = torch.zeros(len(completions), device=device)

    for idx, response in enumerate(completions):
        reward = 0.0
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        gt_tool = gt.get("tool") if gt else None
        gt_args = gt.get("args", {}) if gt else {}
        turn_outputs = turn_outputs_batch[idx]
        unfinished = unfinished_batch[idx]

        # Parse all tool calls across turns
        tool_calls = []
        for turn in turn_outputs:
            tool_calls.extend(parse_tool_calls(turn))

        if gt_tool is None:
            # Should NOT call any tool
            if not tool_calls:
                reward += 2.0  # Correct: no false trigger
                # Bonus for reasonable response length
                if 5 <= len(response.strip()) <= 500:
                    reward += 0.5
            else:
                reward -= 1.0  # False trigger
                reward -= 0.5 * len(tool_calls)  # Extra penalty per false call
        else:
            # Should call a specific tool
            if not tool_calls:
                reward -= 1.0  # Missed tool call
            else:
                first_call = tool_calls[0]
                name = first_call.get("name", "")
                raw_args = first_call.get("arguments", {})

                # Parse args if string
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = None

                # Tool name check
                if name == gt_tool:
                    reward += 2.0  # Correct tool

                    # JSON args valid check
                    if isinstance(raw_args, dict):
                        reward += 1.0  # Valid JSON args

                        # Required args check
                        check_fn = CHECK_ARGS.get(name)
                        if check_fn and check_fn(raw_args):
                            reward += 1.0  # Required args present

                        # Argument values match gt
                        reward += match_args_score(raw_args, gt_args)

                    # else: args not valid dict, no bonus
                elif name in VALID_TOOL_NAMES:
                    reward -= 1.0  # Wrong tool (but valid name)
                else:
                    reward -= 1.5  # Invalid tool name

                # Extra tool calls penalty
                if len(tool_calls) > 1:
                    reward -= 0.5 * (len(tool_calls) - 1)

        # Unfinished penalty
        if unfinished:
            reward -= 0.5

        # Repetition penalty
        reward -= rep_penalty(response)

        # Clip to [-3, 5]
        rewards[idx] = max(min(reward, 5.0), -3.0)

    return rewards


# ================================ Training Loop ================================

def rl_train_epoch(epoch, loader, iters, model, ref_model, tokenizer, optimizer, scheduler, args):
    for step, batch in enumerate(loader, start=1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']

        # ---- Rollout ----
        model.eval()
        with torch.no_grad():
            completions, prompt_ids_batch, response_ids_batch, response_masks_batch, \
                response_old_logps_batch, turn_outputs_batch, unfinished_batch = \
                rollout_batch(model, tokenizer, messages_batch, tools_batch,
                              args.num_generations, max_turns=3,
                              max_new_tokens=args.max_gen_len, device=args.device)
        model.train()

        # ---- Pack sequences ----
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch,
                                    response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            packed_samples.append((ids, mask, old_logps))

        max_len = max(len(ids) for ids, _, _ in packed_samples)
        input_ids = torch.tensor(
            [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _ in packed_samples],
            device=args.device
        )
        full_response_masks = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for _, mask, _ in packed_samples],
            device=args.device, dtype=torch.float32
        )
        old_per_token_logps = torch.tensor(
            [lps + [0.0] * ((max_len - 1) - len(lps)) for _, _, lps in packed_samples],
            device=args.device, dtype=torch.float32
        )
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        # ---- Rewards ----
        rewards = calculate_rewards(
            completions, gt_batch, args.num_generations,
            turn_outputs_batch, unfinished_batch, device=args.device
        )

        # ---- Policy log-probs (current) ----
        res = model(input_ids, attention_mask=full_mask)
        logits = res.logits[:, :-1, :]
        per_token_logps = F.log_softmax(logits, dim=-1).gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # ---- Reference log-probs ----
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(
                ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask
            )

        # ---- Completion mask (only train on model-generated tokens) ----
        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1,
                             device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        # ---- Advantages (GRPO group normalization) ----
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ---- Loss ----
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                               - args.beta * per_token_kl)
        else:  # grpo
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                               - args.beta * per_token_kl)

        policy_loss = (
            ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] /
             token_counts[valid_rows].clamp(min=1)).mean()
            if valid_rows.any()
            else per_token_loss.sum() * 0.0
        )

        loss = policy_loss / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- Logging ----
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                  f'Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, '
                  f'Loss:{pl:.4f}, AvgLen:{al:.1f}, LR:{lr:.2e}')

            # Debug: show sample completions
            if args.debug_mode:
                for i in range(min(2, len(completions))):
                    short = completions[i].replace('\n', '\\n')[:150]
                    print(f'  [gen {i}] reward={rewards[i].item():.2f} | {short}')

        # ---- Save checkpoint ----
        if (step % args.save_interval == 0 or step == iters):
            model.eval()
            ckp = os.path.join(args.save_dir, f'{args.output_name}_768.pth')
            state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            print(f'  Checkpoint saved: {ckp}')
            model.train()

        # Cleanup
        del per_token_logps, ref_per_token_logps, completions, rewards
        del grouped_rewards, mean_r, std_r, advantages, completion_mask

    # Flush remaining gradients
    if iters > 0 and iters % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


# ================================ Main ================================

def main():
    parser = argparse.ArgumentParser(description="MinCode Agentic RL Training")
    parser.add_argument("--output_name", type=str, default="mincode_v1_rl",
                        help="Output weight name prefix (e.g. mincode_v1_rl -> mincode_v1_rl_768.pth)")
    parser.add_argument("--from_weight", type=str, default="mincode_v1",
                        help="SFT weight to start from (e.g. mincode_v1)")
    parser.add_argument("--weight_dir", type=str, default=None,
                        help="Directory containing weight files (default: PROJECT_DIR/out)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save output weights (default: PROJECT_DIR/out)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="RL dataset path (default: dataset/mincode_rl.jsonl)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (prompts per step)")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt for GRPO")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence penalty coefficient")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon (GRPO)")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="Upper clip bound (CISPO)")
    parser.add_argument("--max_gen_len", type=int, default=256, help="Max new tokens per generation")
    parser.add_argument("--max_total_len", type=int, default=1024, help="Max total sequence length")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=1, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N steps")
    parser.add_argument("--debug_mode", action="store_true", help="Print sample completions")
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="Allow overwriting existing output file (for resumed training)")
    args = parser.parse_args()

    # Resolve paths
    args.device = "cpu"
    args.weight_dir = args.weight_dir or os.path.join(PROJECT_DIR, "out")
    args.save_dir = args.save_dir or os.path.join(PROJECT_DIR, "out")
    args.data_path = args.data_path or os.path.join(PROJECT_DIR, "dataset", "mincode_rl.jsonl")

    # File isolation check: refuse to overwrite existing output
    output_path = os.path.join(args.save_dir, f"{args.output_name}_768.pth")
    if os.path.exists(output_path) and not args.allow_overwrite:
        print(f"ERROR: Output file already exists: {output_path}")
        print("Choose a different --output_name, or use --allow_overwrite for resumed training.")
        sys.exit(1)

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"MinCode Agentic RL Training")
    print(f"  From weight: {args.from_weight}")
    print(f"  Output name: {args.output_name}")
    print(f"  Data: {args.data_path}")
    print(f"  Device: {args.device}")
    print(f"  Loss: {args.loss_type}, beta={args.beta}")
    print(f"  Epochs: {args.epochs}, batch_size={args.batch_size}, num_gen={args.num_generations}")
    print(f"  LR: {args.learning_rate}")
    print()

    # ---- Load model config ----
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MINIMIND_DIR, "model"))

    # ---- Load policy model ----
    weight_path = os.path.join(args.weight_dir, f"{args.from_weight}_768.pth")
    print(f"Loading policy model: {weight_path}")
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    model = model.float().to(args.device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {param_count:.1f}M")

    # ---- Load reference model (frozen copy) ----
    print(f"Loading reference model: {weight_path}")
    ref_model = MiniMindForCausalLM(lm_config)
    ref_model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    ref_model = ref_model.float().eval().to(args.device)
    ref_model.requires_grad_(False)
    print(f"  Reference model frozen.")
    print()

    # ---- Dataset ----
    train_ds = MinCodeRLDataset(args.data_path, tokenizer, max_length=768)
    print(f"Dataset: {len(train_ds)} prompts")

    def collate_fn(batch):
        return {
            'messages': [b['messages'] for b in batch],
            'tools': [b['tools'] for b in batch],
            'gt': [b['gt'] for b in batch],
        }

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_optimizer_steps = math.ceil(iters_per_epoch / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps,
                                   eta_min=args.learning_rate / 10)

    print(f"Steps per epoch: {iters_per_epoch}, total optimizer steps: {total_optimizer_steps}")
    print()

    # ---- Training ----
    model.train()
    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip_batches=0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=0, collate_fn=collate_fn)
        rl_train_epoch(epoch, loader, len(loader), model, ref_model, tokenizer,
                       optimizer, scheduler, args)
        gc.collect()

    # ---- Final save ----
    final_path = os.path.join(args.save_dir, f"{args.output_name}_768.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final weights: {final_path}")


if __name__ == "__main__":
    main()
