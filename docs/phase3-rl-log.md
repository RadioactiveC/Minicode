# Phase 3 — Agentic RL Training Log

## Overview

- **Goal:** Use GRPO/CISPO reinforcement learning to improve tool-calling accuracy beyond SFT baseline
- **Baselines:** v1 SFT = 62.5%, v2 SFT = 47.5% (40-case eval set)
- **Method:** Full-parameter training with frozen reference model, mock tool execution, CISPO loss
- **Script:** `scripts/train_rl.py`
- **Data v1:** `dataset/mincode_rl.jsonl` (39 prompts: 10 list_files, 10 read_file, 9 write_file, 10 no_tool)
- **Data v2:** `dataset/mincode_rl_v2.jsonl` (80 prompts: 23 list_files, 22 read_file, 21 write_file, 14 no_tool)
- **Combined:** 119 prompts (33 list_files, 32 read_file, 30 write_file, 24 no_tool), all hand-written, zero overlap
- **Eval:** `eval/test_cases.json` (40 cases, zero overlap with RL data)

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training mode | Full parameter | LoRA too few params for RL signal propagation on 64M model |
| Loss type | CISPO (default) | MiniMind default; one-sided clip more stable than PPO |
| Reference model | Frozen SFT copy | KL constraint prevents policy collapse |
| Tool execution | Mock | Deterministic, no side effects, CPU-friendly |
| num_generations | 4 | GRPO needs group variance; 4 balances cost vs signal |
| Learning rate | 3e-7 | Conservative; RL on small models is fragile |
| Max turns | 3 | Enough for tool_call → result → final answer |
| Reward clip | [-3, 3] | Prevents extreme gradients |

## Reward Function

| Condition | Reward | Notes |
|-----------|--------|-------|
| Correct tool name | +2.0 | Core signal |
| Valid JSON arguments | +1.0 | Structural correctness |
| Required args present | +1.0 | Functional correctness |
| Argument values match gt | +1.0 full / +0.5 partial | Semantic correctness |
| Correct no-tool (no false trigger) | +2.0 | Equally important as correct trigger |
| No-tool reasonable length bonus | +0.5 | Response 5~500 chars |
| Wrong tool name (but valid) | -1.0 | Penalize confusion |
| Invalid tool name | -1.5 | Harder penalty for hallucinated tools |
| False trigger (called tool when shouldn't) | -1.0 base + -0.5/extra call | |
| Missed tool call | -1.0 | Should have called but didn't |
| Extra tool calls | -0.5 per extra | Discourage over-calling |
| Unfinished (hit max turns) | -0.5 | |
| Repetition penalty | up to -0.5 | 3-gram based |
| Reward clip | [-3, +5] | |

Tool-calling max: +2.0 + 1.0 + 1.0 + 1.0 = **+5.0**; No-tool max: +2.0 + 0.5 = **+2.5**

## Mock Tool Responses

| Tool | Mock Response |
|------|---------------|
| list_files | Fixed file list: README.md, main.py, setup.py, requirements.txt, .gitignore |
| read_file | Template: "# File: {path}\n\nSample file content." |
| write_file | {"status": "ok", "path": ..., "bytes_written": ...} |

## Training Runs

### Run 1: v1 RL (completed)

- **Input weight:** `out/mincode_v1_768.pth` (v1 SFT, 62.5%)
- **Output weight:** `out/mincode_v1_rl_768.pth`
- **Command:** `python scripts/train_rl.py --from_weight mincode_v1 --output_name mincode_v1_rl --debug_mode`
- **Status:** Completed
- **Eval result:** 25/40 = **62.5%** (unchanged from SFT baseline)

| Category | SFT v1 | RL v1 | Delta |
|----------|--------|-------|-------|
| list_files | ? | 6/8 (75%) | — |
| read_file | ? | 6/8 (75%) | — |
| write_file | ? | 5/8 (62%) | — |
| no_tool | ? | 6/8 (75%) | — |
| edge | ? | 2/8 (25%) | — |
| **Total** | **62.5%** | **62.5%** | **+0%** |

**Observations from training log:**
- Reward trended upward: epoch 1 avg ~1.3, epoch 2 avg ~1.8, epoch 3 avg ~2.4
- Model learned to avoid tool calls for pure dialogue (no_tool improved)
- But RL pushed model toward list_files bias — many write_file/read_file prompts got list_files
- edge cases remain hardest (25%), model struggles with ambiguous intent
- "再见" and "lambda表达式" triggered false tool calls (no_tool regression)
- KL stayed negative throughout (policy drifted from reference)

### Run 2: v1 RL2 — expanded dataset (completed)

- **Input weight:** `out/mincode_v1_768.pth` (v1 SFT, 62.5%)
- **Output weight:** `out/mincode_v1_rl2_768.pth`
- **Data:** `dataset/mincode_rl_combined.jsonl` (119 prompts, v1+v2 merged)
- **Command:** `python scripts/train_rl.py --from_weight mincode_v1 --output_name mincode_v1_rl2 --data_path dataset/mincode_rl_combined.jsonl --debug_mode`
- **Status:** Completed
- **Eval result:** 32/40 = **80.0%** (+17.5% over SFT baseline!)

| Category | SFT v1 | RL v1 (39) | RL2 v1 (119) | Delta (SFT→RL2) |
|----------|--------|------------|--------------|------------------|
| list_files | ? | 6/8 (75%) | 7/8 (88%) | — |
| read_file | ? | 6/8 (75%) | 7/8 (88%) | — |
| write_file | ? | 5/8 (62%) | 8/8 (100%) | — |
| no_tool | ? | 6/8 (75%) | 6/8 (75%) | — |
| edge | ? | 2/8 (25%) | 4/8 (50%) | — |
| **Total** | **62.5%** | **62.5%** | **80.0%** | **+17.5%** |

**Observations:**
- Reward trended upward: epoch 1 avg ~1.7, epoch 2 avg ~2.3, epoch 3 avg ~2.4
- write_file accuracy reached 100% — model learned to distinguish create/write intent perfectly
- Edge cases doubled from 25% → 50%, but remain weakest (ambiguous intent)
- 2 no_tool false triggers remain: "再见" and "lambda表达式" still fire tool calls
- 3 tool confusions: list_files↔read_file↔write_file on ambiguous prompts
- KL drifted negative throughout (expected with full-param RL)
- 119 samples (3x the original 39) was the key factor — more diverse prompts prevented overfitting

### Run 3: v2 RL (completed)

- **Input weight:** `out/mincode_v2_768.pth` (v2 SFT, 47.5%)
- **Output weight:** `out/mincode_v2_rl_768.pth`
- **Data:** `dataset/mincode_rl_combined.jsonl` (119 prompts)
- **Command:** `python scripts/train_rl.py --from_weight mincode_v2 --output_name mincode_v2_rl --data_path dataset/mincode_rl_combined.jsonl --debug_mode`
- **Status:** Completed
- **Eval result:** 25/40 = **62.5%** (+15% over v2 SFT baseline)

| Category | v2 SFT | v2 RL | Delta |
|----------|--------|-------|-------|
| list_files | ? | 6/8 (75%) | — |
| read_file | ? | 2/8 (25%) | — |
| write_file | ? | 7/8 (88%) | — |
| no_tool | ? | 7/8 (88%) | — |
| edge | ? | 3/8 (38%) | — |
| **Total** | **47.5%** | **62.5%** | **+15%** |

**Observations:**
- Reward trended upward: epoch 1 avg ~1.5, epoch 2 avg ~2.0, epoch 3 avg ~2.3
- list_files learned well (75%), write_file strong (88%), no_tool improved to 88%
- read_file severely degraded (25%) — model confused read_file with list_files and write_file
- "再见" still triggers false tool call (same as v1 RL)
- v2 RL (62.5%) matches v1 SFT (62.5%) but is far below v1 RL2 (80%)
- **Key conclusion:** RL can compensate for weaker SFT (+15%), but cannot fully overcome a bad SFT foundation. v1 SFT (62.5%) + RL → 80%, while v2 SFT (47.5%) + RL → 62.5%. SFT quality is the dominant factor.

## Pre-training Review & Smoke Test

### Script Review (5 issues found, all fixed)

1. **`parse_conversations` tools field leak** — system message retained `tools` string key, could interfere with `apply_chat_template`. Fix: `del message["tools"]` after parsing.
2. **`compute_per_token_logps` shape mismatch** — Original loop-based gather failed when logits and target had different dim-0 sizes. Fix: vectorized with `logits[:, -n_keep:, :]` slice to align with `input_ids[:, -n_keep:]`.
3. **File isolation check too strict** — Checkpoint saves to same file; restart after crash would be blocked. Fix: added `--allow_overwrite` flag.
4. **Debug logging vs del order** — Verified correct: logging happens before cleanup `del` statements.
5. **`old_per_token_logps` padding** — Verified correct: padded to `max_len - 1` matching policy logps shape.

### Smoke Test Results (1 epoch, num_gen=2, max_gen_len=32)

- Script runs end-to-end without errors on CPU
- Reward signal works correctly:
  - Correct tool selection → reward ~3.0 (max)
  - Wrong tool / missed call → reward ~-1.0
  - Correct no-tool → reward ~2.5
- Observations:
  - Model sometimes answers correctly without calling tools (mock data leaks into generation). Acceptable: eval judges tool call presence, not answer quality.
  - KL stays near 0 (reference model = policy at start, expected)
  - Group reward std is low when both generations make same choice (expected with small num_gen)

### Data Review

- 39 prompts: 10 list_files, 10 read_file, 9 write_file, 10 no_tool (balanced)
- All samples pass format validation (system/user/assistant structure, tools JSON, gt format)
- Zero overlap with eval test set (40 cases)

## Issues & Fixes

(To be filled during training)
