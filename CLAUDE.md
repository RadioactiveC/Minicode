# MinCode - MiniMind + ApeCode Agent Harness

## Project Overview

MinCode is an educational project that connects the MiniMind small language model (64M params) to a simplified version of the ApeCode terminal coding agent harness. The goal is to create a fully self-contained, trainable coding agent — from model weights to agent loop — that can be understood and reproduced on consumer hardware.

## Architecture

```
User Input
    |
    v
CLI (cli.py) ──> Agent Loop (agent.py)
                      |
                      v
              Model Adapter (model_adapter.py)
                 /          \
    MiniMind API Client    MiniMind Local Inference
    (OpenAI-compat)        (Direct PyTorch)
                      |
                      v
              Tool Execution (tools.py)
              [list_files, read_file, write_file]
                      |
                      v
              Skill Discovery (skills.py)
              [SKILL.md file-based skills]
```

### Core Components

- **`agent.py`** — Main agent loop. Receives user input, calls LLM, parses tool calls, executes tools, feeds results back. Max 20 steps per turn.
- **`model_adapter.py`** — Adapts MiniMind to the `ChatModel` protocol. Two modes:
  - API mode: connects to MiniMind's `serve_openai_api.py` via OpenAI SDK
  - Local mode: loads PyTorch weights directly (future)
- **`tools.py`** — Tool registry + 3 built-in tools (list_files, read_file, write_file). Tools are defined as OpenAI function-calling schema.
- **`skills.py`** — Discovers `SKILL.md` files and makes them available as slash commands.
- **`system_prompt.py`** — Builds the system prompt with tool instructions, workspace context, and AGENTS.md chain.
- **`cli.py`** — Terminal REPL entry point.

## Key Design Decisions

- **Internal message format:** OpenAI Chat Completions format (same as both MiniMind API and ApeCode).
- **Tool call format:** MiniMind uses `<tool_call>{"name":..., "arguments":...}</tool_call>` tags; the API server parses these into structured `tool_calls` objects. The agent consumes the structured format.
- **Simplified scope:** No MCP, no plugins, no subagents, no memory system (for now). Only tools + skills.
- **MiniMind tokenizer:** vocab=6400, special tokens include `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`.

## Reference Projects (sibling directories)

- `../minimind/` — MiniMind model, training pipeline, and inference server
- `../apecode/` — ApeCode agent harness (reference implementation)

## CRITICAL RULE: File Isolation

**所有产生的权重、数据、评测结果文件必须用版本标识独立命名，严禁覆盖！**

命名规范：
- 权重: `out/mincode_v1_768.pth`, `out/mincode_v2_768.pth`, `out/mincode_v1_rl_768.pth`
- LoRA: `out/lora_mincode_v1_768.pth`, `out/lora_mincode_v2_768.pth`
- 数据: `dataset/mincode_sft_v1.jsonl`, `dataset/mincode_sft_v2.jsonl`, `dataset/mincode_rl.jsonl`
- 评测: `eval/eval_v1_sft.json`, `eval/eval_v2_sft.json`, `eval/eval_v1_rl.json`
- 训练脚本: 使用 `--output_name` 参数控制输出前缀

每次训练前检查输出路径是否已存在，若存在则报错而非覆盖。

## Development Phases

1. **Phase 1:** Wire up MiniMind API to the agent harness, verify tool-call round-trip works
2. **Phase 2:** LoRA fine-tune with 3 coding tools (completed)
3. **Phase 2.5:** Data augmentation + retraining experiments (completed)
4. **Phase 3 (current):** Agentic RL with coding-task rewards
5. **Phase 4:** Add memory system

## Progress Log

### Phase 1 — Harness Integration (completed)

- Analyzed minimind + apecode codebases, confirmed OpenAI-format protocol compatibility
- Built MinCode project: agent.py, model_adapter.py, tools.py, skills.py, cli.py, console.py, commands.py, system_prompt.py
- Created `scripts/start_minimind_server.py` — CPU-friendly launcher (float32, importlib injection to bypass .half() and __main__ guard)
- Fixed stream default mismatch: MiniMind defaults `stream=True`, must send `stream=False` explicitly
- Fixed max_tokens dual-purpose bug: MiniMind uses it for both prompt truncation and generation limit, settled on 4096
- Minimized system prompt to 78 chars + Chinese tool descriptions to fit 768 token budget
- End-to-end verified: model calls tools but picks wrong names (e.g., `get_directory` instead of `list_files`) — expected, needs SFT
- Docs: `docs/phase1-report.md`, `docs/communication-deep-dive.md`, `docs/http-bridge-guide.md`

### Phase 2 — LoRA Fine-tuning (completed)

- **Strategy:**
  - Tools: reduced from 6 → 3 (list_files, read_file, write_file) to lower token budget (~400 tokens)
  - Training: LoRA (rank=16, ~0.4M params, 0.61% of total) on minimind-3 dense (64M)
  - Data: 272 samples via DeepSeek V4 Pro (30 seeds + 242 generated), max_seq_len=768
- **Data generation:** `scripts/generate_sft_data.py`
  - read_file: 119 (44%), list_files: 77 (28%), write_file: 58 (21%), no_tool: 59 (22%)
  - Multi-turn: 41 (15%), Chinese/English: ~3:1
  - Output: `dataset/mincode_sft.jsonl`
- **Training:** `scripts/train_lora.py`
  - Local CPU, 5 epochs, ~30min, loss: 1.074 → 0.802
  - Output: `out/lora_mincode_v1_768.pth` (779KB), `out/mincode_v1_768.pth` (131MB merged)
- **Evaluation:** `scripts/eval_toolcall.py`
  - 15-case eval: 40% (base) → 73.3% (LoRA)
  - 40-case eval: 62.5% (more rigorous test set, see Phase 3)
  - Results: `eval/eval_full_sft.json`, `eval/eval_mincode_v1_test_cases.json`
  - Plots: `eval/train_loss.png`, `eval/eval_comparison.png`
- **End-to-end harness test:** all 4 scenarios passed (list/read/write/dialogue)
  - Tool selection: 4/4 correct
  - Arg quality limited by 64M model capacity (expected, Phase 3 target)
### Phase 2.5 — Data Augmentation (completed)

- Generated 210 supplementary samples (`dataset/mincode_sft_v2.jsonl`): write_file 98, no_tool 76, balanced 36
- Incremental training (resume LoRA + new data only) failed: catastrophic forgetting (73.3% → 33.3%)
- Combined retraining (v1+v2 = 482 samples) scored 53.3% — worse than v1 alone (data quality > quantity)
- **Conclusion:** v1 (272 samples) remains best SFT result; new data introduced noise

### Current Weight Files

| File | Description | Accuracy (40-case) |
|------|-------------|----------|
| `out/mincode_v1_768.pth` | v1 SFT merged (272 samples) | 62.5% |
| `out/mincode_v2_768.pth` | v2 SFT merged (482 samples) | 47.5% |
| `out/mincode_v1_rl_768.pth` | v1 RL (39 RL samples) | 62.5% |
| `out/mincode_v1_rl2_768.pth` | v1 RL2 (119 RL samples) | **80.0%** |
| `out/lora_mincode_v1_768.pth` | v1 LoRA weights | — |
| `out/lora_mincode_v2_768.pth` | v2 LoRA weights | — |

### Phase 3 — Agentic RL (in progress)

- **Log:** `docs/phase3-rl-log.md` — detailed training flow, decisions, and run results
- **Script:** `scripts/train_rl.py` — GRPO/CISPO full-parameter RL training
- **Data:** `dataset/mincode_rl.jsonl` (39) + `dataset/mincode_rl_v2.jsonl` (80) → `dataset/mincode_rl_combined.jsonl` (119 prompts)
- **Eval:** `eval/test_cases.json` (40 cases, zero overlap with RL data)
- **Best result:** v1 RL2 = **80.0%** (SFT 62.5% → RL 80.0%, +17.5%)
- **Next:** v2 RL training, multi-turn tool calling exploration

## Conventions

- Python >= 3.10
- Use `pyproject.toml` for project metadata and dependencies
- No heavy frameworks — keep it minimal and readable
- Follow existing code style from apecode (type hints, dataclasses, Protocol)
- Test with `pytest`

## Commands

```bash
# Start MiniMind API server (in ../minimind/)
python scripts/serve_openai_api.py

# Run MinCode agent
python -m mincode

# Run with auto-approve (no confirmation for file writes)
python -m mincode --yolo
```

## Model Limitations to Keep in Mind

- 64M params, vocab=6400 — very limited reasoning capability
- SFT max_seq_len=768 — agent context will easily exceed this
- Single-turn tool calling works well (80%), multi-turn tool calling not yet supported
- Tool call format learned from base SFT; specific tool names learned via LoRA + RL
