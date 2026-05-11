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
              [list_files, read_file, grep_files,
               write_file, replace_in_file, exec_command]
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
- **`tools.py`** — Tool registry + 6 built-in tools for file operations and shell execution. Tools are defined as OpenAI function-calling schema.
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

## Development Phases

1. **Phase 1 (current):** Wire up MiniMind API to the agent harness, verify tool-call round-trip works
2. **Phase 2:** Construct coding-tool SFT data, retrain MiniMind with our tools
3. **Phase 3:** Agentic RL with coding-task rewards
4. **Phase 4:** Add memory system

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
- Model has never seen coding tools (list_files, read_file, etc.) — needs fine-tuning
- Tool call format works but tool selection accuracy will be low without training
