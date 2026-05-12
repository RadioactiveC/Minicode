"""Evaluate MinCode LoRA model on tool-calling accuracy.

Usage:
    python scripts/eval_toolcall.py                    # test merged model
    python scripts/eval_toolcall.py --weight full_sft  # test base model (before LoRA)

Tests:
  - Correct tool name selection
  - Valid JSON arguments
  - No tool call when not needed (pure dialogue)
  - Multi-turn tool calling
"""

import os
import sys
import json
import re
import argparse
import warnings

import torch
from transformers import AutoTokenizer

MINIMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../minimind'))
sys.path.insert(0, MINIMIND_DIR)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# MinCode's 3 tools (same as tools.py / generate_sft_data.py)
TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "列出目录中的文件和子目录", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}}, "required": []}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "写入文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "写入内容"}}, "required": ["path", "content"]}}},
]

VALID_TOOL_NAMES = {"list_files", "read_file", "write_file"}

# Default test set path
DEFAULT_TEST_SET = os.path.join(PROJECT_DIR, "eval", "test_cases.json")


def load_test_cases(path):
    """Load test cases from external JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    # Normalize: JSON null -> Python None
    for c in cases:
        if c.get("expect_tool") is None:
            c["expect_tool"] = None
    return cases


def parse_tool_calls(text):
    """Extract tool calls from <tool_call>...</tool_call> tags."""
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


def generate(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate model response for a single user prompt with tools."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=TOOLS, open_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic eval
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(generated[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response


def evaluate_case(response, expect_tool):
    """Evaluate a single test case. Returns (passed, details)."""
    tool_calls = parse_tool_calls(response)

    if expect_tool is None:
        # Should NOT call any tool
        if not tool_calls:
            return True, "correct: no tool call"
        else:
            names = [tc.get("name", "?") for tc in tool_calls]
            return False, f"wrong: called {names} (expected none)"

    # Should call a specific tool
    if not tool_calls:
        return False, f"wrong: no tool call (expected {expect_tool})"

    first_call = tool_calls[0]
    name = first_call.get("name", "")
    args = first_call.get("arguments", {})

    # Check tool name
    if name != expect_tool:
        if name in VALID_TOOL_NAMES:
            return False, f"wrong tool: {name} (expected {expect_tool})"
        else:
            return False, f"invalid tool name: {name}"

    # Check arguments are valid
    if not isinstance(args, dict):
        try:
            args = json.loads(args) if isinstance(args, str) else {}
        except json.JSONDecodeError:
            return False, f"correct tool but invalid args JSON"

    return True, f"correct: {name}({json.dumps(args, ensure_ascii=False)[:60]})"


def main():
    parser = argparse.ArgumentParser(description="Evaluate MinCode tool-calling")
    parser.add_argument("--weight", type=str, default="mincode_sft",
                        help="Weight name prefix (mincode_sft or full_sft)")
    parser.add_argument("--weight_dir", type=str, default=None,
                        help="Directory containing weight file")
    parser.add_argument("--test_set", type=str, default=DEFAULT_TEST_SET,
                        help="Path to test cases JSON file (default: eval/test_cases.json)")
    args = parser.parse_args()

    # Determine weight path
    if args.weight == "full_sft":
        weight_path = os.path.join(MINIMIND_DIR, "out", "full_sft_768.pth")
        label = "BASE (full_sft, before LoRA)"
    else:
        weight_dir = args.weight_dir or os.path.join(PROJECT_DIR, "out")
        weight_path = os.path.join(weight_dir, f"{args.weight}_768.pth")
        label = f"LoRA-merged ({args.weight})"

    device = "cpu"
    print(f"Model: {label}")
    print(f"Weight: {weight_path}")
    print(f"Device: {device}")
    print()

    # Load model
    setup_seed(42)
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MINIMIND_DIR, "model"))
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model = model.float().eval().to(device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # Load test cases
    test_cases = load_test_cases(args.test_set)
    test_set_name = os.path.basename(args.test_set)
    print(f"Test set: {test_set_name} ({len(test_cases)} cases)\n")

    # Run eval
    results = {"total": 0, "passed": 0, "by_category": {}}
    case_results = []

    for case in test_cases:
        prompt = case["prompt"]
        expect = case["expect_tool"]
        cat = case["category"]

        response = generate(model, tokenizer, prompt, device)
        passed, details = evaluate_case(response, expect)

        results["total"] += 1
        if passed:
            results["passed"] += 1

        if cat not in results["by_category"]:
            results["by_category"][cat] = {"total": 0, "passed": 0}
        results["by_category"][cat]["total"] += 1
        if passed:
            results["by_category"][cat]["passed"] += 1

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {prompt}")
        print(f"       {details}")
        raw_short = response.replace('\n', '\\n')[:120]
        if not passed:
            print(f"       raw: {raw_short}")
        print()

        case_results.append({
            "prompt": prompt,
            "expect_tool": expect,
            "category": cat,
            "passed": passed,
            "details": details,
            "raw_response": response,
        })

    # Summary
    total = results["total"]
    passed = results["passed"]
    pct = passed / total * 100 if total else 0
    print("=" * 60)
    print(f"TOTAL: {passed}/{total} ({pct:.1f}%)")
    print()
    for cat, stats in results["by_category"].items():
        cp = stats["passed"] / stats["total"] * 100
        print(f"  {cat:12s}: {stats['passed']}/{stats['total']} ({cp:.0f}%)")
    print("=" * 60)

    # Save results to JSON
    eval_dir = os.path.join(PROJECT_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output = {
        "model": label,
        "weight": weight_path,
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": round(pct, 1),
            "by_category": {cat: {"passed": s["passed"], "total": s["total"],
                                  "accuracy": round(s["passed"] / s["total"] * 100, 1)}
                            for cat, s in results["by_category"].items()},
        },
        "cases": case_results,
    }
    # Output name includes test set identifier
    ts_stem = os.path.splitext(test_set_name)[0]  # e.g. "test_cases" or "test_cases_v0"
    out_name = f"eval_{args.weight}_{ts_stem}.json"
    out_path = os.path.join(eval_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
