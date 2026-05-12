"""Start MiniMind API server with MinCode's LoRA-merged weights.

Usage:
    python scripts/start_minimind_server.py                      # use mincode_sft weights
    python scripts/start_minimind_server.py --weight full_sft    # use base weights (for comparison)
"""

import sys
import os
import argparse

import torch

MINIMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../minimind"))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, MINIMIND_DIR)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Start MiniMind API server for MinCode")
    parser.add_argument("--weight", default="mincode_sft", type=str,
                        help="Weight name: mincode_sft (LoRA-merged) or full_sft (base)")
    parser.add_argument("--port", default=8998, type=int)
    args = parser.parse_args()

    # Determine weight path
    if args.weight == "full_sft":
        weight_path = os.path.join(MINIMIND_DIR, "out", "full_sft_768.pth")
    else:
        weight_path = os.path.join(PROJECT_DIR, "out", f"{args.weight}_768.pth")

    device = "cpu"
    print(f"Loading model: {args.weight}")
    print(f"Weight path: {weight_path}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MINIMIND_DIR, "model"))
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model = model.float().eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {param_count:.2f}M parameters, device={device}")

    # Import and configure the FastAPI app from serve_openai_api.py
    os.chdir(os.path.join(MINIMIND_DIR, "scripts"))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serve_openai_api",
        os.path.join(MINIMIND_DIR, "scripts", "serve_openai_api.py"),
    )
    server_module = importlib.util.module_from_spec(spec)

    # Inject model/tokenizer/device before exec to skip __main__ block
    server_module.model = model
    server_module.tokenizer = tokenizer
    server_module.device = device
    server_module.__name__ = "serve_openai_api"
    spec.loader.exec_module(server_module)

    # Re-inject after exec (module init may overwrite)
    server_module.model = model
    server_module.tokenizer = tokenizer
    server_module.device = device

    import uvicorn
    print(f"Starting server on port {args.port}...")
    uvicorn.run(server_module.app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
