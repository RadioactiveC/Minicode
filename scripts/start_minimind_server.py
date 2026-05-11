"""Start MiniMind API server with CPU-compatible settings.

This wrapper patches the model loading to use float32 on CPU
(the original serve_openai_api.py uses .half() which only works on CUDA).
"""

import sys
import os
import argparse

import torch

# Add minimind to path
MINIMIND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../minimind"))
sys.path.insert(0, MINIMIND_DIR)
os.chdir(os.path.join(MINIMIND_DIR, "scripts"))

import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Start MiniMind API server (CPU-friendly)")
    parser.add_argument("--load_from", default="jingyaogong/minimind-3", type=str,
                        help="HuggingFace model name or local path")
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    device = args.device
    print(f"Loading model from {args.load_from}...")
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {param_count:.2f}M parameters")

    # Use float32 on CPU, half on CUDA
    if device == "cpu":
        model = model.float().eval()
    else:
        model = model.half().eval().to(device)

    print(f"Device: {device}, dtype: {next(model.parameters()).dtype}")

    # Import the FastAPI app via importlib (serve_openai_api uses __package__ hack)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serve_openai_api",
        os.path.join(MINIMIND_DIR, "scripts", "serve_openai_api.py"),
    )
    server_module = importlib.util.module_from_spec(spec)

    # Pre-inject globals so the module doesn't try to parse args and init_model on import
    server_module.model = model
    server_module.tokenizer = tokenizer
    server_module.device = device

    # Block the __main__ guard from executing
    server_module.__name__ = "serve_openai_api"
    spec.loader.exec_module(server_module)

    # Now inject again (exec_module may have overwritten)
    server_module.model = model
    server_module.tokenizer = tokenizer
    server_module.device = device

    print(f"Starting server on port {args.port}...")
    uvicorn.run(server_module.app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
