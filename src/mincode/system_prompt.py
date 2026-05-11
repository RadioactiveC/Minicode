"""System prompt builder."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_system_prompt(cwd: Path, *, skills_overview: str = "(none)", dir_listing: str | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    dir_section = ""
    if dir_listing:
        dir_section = f"- Top-level files:\n{dir_listing}\n"

    # Keep system prompt extremely short — MiniMind was trained with max_seq_len=768,
    # and the chat template already injects tool definitions in <tools> XML tags.
    # Every token in the system prompt is a token less for user input + model output.
    return f"你是MinCode编程助手。工作目录:{cwd}"
