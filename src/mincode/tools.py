"""Tool system: registry, sandbox, and built-in coding tools."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ApprovalCallback = Callable[[str, str], bool]


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


@dataclass
class ToolContext:
    """Shared mutable state for tool handlers."""

    cwd: Path
    ask_approval: ApprovalCallback
    auto_approve: bool = False

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()

    def resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (self.cwd / candidate).resolve()
        if not _is_within(self.cwd, resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


ToolHandler = Callable[[ToolContext, dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    mutating: bool = False


class ToolRegistry:
    """Runtime registry for callable tools."""

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._tools)]

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.list_tools()
        ]

    def execute(self, name: str, arguments_json: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                return "Tool arguments must be a JSON object."
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"

        if tool.mutating and not self.context.auto_approve:
            preview = json.dumps(arguments, ensure_ascii=False, indent=2)[:600]
            if not self.context.ask_approval(name, preview):
                return "Rejected by user."

        try:
            return tool.handler(self.context, arguments)
        except Exception as exc:
            return f"Tool error: {exc}"


# ── Built-in tool handlers ────────────────────────────────────────


def _list_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args.get("path", "."))
    recursive = bool(args.get("recursive", True))
    max_entries = max(1, min(int(args.get("max_entries", 200)), 2000))
    root = ctx.resolve_path(raw_path)
    if not root.exists():
        return f"path does not exist: {root}"
    if root.is_file():
        return str(root.relative_to(ctx.cwd))

    entries: list[str] = []
    iterator = sorted(root.rglob("*")) if recursive else sorted(root.iterdir())
    for item in iterator:
        rel = item.relative_to(ctx.cwd)
        entries.append(f"{rel}/" if item.is_dir() else str(rel))
        if len(entries) >= max_entries:
            entries.append(f"... truncated at {max_entries} entries")
            break
    return "\n".join(entries) if entries else "(empty directory)"


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    start_line = max(1, int(args.get("start_line", 1)))
    num_lines = max(1, min(int(args.get("num_lines", 200)), 2000))
    path = ctx.resolve_path(raw_path)
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[start_line - 1 : start_line - 1 + num_lines]
    if not chunk:
        return "(no content)"
    return "\n".join(f"{i:>6}\t{text}" for i, text in enumerate(chunk, start=start_line))


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    content = str(args.get("content", ""))
    mode = str(args.get("mode", "overwrite"))
    if mode not in {"overwrite", "append"}:
        return "mode must be: overwrite or append"
    path = ctx.resolve_path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "overwrite":
        path.write_text(content, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
    return f"wrote {len(content)} bytes to {path.relative_to(ctx.cwd)} ({mode})"


def _replace_in_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    old = str(args["old"])
    new = str(args["new"])
    path = ctx.resolve_path(raw_path)
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    content = path.read_text(encoding="utf-8", errors="replace")
    if old not in content:
        return "no match found — old text does not exist in file"
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    return f"replaced in {path.relative_to(ctx.cwd)}"


def _grep_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    pattern = str(args["pattern"])
    raw_path = str(args.get("path", "."))
    max_results = max(1, min(int(args.get("max_results", 200)), 2000))
    root = ctx.resolve_path(raw_path)

    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", pattern, str(root)]
        proc = subprocess.run(cmd, cwd=ctx.cwd, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            return f"rg failed: {proc.stderr.strip()}"
        lines = proc.stdout.splitlines()[:max_results]
        return "\n".join(lines) if lines else "(no matches)"

    matches: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                matches.append(f"{p.relative_to(ctx.cwd)}:{line_no}:{line}")
                if len(matches) >= max_results:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"


def _exec_command(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = str(args["command"])
    timeout = max(1, min(int(args.get("timeout_sec", 120)), 600))
    proc = subprocess.run(
        command, shell=True, cwd=ctx.cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(output) > 6000:
        output = output[:6000] + "\n... (truncated)"
    return f"exit_code={proc.returncode}\n{output}"


# ── Registry factory ──────────────────────────────────────────────


def create_default_tools(context: ToolContext) -> ToolRegistry:
    """Build the 6 built-in coding tools.

    Tool descriptions and schemas are kept minimal to fit within MiniMind's
    limited context window (~768 tokens at SFT training time).
    """
    registry = ToolRegistry(context)

    registry.register(Tool(
        name="list_files",
        description="列出目录中的文件和子目录",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径"},
            },
            "required": [],
        },
        handler=_list_files,
    ))

    registry.register(Tool(
        name="read_file",
        description="读取文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ))

    registry.register(Tool(
        name="grep_files",
        description="在文件中搜索文本",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索文本"},
                "path": {"type": "string", "description": "搜索目录"},
            },
            "required": ["pattern"],
        },
        handler=_grep_files,
    ))

    registry.register(Tool(
        name="write_file",
        description="写入文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
        mutating=True,
    ))

    registry.register(Tool(
        name="replace_in_file",
        description="替换文件中的文本",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old": {"type": "string", "description": "要替换的原文"},
                "new": {"type": "string", "description": "替换后的文本"},
            },
            "required": ["path", "old", "new"],
        },
        handler=_replace_in_file,
        mutating=True,
    ))

    registry.register(Tool(
        name="exec_command",
        description="执行Shell命令",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "命令"},
            },
            "required": ["command"],
        },
        handler=_exec_command,
        mutating=True,
    ))

    return registry
