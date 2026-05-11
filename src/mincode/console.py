"""Terminal I/O: Rich console + prompt_toolkit input."""

from __future__ import annotations

import json
from collections.abc import Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# ── Output helpers ────────────────────────────────────────────────


def print_agent(text: str) -> None:
    console.print(Panel(Markdown(text), title="mincode", border_style="green"))


def print_error(text: str) -> None:
    console.print(f"[bold red]error>[/bold red] {text}")


def print_status(text: str) -> None:
    if text:
        console.print(f"[dim]{text}[/dim]")


# ── Spinner ───────────────────────────────────────────────────────

_spinner = None


def set_status(text: str) -> None:
    global _spinner
    if _spinner is not None:
        _spinner.__exit__(None, None, None)
        _spinner = None
    if text:
        _spinner = console.status(f"[bold green]{text}[/bold green]")
        _spinner.__enter__()


def ask_approval(action: str, preview: str) -> bool:
    console.print(Panel(preview, title=f"[yellow]approve: {action}[/yellow]", border_style="yellow"))
    answer = console.input("[yellow]Approve? [y/N] [/yellow]").strip().lower()
    return answer in {"y", "yes"}


# ── Event display ─────────────────────────────────────────────────


def print_tool_call(name: str, arguments_json: str) -> None:
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        args = {}
    # Show the most interesting argument
    key_arg = ""
    if isinstance(args, dict):
        for key in ("path", "command", "pattern", "content"):
            if key in args and isinstance(args[key], str):
                key_arg = args[key][:60]
                break
    suffix = f" [dim]({key_arg})[/dim]" if key_arg else ""
    console.print(f"  [bold blue]> {name}[/bold blue]{suffix}")


def print_tool_result(name: str, result: str) -> None:
    is_error = result.startswith(("blocked", "Rejected", "Unknown tool", "Tool error"))
    marker = "[red]x[/red]" if is_error else "[green]ok[/green]"
    lines = [l.strip() for l in result.splitlines() if l.strip() and not l.strip().startswith("exit_code=")][:3]
    preview = " | ".join(lines) if lines else "(empty)"
    if len(preview) > 120:
        preview = preview[:117] + "..."
    console.print(f"  {marker} [dim]{preview}[/dim]")


def print_thinking(text: str) -> None:
    lines = text.strip().splitlines()
    shown = [*lines[:3], "...", *lines[-2:]] if len(lines) > 6 else lines
    console.print(f"[dim italic]{'chr(10)'.join(shown)}[/dim italic]")


# ── Input session ─────────────────────────────────────────────────


class _SlashCompleter(Completer):
    def __init__(self, names: Sequence[str]) -> None:
        self._names = sorted(names)

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/") or " " in text:
            return
        token = text[1:]
        for name in self._names:
            if name.startswith(token):
                yield Completion(f"/{name}", start_position=-len(text))


class InputSession:
    def __init__(self, command_names: Sequence[str] = ()) -> None:
        kb = KeyBindings()

        @kb.add("escape", "enter", eager=True)
        @kb.add("c-j", eager=True)
        def _newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        self._session: PromptSession[str] = PromptSession(
            message=FormattedText([("bold ansibrightcyan", "you> ")]),
            prompt_continuation=FormattedText([("ansigray", " ... ")]),
            completer=_SlashCompleter(command_names) if command_names else None,
            complete_while_typing=True,
            key_bindings=kb,
            history=InMemoryHistory(),
            multiline=False,
        )

    def prompt(self) -> str:
        with patch_stdout(raw=True):
            return self._session.prompt().strip()
