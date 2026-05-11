"""CLI entry point for MinCode."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mincode import __version__
from mincode.agent import Agent, AgentCallbacks
from mincode.commands import CommandRegistry, create_default_commands
from mincode.console import (
    InputSession,
    ask_approval,
    console,
    print_agent,
    print_error,
    print_status,
    print_thinking,
    print_tool_call,
    print_tool_result,
    set_status,
)
from mincode.model_adapter import MiniMindClient, ModelError
from mincode.skills import SkillCatalog
from mincode.system_prompt import build_system_prompt
from mincode.tools import ToolContext, create_default_tools


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mincode {__version__}")
        raise typer.Exit()


def _build_agent(
    *,
    base_url: str,
    model: str,
    max_steps: int,
    timeout: int,
    temperature: float,
    max_tokens: int,
    cwd: Path,
    yolo: bool,
) -> tuple[Agent, CommandRegistry]:
    cwd = cwd.expanduser().resolve()

    tool_context = ToolContext(
        cwd=cwd,
        ask_approval=ask_approval,
        auto_approve=yolo,
    )
    tools = create_default_tools(tool_context)

    skills = SkillCatalog.from_roots([cwd / "skills"])

    # Directory listing for system prompt
    dir_entries: list[str] = []
    try:
        for item in sorted(cwd.iterdir()):
            dir_entries.append(f"{item.name}{'/' if item.is_dir() else ''}")
    except OSError:
        pass
    dir_listing = "\n".join(dir_entries) if dir_entries else None

    system_prompt = build_system_prompt(cwd, skills_overview=skills.format_for_system_prompt(), dir_listing=dir_listing)

    model_client = MiniMindClient(
        base_url=base_url,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    callbacks = AgentCallbacks(
        on_status=set_status,
        on_thinking=print_thinking,
        on_tool_call=print_tool_call,
        on_tool_result=print_tool_result,
    )

    agent = Agent(
        model=model_client,
        tools=tools,
        system_prompt=system_prompt,
        max_steps=max_steps,
        callbacks=callbacks,
    )

    commands = create_default_commands(tools=tools, skills=skills)
    return agent, commands


def _run_repl(agent: Agent, commands: CommandRegistry) -> int:
    print_status("MinCode agent ready. Type /help for commands, /exit to quit.")
    cmd_names = [c.name for c in commands.list_commands()]
    session = InputSession(command_names=cmd_names)

    while True:
        try:
            user_input = session.prompt()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return 0
        if not user_input:
            continue

        # Check for slash command first
        cmd_result = commands.run(user_input)
        if cmd_result is not None:
            if cmd_result.output:
                print_agent(cmd_result.output)
            if cmd_result.should_exit:
                return 0
            if cmd_result.agent_input is None:
                continue
            user_input = cmd_result.agent_input

        try:
            output = agent.run(user_input)
            print_agent(output)
        except (ModelError, RuntimeError) as exc:
            print_error(str(exc))


app = typer.Typer(name="mincode", help="MinCode - MiniMind terminal coding agent", add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Annotated[list[str] | None, typer.Argument(help="One-shot prompt (omit for REPL).")] = None,
    base_url: Annotated[str, typer.Option(envvar="MINIMIND_BASE_URL", help="MiniMind API URL.")] = "http://localhost:8998/v1",
    model: Annotated[str, typer.Option(help="Model name to send.")] = "minimind",
    max_steps: Annotated[int, typer.Option(help="Max agent loop steps.")] = 20,
    timeout: Annotated[int, typer.Option(help="Request timeout (seconds).")] = 120,
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="Max tokens (also controls prompt truncation in MiniMind API).")] = 4096,
    cwd: Annotated[str, typer.Option(help="Workspace directory.")] = "",
    yolo: Annotated[bool, typer.Option("--yolo", help="Auto-approve all mutating operations.")] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """MinCode - a terminal coding agent powered by MiniMind."""
    workspace = Path(cwd) if cwd else Path.cwd()

    try:
        agent, commands = _build_agent(
            base_url=base_url,
            model=model,
            max_steps=max_steps,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            cwd=workspace,
            yolo=yolo,
        )
    except RuntimeError as exc:
        print_error(f"Setup error: {exc}")
        raise typer.Exit(code=1) from None

    prompt_text = " ".join(prompt).strip() if prompt else ""
    if not prompt_text:
        code = _run_repl(agent, commands)
        raise typer.Exit(code=code)

    # One-shot mode
    cmd_result = commands.run(prompt_text)
    if cmd_result is not None:
        print_agent(cmd_result.output)
        if cmd_result.should_exit or cmd_result.agent_input is None:
            raise typer.Exit()
        prompt_text = cmd_result.agent_input

    try:
        output = agent.run(prompt_text)
        print_agent(output)
    except (ModelError, RuntimeError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=2) from None
