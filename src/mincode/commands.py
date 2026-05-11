"""Slash commands for interactive use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mincode.skills import SkillCatalog
from mincode.tools import ToolRegistry

CommandHandler = Callable[[str], "CommandResult"]


@dataclass
class CommandResult:
    output: str
    agent_input: str | None = None
    should_exit: bool = False


@dataclass
class SlashCommand:
    name: str
    description: str
    handler: CommandHandler


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd

    def list_commands(self) -> list[SlashCommand]:
        return [self._commands[n] for n in sorted(self._commands)]

    def run(self, raw_input: str) -> CommandResult | None:
        if not raw_input.startswith("/"):
            return None
        payload = raw_input[1:].strip()
        if not payload:
            return CommandResult(output="Empty command. Use /help.")
        name, _, args = payload.partition(" ")
        args = args.strip()
        cmd = self._commands.get(name)
        if cmd is None:
            return CommandResult(output=f"Unknown command: /{name}. Use /help.")
        return cmd.handler(args)


def create_default_commands(*, tools: ToolRegistry, skills: SkillCatalog) -> CommandRegistry:
    registry = CommandRegistry()

    def _help(_args: str) -> CommandResult:
        lines = ["Commands:"]
        for c in registry.list_commands():
            lines.append(f"  /{c.name} - {c.description}")
        return CommandResult(output="\n".join(lines))

    def _tools_cmd(_args: str) -> CommandResult:
        names = [t.name for t in tools.list_tools()]
        return CommandResult(output="Tools:\n" + "\n".join(f"  - {n}" for n in names))

    def _skills_cmd(_args: str) -> CommandResult:
        records = skills.list_skills()
        if not records:
            return CommandResult(output="No skills found.")
        lines = ["Skills:"]
        for s in records:
            lines.append(f"  - {s.name}: {s.description}")
        return CommandResult(output="\n".join(lines))

    def _skill(args: str) -> CommandResult:
        if not args:
            return CommandResult(output="Usage: /skill <name> [extra request]")
        parts = args.split(" ", 1)
        name = parts[0].strip().lower()
        extra = parts[1].strip() if len(parts) > 1 else ""
        skill = skills.get(name)
        if skill is None:
            return CommandResult(output=f"Skill not found: {name}")
        body = skill.read_text()
        if extra:
            body = f"{body}\n\nUser request:\n{extra}"
        return CommandResult(output=f"Running skill `{name}`...", agent_input=body)

    def _exit(_args: str) -> CommandResult:
        return CommandResult(output="Bye.", should_exit=True)

    registry.register(SlashCommand("help", "Show available commands", _help))
    registry.register(SlashCommand("tools", "List registered tools", _tools_cmd))
    registry.register(SlashCommand("skills", "List discovered skills", _skills_cmd))
    registry.register(SlashCommand("skill", "Run a skill: /skill <name> [request]", _skill))
    registry.register(SlashCommand("exit", "Exit session", _exit))

    return registry
