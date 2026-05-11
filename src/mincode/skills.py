"""Skill discovery: find SKILL.md files and make them available."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_description(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line[:160]
    return "No description."


@dataclass
class SkillCatalog:
    _skills: dict[str, Skill]

    @classmethod
    def from_roots(cls, roots: list[Path]) -> "SkillCatalog":
        indexed: dict[str, Skill] = {}
        for root in roots:
            root = root.expanduser().resolve()
            if not root.exists() or not root.is_dir():
                continue
            # Check root/SKILL.md
            direct = root / SKILL_FILE_NAME
            if direct.is_file():
                name = root.name.lower().replace(" ", "-")
                if name not in indexed:
                    content = direct.read_text(encoding="utf-8", errors="replace")
                    indexed[name] = Skill(name=name, description=_extract_description(content), path=direct)
            # Check root/*/SKILL.md (one level deep)
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                nested = child / SKILL_FILE_NAME
                if nested.is_file():
                    name = child.name.lower().replace(" ", "-")
                    if name not in indexed:
                        content = nested.read_text(encoding="utf-8", errors="replace")
                        indexed[name] = Skill(name=name, description=_extract_description(content), path=nested)
        return cls(_skills=indexed)

    def list_skills(self) -> list[Skill]:
        return [self._skills[n] for n in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.strip().lower())

    def format_for_system_prompt(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "(none)"
        lines = ["Available skills (invoke with /skill <name>):"]
        for s in skills:
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)
