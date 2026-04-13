"""Skill invocation message widget — port of ``skill-invocation-message.ts``.

Renders a skill block in collapsed or expanded form.  Clicking the widget
toggles between the two states (same as pressing the ``app.tools.expand``
keybinding in the upstream).
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText
from textual.widgets import Static

from nu_coding_agent.modes.interactive.components.keybinding_hints import key_hint

# Matches ``customMessageBg`` from the upstream dark theme
_BG = "on #2b2b3b"


@dataclass(slots=True)
class ParsedSkillBlock:
    """Parsed representation of a ``<skill …>`` block in a user message.

    Port of the ``ParsedSkillBlock`` interface from ``agent-session.ts``.
    """

    name: str
    location: str
    content: str
    user_message: str | None


class SkillInvocationWidget(Static):
    """Renders a skill invocation in collapsed/expanded form.

    Port of ``SkillInvocationMessageComponent`` (55 LoC). Background
    matches the upstream ``customMessageBg`` colour.
    """

    DEFAULT_CSS = """
    SkillInvocationWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, skill_block: ParsedSkillBlock) -> None:
        super().__init__("")
        self._skill_block = skill_block
        self._expanded = False
        self._update_display()

    # ------------------------------------------------------------------

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self._update_display()

    def on_click(self) -> None:
        self._expanded = not self._expanded
        self._update_display()

    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        if self._expanded:
            t = RichText()
            t.append("[skill] ", style=f"bold {_BG}")
            t.append(self._skill_block.name, style=f"bold {_BG}")
            t.append("\n")
            md_text = f"**{self._skill_block.name}**\n\n{self._skill_block.content}"
            self.update(RichMarkdown(md_text, code_theme="monokai"))
        else:
            hint = key_hint("app.tools.expand", "to expand")
            t = RichText()
            t.append("[skill] ", style=f"bold {_BG}")
            t.append(self._skill_block.name, style=_BG)
            t.append(f"  ({hint})", style=f"dim {_BG}")
            self.update(t)


__all__ = ["ParsedSkillBlock", "SkillInvocationWidget"]
