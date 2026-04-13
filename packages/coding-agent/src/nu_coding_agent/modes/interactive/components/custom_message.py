"""Custom message widget — port of ``custom-message.ts``.

Renders extension-injected :class:`~nu_coding_agent.core.messages.CustomMessage`
entries. When the extension registered a ``MessageRenderer`` for the message's
``custom_type``, that renderer is called first; on failure (or if absent) the
default purple-box rendering kicks in.

The ``MessageRenderer`` protocol maps to:

    (message, options, theme) -> renderable | None

where ``renderable`` is anything Textual's ``Static.update()`` accepts
(Rich renderables or plain strings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText
from textual.widgets import Static

if TYPE_CHECKING:
    from collections.abc import Callable

    from nu_coding_agent.core.messages import CustomMessage

# Matches upstream ``customMessageBg`` and ``customMessageLabel`` colours.
_BG = "on #2b2b3b"
_LABEL_STYLE = f"bold {_BG}"
_TEXT_STYLE = _BG

# The Python analogue of the upstream ``MessageRenderer`` type alias.
# (message, options_dict, theme_object) -> renderable | None
type MessageRenderer = Callable[[Any, dict[str, Any], Any], Any]


class CustomMessageWidget(Static):
    """Renders a ``CustomMessage`` from an extension.

    Port of ``CustomMessageComponent`` (99 LoC).  The widget supports
    click-to-toggle expand, and delegates to a caller-supplied
    ``renderer`` callable when one is registered for the message's
    ``custom_type``.
    """

    DEFAULT_CSS = """
    CustomMessageWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        message: CustomMessage,
        *,
        renderer: MessageRenderer | None = None,
    ) -> None:
        super().__init__("")
        self._message = message
        self._renderer = renderer
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
        options = {"expanded": self._expanded}

        # Try extension-provided renderer first
        if self._renderer is not None:
            try:
                result = self._renderer(self._message, options, None)
                if result is not None:
                    self.update(result)
                    return
            except Exception:
                pass  # fall through to default rendering

        # Default rendering
        label = self._message.custom_type
        content = self._message.content
        if isinstance(content, str):
            text = content
        else:
            parts = []
            for block in content:
                if hasattr(block, "text") and isinstance(getattr(block, "text", None), str):
                    parts.append(block.text)  # type: ignore[union-attr]
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "\n".join(parts)

        header = RichText()
        header.append(f"[{label}]", style=_LABEL_STYLE)
        header.append("\n")

        if text.strip():
            self.update(RichMarkdown(f"**[{label}]**\n\n{text}", code_theme="monokai"))
        else:
            self.update(header)


__all__ = ["CustomMessageWidget", "MessageRenderer"]
