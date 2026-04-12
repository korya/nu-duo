"""Built-in nu_tui components.

The first sub-slice of the nu_tui port ships only the simplest
components (:class:`Spacer`, :class:`Text`) so the Textual wrapper
can be exercised end-to-end with a minimal surface. The remaining
components (Editor, Input, TruncatedText, SelectList, SettingsList,
Markdown, Loader, Image, Box) land in follow-up sub-slices.
"""

from nu_tui.components.spacer import Spacer
from nu_tui.components.text import Text

__all__ = ["Spacer", "Text"]
