"""Built-in nu_tui components.

The first sub-slice of the nu_tui port ships only the simplest
components (:class:`Spacer`, :class:`Text`) so the Textual wrapper
can be exercised end-to-end with a minimal surface. The remaining
components (Editor, Input, TruncatedText, SelectList, SettingsList,
Markdown, Loader, Image, Box) land in follow-up sub-slices.
"""

from nu_tui.components.box import Box
from nu_tui.components.select_list import SelectItem, SelectList, SelectListTheme, default_select_list_theme
from nu_tui.components.spacer import Spacer
from nu_tui.components.text import Text
from nu_tui.components.truncated_text import TruncatedText

__all__ = [
    "Box",
    "SelectItem",
    "SelectList",
    "SelectListTheme",
    "Spacer",
    "Text",
    "TruncatedText",
    "default_select_list_theme",
]
