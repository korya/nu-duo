"""Interactive-mode components — message widgets, tool renderers, utility helpers.

These widgets render conversation entries, tool executions, bash output,
skill invocations, and extension messages in the interactive REPL.
"""

from nu_coding_agent.modes.interactive.components.bordered_loader import (
    BorderedLoader,
    BorderedLoaderWithSpinner,
)
from nu_coding_agent.modes.interactive.components.custom_message import (
    CustomMessageWidget,
    MessageRenderer,
)
from nu_coding_agent.modes.interactive.components.diff import render_diff
from nu_coding_agent.modes.interactive.components.keybinding_hints import (
    key_hint,
    key_text,
    raw_key_hint,
)
from nu_coding_agent.modes.interactive.components.login_dialog import LoginDialog
from nu_coding_agent.modes.interactive.components.message_renderers import (
    AssistantMessageWidget,
    BashExecutionWidget,
    BranchSummaryWidget,
    CompactionSummaryWidget,
    ErrorWidget,
    InfoWidget,
    ToolExecutionWidget,
    UserMessageWidget,
)
from nu_coding_agent.modes.interactive.components.skill_invocation_message import (
    ParsedSkillBlock,
    SkillInvocationWidget,
)
from nu_coding_agent.modes.interactive.components.tool_renderers import (
    render_tool_call,
    render_tool_result,
)

__all__ = [
    "AssistantMessageWidget",
    "BashExecutionWidget",
    "BorderedLoader",
    "BorderedLoaderWithSpinner",
    "BranchSummaryWidget",
    "CompactionSummaryWidget",
    "CustomMessageWidget",
    "ErrorWidget",
    "InfoWidget",
    "LoginDialog",
    "MessageRenderer",
    "ParsedSkillBlock",
    "SkillInvocationWidget",
    "ToolExecutionWidget",
    "UserMessageWidget",
    "key_hint",
    "key_text",
    "raw_key_hint",
    "render_diff",
    "render_tool_call",
    "render_tool_result",
]
