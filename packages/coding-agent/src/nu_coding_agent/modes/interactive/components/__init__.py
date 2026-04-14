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
from nu_coding_agent.modes.interactive.components.extension_ui import (
    ExtensionEditorScreen,
    ExtensionInputScreen,
    ExtensionSelectorScreen,
)
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
from nu_coding_agent.modes.interactive.components.model_selector import ModelSelectorScreen
from nu_coding_agent.modes.interactive.components.session_selector import (
    SessionSelectorScreen,
    filter_and_sort_sessions,
    has_session_name,
)
from nu_coding_agent.modes.interactive.components.settings_selector import (
    ConfigSelectorScreen,
    ResourceGroup,
    ResourceItem,
    ResourceSubgroup,
    ScopedModelsSelectorScreen,
    SettingItem,
    SettingsSelectorScreen,
    UserMessageSelectorScreen,
)
from nu_coding_agent.modes.interactive.components.skill_invocation_message import (
    ParsedSkillBlock,
    SkillInvocationWidget,
)
from nu_coding_agent.modes.interactive.components.tool_renderers import (
    render_tool_call,
    render_tool_result,
)
from nu_coding_agent.modes.interactive.components.tree_selector import (
    FilterMode,
    TreeSelectorScreen,
)
from nu_coding_agent.modes.interactive.components.visual_truncate import (
    VisualTruncateResult,
    count_visual_lines,
    truncate_to_visual_lines,
)

__all__ = [
    "AssistantMessageWidget",
    "BashExecutionWidget",
    "BorderedLoader",
    "BorderedLoaderWithSpinner",
    "BranchSummaryWidget",
    "CompactionSummaryWidget",
    "ConfigSelectorScreen",
    "CustomMessageWidget",
    "ErrorWidget",
    "ExtensionEditorScreen",
    "ExtensionInputScreen",
    "ExtensionSelectorScreen",
    "FilterMode",
    "InfoWidget",
    "LoginDialog",
    "MessageRenderer",
    "ModelSelectorScreen",
    "ParsedSkillBlock",
    "ResourceGroup",
    "ResourceItem",
    "ResourceSubgroup",
    "ScopedModelsSelectorScreen",
    "SessionSelectorScreen",
    "SettingItem",
    "SettingsSelectorScreen",
    "SkillInvocationWidget",
    "ToolExecutionWidget",
    "TreeSelectorScreen",
    "UserMessageSelectorScreen",
    "UserMessageWidget",
    "VisualTruncateResult",
    "count_visual_lines",
    "filter_and_sort_sessions",
    "has_session_name",
    "key_hint",
    "key_text",
    "raw_key_hint",
    "render_diff",
    "render_tool_call",
    "render_tool_result",
    "truncate_to_visual_lines",
]
