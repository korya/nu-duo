"""nu-mom — Slack bot delegating to pi agent.

Python port of ``@mariozechner/pi-mom``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "AgentRunner",
    "ChannelStore",
    "SlackBot",
]

if TYPE_CHECKING:
    from nu_mom.agent import AgentRunner
    from nu_mom.slack import SlackBot
    from nu_mom.store import ChannelStore


def __getattr__(name: str) -> object:
    if name == "AgentRunner":
        from nu_mom.agent import AgentRunner

        return AgentRunner
    if name == "SlackBot":
        from nu_mom.slack import SlackBot

        return SlackBot
    if name == "ChannelStore":
        from nu_mom.store import ChannelStore

        return ChannelStore
    raise AttributeError(f"module 'nu_mom' has no attribute {name!r}")
