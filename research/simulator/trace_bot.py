"""Exact observation/action recorder for the official engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import AgentState, obs_to_agent_state


@dataclass
class TraceEvent:
    method: str
    round: int
    turn: int | None
    state: AgentState
    offered: tuple[str, ...]
    quote_before: tuple[int, int] | None
    action: Any


class TraceBot:
    """Wrap one existing bot and record exactly what it sees/returns."""

    def __init__(self, bot_factory, label: str = "TraceBot"):
        self._inner = bot_factory()
        self._label = label
        self.events: list[TraceEvent] = []
        self.last_call_ms = 0.0

    @property
    def name(self):
        return getattr(self._inner, "name", self._label)

    def reset(self, seat, config, seed):
        return self._inner.reset(seat, config, seed)

    def bid(self, obs, offered):
        result = self._inner.bid(obs, offered)
        self.events.append(
            TraceEvent(
                method="bid",
                round=int(obs.round),
                turn=None,
                state=obs_to_agent_state(obs),
                offered=tuple(offered),
                quote_before=None,
                action=result,
            )
        )
        return result

    def quote(self, obs):
        result = self._inner.quote(obs)
        self.events.append(
            TraceEvent(
                method="quote",
                round=int(obs.round),
                turn=1,
                state=obs_to_agent_state(obs),
                offered=(),
                quote_before=None,
                action=result,
            )
        )
        return result

    def respond(self, obs, quote, turn):
        result = self._inner.respond(obs, quote, turn)
        self.events.append(
            TraceEvent(
                method="respond",
                round=int(obs.round),
                turn=int(turn),
                state=obs_to_agent_state(obs),
                offered=(),
                quote_before=tuple(quote),
                action=result,
            )
        )
        return result

    def use_transform(self, obs):
        result = self._inner.use_transform(obs)
        self.events.append(
            TraceEvent(
                method="use_transform",
                round=int(obs.round),
                turn=None,
                state=obs_to_agent_state(obs),
                offered=(),
                quote_before=None,
                action=result,
            )
        )
        return result
