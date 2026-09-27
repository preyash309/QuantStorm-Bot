"""Research state representations.

AgentState contains only information a real bot can observe through Obs.
FullDealState is analysis-only and must never be passed to a submission bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AuctionEvent:
    round: int
    seat: int
    power: str
    cost: int


@dataclass(frozen=True)
class ContractState:
    round: int
    price: int
    long_seat: int
    forced: bool
    forcer: int
    shift: int
    maker_seat: int
    open_bid: int
    open_ask: int


@dataclass(frozen=True)
class AgentState:
    """Exact research copy of the information exposed by Obs."""

    seat: int
    round: int
    my_revealed: tuple[int, ...]
    te_mine: int
    te_theirs: int
    spread_cap: int
    final_cap: int
    is_maker: bool
    powers_mine: frozenset[str]
    powers_theirs: frozenset[str]
    auction_log: tuple[AuctionEvent, ...]
    contracts: tuple[ContractState, ...]
    foresight: tuple[int, ...]
    n_unknown_both: int
    n_turns: int

    @property
    def k_mine(self) -> int:
        return sum(self.my_revealed)


@dataclass(frozen=True)
class DecisionState:
    """AgentState plus the negotiation/auction context at a decision."""

    agent: AgentState
    turn: int | None = None
    offered: tuple[str, ...] = ()
    current_quote: tuple[int, int] | None = None


@dataclass
class FullDealState:
    """Analysis-only full game state for later simulator/DP work."""

    seed: int
    coins: tuple[int, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    score: int
    round: int = 0
    turn: int = 0
    revealed: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    te: tuple[int, int] = (24, 24)
    maker_seat: int = 0
    offered: tuple[str, ...] = ()
    powers: tuple[frozenset[str], frozenset[str]] = (
        frozenset(),
        frozenset(),
    )
    auction_log: tuple[AuctionEvent, ...] = ()
    contracts: tuple[ContractState, ...] = ()
    current_quote: tuple[int, int] | None = None
    foresight: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())


def obs_to_agent_state(obs) -> AgentState:
    """Convert an official immutable Obs into a research AgentState."""

    auctions = tuple(
        AuctionEvent(
            round=int(e["round"]),
            seat=int(e["seat"]),
            power=str(e["power"]),
            cost=int(e["cost"]),
        )
        for e in obs.auction_log
    )

    contracts = tuple(
        ContractState(
            round=int(c.round),
            price=int(c.price),
            long_seat=int(c.long_seat),
            forced=bool(c.forced),
            forcer=int(c.forcer),
            shift=int(c.shift),
            maker_seat=int(c.maker_seat),
            open_bid=int(c.open_bid),
            open_ask=int(c.open_ask),
        )
        for c in obs.contracts
    )

    return AgentState(
        seat=int(obs.seat),
        round=int(obs.round),
        my_revealed=tuple(int(x) for x in obs.my_revealed),
        te_mine=int(obs.te_mine),
        te_theirs=int(obs.te_theirs),
        spread_cap=int(obs.spread_cap),
        final_cap=int(obs.final_cap),
        is_maker=bool(obs.is_maker),
        powers_mine=frozenset(obs.powers_mine),
        powers_theirs=frozenset(obs.powers_theirs),
        auction_log=auctions,
        contracts=contracts,
        foresight=tuple(int(x) for x in obs.foresight),
        n_unknown_both=int(obs.n_unknown_both),
        n_turns=int(obs.n_turns),
    )

@dataclass
class SimState:
    """Mutable internal state used exclusively by ExactSimulator."""

    seed: int
    coins: tuple[int, ...]
    hands: list[list[int]]
    revealed: list[list[int]]
    score: int
    round: int = 0
    turn: int = 0
    te: list[int] = None
    maker_seat: int = 0
    offered: tuple[str, ...] = ()
    powers: object = None
    auction_log: list[dict] = None
    contracts: list = None
    current_quote: tuple[int, int] | None = None
    last_quoter: int | None = None
    foresight: list[tuple[int, ...]] = None
    slate: dict = None
    terminal: bool = False

    def __post_init__(self):
        if self.te is None:
            self.te = [24, 24]
        if self.auction_log is None:
            self.auction_log = []
        if self.contracts is None:
            self.contracts = []
        if self.foresight is None:
            self.foresight = [(), ()]
        if self.slate is None:
            self.slate = {}