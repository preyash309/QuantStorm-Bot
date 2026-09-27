"""
PHASE 4A — Exact negotiation EV evaluator.

Research only.

Purpose:
    Given an observed quote and our current information state,
    enumerate every legal immediate negotiation action and calculate
    its mathematical contract EV.

This does NOT:
    - modify AdaptiveFinal
    - simulate the entire deal
    - model future rounds
    - perform Monte Carlo
    - assume opponent probabilities

Those come later.

The official engine remains the execution oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ============================================================
# ACTION REPRESENTATION
# ============================================================

@dataclass(frozen=True)
class NegotiationAction:
    kind: str
    bid: int | None = None
    ask: int | None = None

    def as_engine_action(self):
        if self.kind == "ACCEPT_BUY":
            return "ACCEPT_BUY"

        if self.kind == "ACCEPT_SELL":
            return "ACCEPT_SELL"

        if self.kind == "COUNTER":
            return (
                "COUNTER",
                int(self.bid),
                int(self.ask),
            )

        raise ValueError(
            f"Unknown action: {self.kind}"
        )


@dataclass(frozen=True)
class ActionEV:
    action: NegotiationAction
    ev: float


# ============================================================
# EXACT POSTERIOR MEAN
# ============================================================

def posterior_mean(obs) -> float:
    """
    Exact E[S | information currently visible to us].

    Supports both:

    1. Live Obs / older research representation:
           obs.foresight = tuple/list of revealed coins

    2. Phase 3 serialized dataset representation:
           obs.foresight_sum = aggregate revealed value

    Unknown fair coins have expectation zero.
    """

    if hasattr(obs, "foresight_sum"):
        foresight_value = float(
            obs.foresight_sum
        )

    elif hasattr(obs, "foresight"):
        foresight_value = float(
            sum(obs.foresight)
        )

    else:
        foresight_value = 0.0

    return (
        float(obs.k_mine)
        + foresight_value
    )


# ============================================================
# IMMEDIATE CONTRACT EV
# ============================================================

def accept_buy_ev(
    obs,
    price: int,
) -> float:
    """
    We buy at price.

        PnL = S - price
        EV  = E[S] - price
    """

    return (
        posterior_mean(obs)
        - price
    )


def accept_sell_ev(
    obs,
    price: int,
) -> float:
    """
    We sell at price.

        PnL = price - S
        EV  = price - E[S]
    """

    return (
        price
        - posterior_mean(obs)
    )


# ============================================================
# FORCED MIDPOINT EV
# ============================================================

def forced_fill_price(
    bid: int,
    ask: int,
    shift: int = 0,
) -> int:
    """
    Official forced midpoint rule:

        (bid + ask) // 2 + shift
    """

    return (
        (bid + ask) // 2
        + shift
    )


def forced_ev(
    obs,
    bid: int,
    ask: int,
    shift: int = 0,
    forcing_fee: float = 2.0,
) -> float:
    """
    EV of becoming the short side of a forced midpoint fill.

    Short at forced price p:

        PnL = p - S

    Then pay the forcing fee.
    """

    price = forced_fill_price(
        bid,
        ask,
        shift,
    )

    return (
        accept_sell_ev(
            obs,
            price,
        )
        - forcing_fee
    )


# ============================================================
# LEGAL COUNTER WIDTH
# ============================================================

def maximum_counter_width(
    bid: int,
    ask: int,
    final_cap: int,
    min_reduction: int,
) -> int:
    """
    Official rule:

        max_width =
            min(
                current_width,
                max(
                    final_cap,
                    current_width - MIN_REDUCTION
                )
            )
    """

    width = ask - bid

    return min(
        width,
        max(
            final_cap,
            width - min_reduction,
        ),
    )


# ============================================================
# LEGAL COUNTER ENUMERATION
# ============================================================

def enumerate_counters(
    bid: int,
    ask: int,
    final_cap: int,
    min_reduction: int,
) -> tuple[NegotiationAction, ...]:
    """
    Enumerate the counter actions that the official engine can
    actually accept after its response sanitisation.

    IMPORTANT:
        This intentionally mirrors the ENGINE'S ACTUAL behavior,
        rather than enforcing the intended final_cap floor on
        response widths.

    Official response handling:

        max_width =
            min(
                current_width,
                max(
                    final_cap,
                    current_width - MIN_REDUCTION,
                ),
            )

    A returned counter is:
        - clamped inside the current quote
        - allowed to have any width <= max_width
        - NOT rejected for being narrower than final_cap

    Therefore we enumerate every integer sub-range whose width
    is <= max_width.

    This is required for exact research/engine parity.
    """

    if bid > ask:
        bid, ask = ask, bid

    current_width = ask - bid

    max_width = min(
        current_width,
        max(
            final_cap,
            current_width - min_reduction,
        ),
    )

    counters: list[NegotiationAction] = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT require width >= final_cap.
    #
    # The official engine permits narrower counters because its
    # response sanitiser only clamps widths ABOVE max_width.
    #
    # Width 0 is therefore also a possible engine-level result.
    # --------------------------------------------------------

    for nb in range(
        bid,
        ask + 1,
    ):

        for na in range(
            nb,
            ask + 1,
        ):

            width = na - nb

            if width > max_width:
                continue

            counters.append(
                NegotiationAction(
                    kind="COUNTER",
                    bid=nb,
                    ask=na,
                )
            )

    # Deterministic ordering:
    # narrower first, then midpoint, then endpoints.
    counters.sort(
        key=lambda action: (
            action.ask - action.bid,
            (action.bid + action.ask) / 2,
            action.bid,
            action.ask,
        )
    )

    return tuple(
        counters
    )


# ============================================================
# IMMEDIATE ACTION ENUMERATION
# ============================================================

def enumerate_actions(
    bid: int,
    ask: int,
    final_cap: int,
    min_reduction: int,
) -> tuple[NegotiationAction, ...]:

    actions = [
        NegotiationAction(
            kind="ACCEPT_BUY"
        ),

        NegotiationAction(
            kind="ACCEPT_SELL"
        ),
    ]

    actions.extend(
        enumerate_counters(
            bid,
            ask,
            final_cap,
            min_reduction,
        )
    )

    return tuple(actions)


# ============================================================
# ACTION EV
# ============================================================

def action_ev(
    obs,
    action: NegotiationAction,
    current_quote: tuple[int, int],
    shift: int = 0,
    forcing_fee: float = 2.0,
) -> float:

    bid, ask = current_quote

    if action.kind == "ACCEPT_BUY":

        return accept_buy_ev(
            obs,
            ask,
        )

    if action.kind == "ACCEPT_SELL":

        return accept_sell_ev(
            obs,
            bid,
        )

    if action.kind == "COUNTER":

        assert action.bid is not None
        assert action.ask is not None

        return forced_ev(
            obs,
            action.bid,
            action.ask,
            shift=shift,
            forcing_fee=forcing_fee,
        )

    raise ValueError(
        f"Unknown action: {action}"
    )


# ============================================================
# RANK ACTIONS
# ============================================================

def rank_actions(
    obs,
    current_quote: tuple[int, int],
    final_cap: int,
    min_reduction: int,
    shift: int = 0,
    forcing_fee: float = 2.0,
) -> tuple[ActionEV, ...]:
    """
    Phase 4A ranking.

    Counter EV is only the forced-fill terminal baseline.

    Phase 4B's continuation evaluator replaces this when
    opponent-response probabilities are available.
    """

    bid, ask = current_quote

    actions = enumerate_actions(
        bid,
        ask,
        final_cap,
        min_reduction,
    )

    result = [
        ActionEV(
            action=action,
            ev=action_ev(
                obs,
                action,
                current_quote,
                shift=shift,
                forcing_fee=forcing_fee,
            ),
        )
        for action in actions
    ]

    result.sort(
        key=lambda x: x.ev,
        reverse=True,
    )

    return tuple(result)


# ============================================================
# PRETTY PRINT
# ============================================================

def format_action(
    action: NegotiationAction,
) -> str:

    if action.kind in (
        "ACCEPT_BUY",
        "ACCEPT_SELL",
    ):
        return action.kind

    return (
        f"COUNTER "
        f"[{action.bid},{action.ask}]"
    )


def print_ranking(
    ranked: Iterable[ActionEV],
):

    print()
    print("=" * 72)
    print(
        "PHASE 4A — NEGOTIATION EV"
    )
    print("=" * 72)

    for i, item in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{i:3d}. "
            f"{format_action(item.action):<24}"
            f"EV = {item.ev:+.6f}"
        )