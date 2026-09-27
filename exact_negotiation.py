"""
exact_negotiation.py

Exact EV calculations for the Divided Oracle negotiation.

This module is used ONLY for the final negotiation turn.

On the final turn, the three meaningful actions are:

    ACCEPT_BUY
        Long at ask.

    ACCEPT_SELL
        Short at bid.

    COUNTER
        The negotiation ends in a forced fill.
        The final quoter becomes short.
        The final quoter pays the forcing fee.

Because the score distribution is a finite +/-1 lattice,
every EV below is calculated exactly.

No Monte Carlo.
No approximation.
No learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from exact_value import (
    score_pmf,
)


# ============================================================
# Result
# ============================================================

@dataclass(frozen=True)
class ActionEV:
    """
    Exact expected value of one final-turn action.
    """

    action: str

    ev: float

    price: int

    details: str = ""


# ============================================================
# Settlement payoff
# ============================================================

def protected_payoff(
    raw_pnl: float,
    has_substitute: bool,
    cap: float = 2.0,
) -> float:
    """
    Apply SUBSTITUTE to one contract payoff.

    Without SUBSTITUTE:

        payoff = raw_pnl

    With SUBSTITUTE:

        payoff = max(raw_pnl, -cap)

    Profit is uncapped.
    """

    if not has_substitute:
        return raw_pnl

    return max(
        raw_pnl,
        -cap,
    )


def expected_contract_payoff(
    *,
    price: int,
    long: bool,
    known_sum: int,
    unknown_count: int,
    has_substitute: bool,
    substitute_cap: float = 2.0,
) -> float:
    """
    Exact expected settlement PnL of a contract.

    Long:

        raw = S - price

    Short:

        raw = price - S

    SUBSTITUTE is applied after S is known.
    """

    total = 0.0

    for score, probability in score_pmf(
        known_sum,
        unknown_count,
    ):

        if long:
            raw = score - price
        else:
            raw = price - score

        payoff = protected_payoff(
            raw,
            has_substitute,
            substitute_cap,
        )

        total += (
            probability
            * payoff
        )

    return total


# ============================================================
# Fill-shift calculation
# ============================================================

def shift_magnitude(
    active_powers,
    config,
) -> int:
    """
    Total fill-shift magnitude controlled by a seat.

    TRICK_ROOM:
        current-round forced fill shift.

    STEALTH_ROCK:
        persistent forced-fill shift.
    """

    total = 0

    for name in (
        "TRICK_ROOM",
        "STEALTH_ROCK",
    ):

        if name in active_powers:

            if name in config.POWERS:

                total += int(
                    config.POWERS[name][
                        "magnitude"
                    ]
                )

    return total


def forced_fill_shift(
    *,
    my_powers,
    opponent_powers,
    config,
) -> int:
    """
    The bot using this function becomes SHORT on a forced fill.

    Therefore:

        own shift       -> positive
        opponent shift  -> negative
    """

    mine = shift_magnitude(
        my_powers,
        config,
    )

    theirs = shift_magnitude(
        opponent_powers,
        config,
    )

    return mine - theirs


# ============================================================
# Legal final counter
# ============================================================

def best_forcing_counter(
    *,
    bid: int,
    ask: int,
    final_cap: int,
) -> tuple[int, int]:
    """
    Construct the final-turn counter that maximizes the forced
    midpoint price for the countering player.

    The final quoter becomes SHORT.

    Therefore, all else equal, we want the highest possible
    midpoint.

    The new range must:

        1. remain inside [bid, ask]
        2. have legal width
        3. never be narrower than final_cap
        4. shrink by at least one tick unless already at floor

    The highest possible midpoint is obtained by placing the
    legal range against the old ASK.
    """

    old_width = ask - bid

    max_width = max(
        final_cap,
        old_width - 1,
    )

    max_width = min(
        old_width,
        max_width,
    )

    new_ask = ask

    new_bid = (
        new_ask
        - max_width
    )

    return (
        new_bid,
        new_ask,
    )


# ============================================================
# Exact final-turn optimizer
# ============================================================

def final_turn_actions(
    *,
    bid: int,
    ask: int,
    final_cap: int,
    known_sum: int,
    unknown_count: int,
    my_powers,
    opponent_powers,
    config,
) -> tuple[ActionEV, ActionEV, ActionEV]:
    """
    Calculate exact EV for:

        ACCEPT_BUY
        ACCEPT_SELL
        FORCE

    FORCE means:

        COUNTER on the final turn
        -> forced midpoint
        -> this bot becomes short
        -> this bot pays forcing fee
    """

    has_substitute = (
        "SUBSTITUTE"
        in my_powers
    )

    # --------------------------------------------------------
    # ACCEPT BUY
    # --------------------------------------------------------

    buy_ev = expected_contract_payoff(
        price=ask,
        long=True,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    buy_action = ActionEV(
        action="ACCEPT_BUY",
        ev=buy_ev,
        price=ask,
        details=(
            f"long at ask={ask}"
        ),
    )

    # --------------------------------------------------------
    # ACCEPT SELL
    # --------------------------------------------------------

    sell_ev = expected_contract_payoff(
        price=bid,
        long=False,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    sell_action = ActionEV(
        action="ACCEPT_SELL",
        ev=sell_ev,
        price=bid,
        details=(
            f"short at bid={bid}"
        ),
    )

    # --------------------------------------------------------
    # FORCE
    # --------------------------------------------------------

    new_bid, new_ask = (
        best_forcing_counter(
            bid=bid,
            ask=ask,
            final_cap=final_cap,
        )
    )

    midpoint = (
        new_bid
        + new_ask
    ) // 2

    shift = forced_fill_shift(
        my_powers=my_powers,
        opponent_powers=opponent_powers,
        config=config,
    )

    forced_price = (
        midpoint
        + shift
    )

    force_ev = expected_contract_payoff(
        price=forced_price,
        long=False,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    # Final-turn counter incurs the forcing fee.
    force_ev -= (
        config.FORCED_FILL_FEE
    )

    force_action = ActionEV(
        action="FORCE",
        ev=force_ev,
        price=forced_price,
        details=(
            f"counter=[{new_bid},{new_ask}], "
            f"midpoint={midpoint}, "
            f"shift={shift}, "
            f"forced_price={forced_price}, "
            f"fee={config.FORCED_FILL_FEE}"
        ),
    )

    return (
        buy_action,
        sell_action,
        force_action,
    )


def best_final_action(
    **kwargs,
) -> ActionEV:
    """
    Return the exact EV-maximizing final-turn action.
    """

    actions = final_turn_actions(
        **kwargs
    )

    return max(
        actions,
        key=lambda action: action.ev,
    )


# ============================================================
# Debugging / standalone tests
# ============================================================

if __name__ == "__main__":

    from game_config import GameConfig

    config = GameConfig()

    actions = final_turn_actions(
        bid=0,
        ask=4,
        final_cap=4,
        known_sum=2,
        unknown_count=20,
        my_powers=frozenset(),
        opponent_powers=frozenset(),
        config=config,
    )

    print()
    print("=" * 60)
    print("EXACT FINAL-TURN EV")
    print("=" * 60)

    for action in actions:

        print(
            f"{action.action:<15} "
            f"EV={action.ev:+.6f} "
            f"price={action.price:>3} "
            f"{action.details}"
        )

    best = max(
        actions,
        key=lambda x: x.ev,
    )

    print()
    print(
        f"BEST ACTION: "
        f"{best.action} "
        f"(EV={best.ev:+.6f})"
    )