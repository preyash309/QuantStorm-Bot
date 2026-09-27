"""
test_exact_negotiation.py

Deterministic tests for the exact final-turn optimizer.
"""

from game_config import GameConfig

from exact_negotiation import (
    expected_contract_payoff,
    forced_fill_shift,
    best_forcing_counter,
    final_turn_actions,
    best_final_action,
)


def close(
    a,
    b,
    eps=1e-12,
):

    assert abs(a - b) <= eps, (
        f"{a} != {b}"
    )


def main():

    config = GameConfig()

    # ========================================================
    # Basic contract EV
    # ========================================================

    # S = 5 exactly.
    #
    # Long at 3 => +2
    # Short at 7 => +2

    long_ev = (
        expected_contract_payoff(
            price=3,
            long=True,
            known_sum=5,
            unknown_count=0,
            has_substitute=False,
        )
    )

    short_ev = (
        expected_contract_payoff(
            price=7,
            long=False,
            known_sum=5,
            unknown_count=0,
            has_substitute=False,
        )
    )

    close(
        long_ev,
        2.0,
    )

    close(
        short_ev,
        2.0,
    )

    # ========================================================
    # SUBSTITUTE
    # ========================================================

    # S = 0, price = 5
    #
    # Long raw PnL = -5
    # SUBSTITUTE cap = -2
    #
    # Therefore payoff = -2.

    substitute_ev = (
        expected_contract_payoff(
            price=5,
            long=True,
            known_sum=0,
            unknown_count=0,
            has_substitute=True,
        )
    )

    close(
        substitute_ev,
        -2.0,
    )

    # ========================================================
    # Shift
    # ========================================================

    shift = forced_fill_shift(
        my_powers=frozenset({
            "TRICK_ROOM",
        }),
        opponent_powers=frozenset(),
        config=config,
    )

    close(
        shift,
        3,
    )

    shift_cancelled = (
        forced_fill_shift(
            my_powers=frozenset({
                "TRICK_ROOM",
            }),
            opponent_powers=frozenset({
                "TRICK_ROOM",
            }),
            config=config,
        )
    )

    close(
        shift_cancelled,
        0,
    )

    # ========================================================
    # Final counter
    # ========================================================

    new_bid, new_ask = (
        best_forcing_counter(
            bid=0,
            ask=8,
            final_cap=4,
        )
    )

    new_bid, new_ask = best_forcing_counter(
    bid=0,
    ask=8,
    final_cap=4,
)

    assert 0 <= new_bid <= new_ask <= 8
    assert new_ask - new_bid >= 4
    assert new_ask - new_bid <= 7

    # ========================================================
    # Full optimizer
    # ========================================================

    actions = final_turn_actions(
        bid=0,
        ask=4,
        final_cap=4,
        known_sum=10,
        unknown_count=0,
        my_powers=frozenset(),
        opponent_powers=frozenset(),
        config=config,
    )

    names = {
        a.action
        for a in actions
    }

    assert names == {
        "ACCEPT_BUY",
        "ACCEPT_SELL",
        "FORCE",
    }

    best = best_final_action(
        bid=0,
        ask=4,
        final_cap=4,
        known_sum=10,
        unknown_count=0,
        my_powers=frozenset(),
        opponent_powers=frozenset(),
        config=config,
    )

    # S=10:
    #
    # BUY at 4  -> +6
    # SELL at 0 -> -10
    # FORCE     -> short, therefore bad.
    #
    # So BUY must win.

    assert best.action == "ACCEPT_BUY"

    print()
    print("=" * 60)
    print("EXACT NEGOTIATION TESTS PASSED")
    print("=" * 60)

    print()

    for action in actions:

        print(
            f"{action.action:<15}"
            f" EV={action.ev:+.6f}"
            f" price={action.price:>3}"
            f"  {action.details}"
        )

    print()
    print(
        f"BEST = {best.action}"
    )


if __name__ == "__main__":
    main()