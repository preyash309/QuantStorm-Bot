from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
    ),
)

from game_config import DEFAULT_CONFIG
from research.ev.negotiation_ev import (
    enumerate_counters,
    enumerate_actions,
    posterior_mean,
    rank_actions,
)


class FakeObs:
    k_mine = 4
    foresight = (1, -1)
    final_cap = 2
    spread_cap = 8


def main():

    obs = FakeObs()

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    mean = posterior_mean(obs)

    assert mean == 4

    # --------------------------------------------------------
    # Counter legality
    # --------------------------------------------------------

    counters = enumerate_counters(
        bid=0,
        ask=8,
        final_cap=2,
        min_reduction=1,
    )

    assert len(counters) > 0

    for action in counters:

        assert action.kind == "COUNTER"

        assert action.bid is not None
        assert action.ask is not None

        assert 0 <= action.bid
        assert action.bid <= action.ask <= 8

        width = (
            action.ask
            - action.bid
        )

        # The engine accepts widths below final_cap, including zero.
        # final_cap bounds mandatory reduction; it is not a lower bound.
        assert width >= 0
        assert width <= 7

    assert any(action.bid == action.ask for action in counters)

    # --------------------------------------------------------
    # Action enumeration
    # --------------------------------------------------------

    actions = enumerate_actions(
        bid=0,
        ask=8,
        final_cap=2,
        min_reduction=1,
    )

    assert actions[0].kind == "ACCEPT_BUY"
    assert actions[1].kind == "ACCEPT_SELL"

    assert len(actions) == (
        2 + len(counters)
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked = rank_actions(
        obs=obs,
        current_quote=(0, 8),
        final_cap=2,
        min_reduction=1,
    )

    assert len(ranked) == len(actions)

    # At value 4:
    #
    # BUY at 8 -> -4
    # SELL at 0 -> -4
    #
    # They must therefore be equal.

    buy_ev = next(
        x.ev
        for x in ranked
        if x.action.kind
        == "ACCEPT_BUY"
    )

    sell_ev = next(
        x.ev
        for x in ranked
        if x.action.kind
        == "ACCEPT_SELL"
    )

    assert abs(
        buy_ev - sell_ev
    ) < 1e-12

    print("=" * 60)
    print("PHASE 4A TEST PASSED")
    print("=" * 60)

    print(
        f"Posterior mean: {mean}"
    )

    print(
        f"Legal counters: "
        f"{len(counters)}"
    )

    print(
        f"Total actions: "
        f"{len(actions)}"
    )

    print(
        f"BUY EV:  {buy_ev:+.6f}"
    )

    print(
        f"SELL EV: {sell_ev:+.6f}"
    )


if __name__ == "__main__":
    main()
