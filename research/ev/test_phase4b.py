from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
    ),
)


from research.ev.public_opponent import (
    PublicOpponentModel,
)

from research.ev.continuation import (
    evaluate_all_actions,
)


class FakeObs:

    round = 2

    is_maker = False

    k_mine = 4

    # Keep the old representation for any tests/code
    # that still expect it.
    foresight = ()

    # New Phase 3 dataset-compatible representation.
    foresight_sum = 0

    te_mine = 12

    te_theirs = 12

    powers_mine = frozenset()

    powers_theirs = frozenset()

    final_cap = 2

    spread_cap = 8

def main():

    obs = FakeObs()

    # --------------------------------------------------------
    # Construct deterministic public model.
    # --------------------------------------------------------

    model = PublicOpponentModel()

    state = model.public_state(
        {
            "method": "respond",
            "obs": {
                "round": 2,
                "is_maker": True,
                "te_mine": 12,
                "te_theirs": 12,
                "powers_mine": [],
                "powers_theirs": [],
            },
            "quote": [0, 6],
            "turn": 2,
        }
    )

    model.tables[state][
        "ACCEPT_BUY"
    ] = 8

    model.tables[state][
        "ACCEPT_SELL"
    ] = 1

    model.tables[state][
        "COUNTER"
    ] = 1

    model.global_counts.update({
        "ACCEPT_BUY": 8,
        "ACCEPT_SELL": 1,
        "COUNTER": 1,
    })

    # --------------------------------------------------------
    # Predict.
    # --------------------------------------------------------

    probabilities = model.predict(
        round_no=2,
        is_maker=True,
        te_mine=12,
        te_theirs=12,
        powers_mine=[],
        powers_theirs=[],
        quote=(0, 6),
        turn=2,
    )

    assert abs(
        sum(
            probabilities.values()
        ) - 1.0
    ) < 1e-12

    assert (
        probabilities[
            "ACCEPT_BUY"
        ]
        >
        probabilities[
            "ACCEPT_SELL"
        ]
    )

    # --------------------------------------------------------
    # EV.
    # --------------------------------------------------------

    ranked = evaluate_all_actions(
        obs=obs,
        current_quote=(0, 6),
        opponent_model=model,
        turn=2,
        final_cap=2,
        min_reduction=1,
        forcing_fee=2.0,
    )

    assert len(ranked) > 2

    assert (
        ranked[0].ev
        >= ranked[-1].ev
    )

    # Every result must have a proper probability triple.

    for result in ranked:

        total = (
            result.p_accept_buy
            + result.p_accept_sell
            + result.p_counter
        )

        assert (
            abs(total - 1.0)
            < 1e-12
        )

    print("=" * 60)
    print(
        "PHASE 4B TEST PASSED"
    )
    print("=" * 60)

    print(
        "P(response):",
        probabilities,
    )

    print(
        "Best action:",
        ranked[0].action,
    )

    print(
        "Best EV:",
        f"{ranked[0].ev:+.6f}",
    )


if __name__ == "__main__":
    main()