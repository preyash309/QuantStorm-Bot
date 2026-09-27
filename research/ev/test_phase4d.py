from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from research.ev.quote_opponent import QuoteOpponentModel
from research.ev.continuation import evaluate_all_actions


class FakeObs:

    round = 2
    is_maker = False
    k_mine = 4

    foresight = ()
    foresight_sum = 0

    te_mine = 12
    te_theirs = 12

    powers_mine = frozenset()
    powers_theirs = frozenset()

    final_cap = 2
    spread_cap = 8


def main():

    obs = FakeObs()

    model = QuoteOpponentModel()

    rows = []

    # Strong preference for ACCEPT_BUY
    for _ in range(8):

        rows.append({
            "method": "respond",
            "action": "ACCEPT_BUY",
            "turn": 2,
            "quote": [0, 6],
            "obs": {
                "round": 2,
                "te_mine": 12,
                "te_theirs": 12,
                "is_maker": False,
                "powers_mine": [],
                "powers_theirs": [],
            },
        })

    rows.append({
        "method": "respond",
        "action": "ACCEPT_SELL",
        "turn": 2,
        "quote": [0, 6],
        "obs": {
            "round": 2,
            "te_mine": 12,
            "te_theirs": 12,
            "is_maker": False,
            "powers_mine": [],
            "powers_theirs": [],
        },
    })

    rows.append({
        "method": "respond",
        "action": "COUNTER",
        "turn": 2,
        "quote": [0, 6],
        "obs": {
            "round": 2,
            "te_mine": 12,
            "te_theirs": 12,
            "is_maker": False,
            "powers_mine": [],
            "powers_theirs": [],
        },
    })

    model.fit(rows)

    assert model.n_rows == 10
    assert model.n_states == 1

    probabilities = model.predict(
        round_no=2,
        te_mine=12,
        te_theirs=12,
        is_maker=False,
        powers_mine=[],
        powers_theirs=[],
        turn=2,
        quote=(0, 6),
    )

    assert abs(
        sum(probabilities.values()) - 1.0
    ) < 1e-12

    assert (
        probabilities["ACCEPT_BUY"]
        >
        probabilities["ACCEPT_SELL"]
    )

    print("=" * 60)
    print("PHASE 4D TEST PASSED")
    print("=" * 60)

    print(
        "Quote-conditioned probabilities:",
        probabilities,
    )


if __name__ == "__main__":
    main()