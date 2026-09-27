from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
    ),
)

from research.ev.quote_opponent import (
    QuoteOpponentModel,
)


def main():

    rows = [

        {
            "method": "respond",
            "action": "ACCEPT_BUY",
            "turn": 2,
            "quote": [0, 6],
            "obs": {
                "round": 2,
                "te_mine": 12,
                "te_theirs": 12,
                "is_maker": True,
                "powers_mine": [],
                "powers_theirs": [],
            },
        },

        {
            "method": "respond",
            "action": "ACCEPT_BUY",
            "turn": 2,
            "quote": [0, 6],
            "obs": {
                "round": 2,
                "te_mine": 12,
                "te_theirs": 12,
                "is_maker": True,
                "powers_mine": [],
                "powers_theirs": [],
            },
        },

        {
            "method": "respond",
            "action": "ACCEPT_SELL",
            "turn": 2,
            "quote": [0, 6],
            "obs": {
                "round": 2,
                "te_mine": 12,
                "te_theirs": 12,
                "is_maker": True,
                "powers_mine": [],
                "powers_theirs": [],
            },
        },

        {
            "method": "respond",
            "action": "COUNTER",
            "turn": 2,
            "quote": [0, 6],
            "obs": {
                "round": 2,
                "te_mine": 12,
                "te_theirs": 12,
                "is_maker": True,
                "powers_mine": [],
                "powers_theirs": [],
            },
        },
    ]

    model = (
        QuoteOpponentModel()
        .fit(rows)
    )

    assert model.n_rows == 4

    assert model.n_states == 1

    p = model.predict(
        round_no=2,
        te_mine=12,
        te_theirs=12,
        is_maker=True,
        powers_mine=[],
        powers_theirs=[],
        turn=2,
        quote=(0, 6),
    )

    assert abs(
        sum(p.values()) - 1.0
    ) < 1e-12

    assert p[
        "ACCEPT_BUY"
    ] > p[
        "ACCEPT_SELL"
    ]

    assert p[
        "ACCEPT_BUY"
    ] > p[
        "COUNTER"
    ]

    # Changing the quote must change the state key.
    p2 = model.predict(
        round_no=2,
        te_mine=12,
        te_theirs=12,
        is_maker=True,
        powers_mine=[],
        powers_theirs=[],
        turn=2,
        quote=(-4, 4),
    )

    assert abs(
        sum(p2.values()) - 1.0
    ) < 1e-12

    print("=" * 60)
    print(
        "PHASE 4C TEST PASSED"
    )
    print("=" * 60)

    print(
        "Known quote probabilities:",
        p,
    )

    print(
        "Unseen quote probabilities:",
        p2,
    )

    print(
        "States:",
        model.n_states,
    )


if __name__ == "__main__":
    main()