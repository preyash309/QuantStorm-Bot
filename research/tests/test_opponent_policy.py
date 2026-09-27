from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.opponent.features import state_key, action_label
from research.opponent.policy import EmpiricalPolicy


def main():
    obs = {
        "round": 3,
        "k_mine": 2,
        "te_mine": 12,
        "te_theirs": 16,
        "spread_cap": 8,
        "final_cap": 4,
        "is_maker": False,
        "powers_mine": [],
        "powers_theirs": [],
        "foresight_sum": 0,
        "foresight_n": 0,
    }

    key = state_key(
        obs,
        method="respond",
        quote=(0, 6),
        turn=2,
        offered=(),
    )

    p = EmpiricalPolicy(alpha=1.0)

    rows = [
        {
            "method": "respond",
            "obs": obs,
            "quote": [0, 6],
            "turn": 2,
            "action": "ACCEPT_BUY",
        },
        {
            "method": "respond",
            "obs": obs,
            "quote": [0, 6],
            "turn": 2,
            "action": "ACCEPT_BUY",
        },
        {
            "method": "respond",
            "obs": obs,
            "quote": [0, 6],
            "turn": 2,
            "action": ["COUNTER", 2, 5],
        },
    ]

    p.fit_rows(rows)
    probs = p.probabilities(key)

    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert probs["ACCEPT_BUY"] > probs["COUNTER"]
    assert p.top_action(key) == "ACCEPT_BUY"
    assert action_label(["COUNTER", 1, 4]) == "COUNTER"

    print("=" * 68)
    print("PHASE 3 OPPONENT POLICY TEST PASSED")
    print("=" * 68)
    print("Probabilities:", probs)


if __name__ == "__main__":
    main()
