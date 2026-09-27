"""Unit test for Obs -> AgentState conversion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.simulator.state import obs_to_agent_state


class FakeObs:
    seat = 0
    round = 2
    my_revealed = (1, -1, 1, -1, 1, -1, 1, -1)
    te_mine = 17
    te_theirs = 12
    spread_cap = 7
    final_cap = 2
    is_maker = True
    powers_mine = frozenset({"FORESIGHT"})
    powers_theirs = frozenset()
    auction_log = (
        {"round": 1, "seat": 0, "power": "SUBSTITUTE", "cost": 4},
    )
    contracts = ()
    foresight = (1, -1, 1, -1)
    n_unknown_both = 24
    n_turns = 6


def main():
    s = obs_to_agent_state(FakeObs())

    assert s.seat == 0
    assert s.round == 2
    assert s.k_mine == 0
    assert s.te_mine == 17
    assert s.auction_log[0].power == "SUBSTITUTE"
    assert s.auction_log[0].cost == 4

    print("=" * 60)
    print("PHASE 1 STATE MODEL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
