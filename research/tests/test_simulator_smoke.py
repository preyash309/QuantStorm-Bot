import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from game_config import DEFAULT_CONFIG
from research.simulator.actions import AcceptBuy
from research.simulator.simulator import ExactSimulator

def main():
    c = DEFAULT_CONFIG
    sim = ExactSimulator(c)

    coins = [1 if i % 3 else -1 for i in range(c.N_COINS)]
    actions = []

    for r in range(1, c.N_ROUNDS+1):
        floor = c.final_cap(r)
        low = -(floor // 2)
        actions.append({
            "bids": {0: {}, 1: {}},
            "transform": {0: False, 1: False},
            "opening_quote": (low, low+floor),
            "responses": [AcceptBuy()],
        })

    result = sim.run_explicit_deal(
        coins=coins, round_actions=actions, seed=12345
    )

    assert len(result.contracts) == c.N_ROUNDS
    assert result.score == sum(coins)
    assert result.te_left == (c.TE_BUDGET, c.TE_BUDGET)
    assert abs(sum(result.pnl)) < c.ZERO_SUM_TOL

    print("="*60)
    print("PHASE 2 SIMULATOR SMOKE TEST PASSED")
    print("="*60)
    print("Score:", result.score)
    print("PnL:", result.pnl)


if __name__ == "__main__":
    main()