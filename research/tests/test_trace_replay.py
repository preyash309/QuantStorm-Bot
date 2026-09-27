from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import play_deal
from game_config import DEFAULT_CONFIG
from strategies.naive_ev import Bot as NaiveEV
from strategies.rational import Bot as Rational
from research.simulator.runner import run_traced_deal
from research.simulator.replay import trace_to_round_actions
from research.simulator.simulator import ExactSimulator

def main():
    seed = 712345
    coins = [1 if ((i*11+5)%9)<4 else -1 for i in range(DEFAULT_CONFIG.N_COINS)]

    official, _, _ = play_deal(
        NaiveEV(), Rational(), coins, DEFAULT_CONFIG,
        seed=seed, swap=False, invert_roles=False,
        verbose=False, bot_a_name="A", bot_b_name="B"
    )

    traced, trace_a, trace_b, _, _ = run_traced_deal(
        NaiveEV, Rational, seed=seed, coins=coins,
        swap=False, invert_roles=False, verbose=False,
        config=DEFAULT_CONFIG
    )
    assert official.pnl == traced.pnl

    actions = trace_to_round_actions(
        trace_a, trace_b, DEFAULT_CONFIG.N_ROUNDS
    )

    research = ExactSimulator(DEFAULT_CONFIG).run_explicit_deal(
        coins=coins, round_actions=actions, seed=seed
    )

    assert research.score == official.score
    assert research.te_left == official.te_left
    assert len(research.contracts) == len(official.contracts)

    for a, b in zip(research.contracts, official.contracts):
        assert (a.round, a.price, a.long_seat, a.forced, a.forcer,
                a.shift, a.maker_seat, a.open_bid, a.open_ask) == (
            b.round, b.price, b.long_seat, b.forced, b.forcer,
            b.shift, b.maker_seat, b.open_bid, b.open_ask
        )

    assert abs(research.pnl[0]-official.pnl[0]) < 1e-9
    assert abs(research.pnl[1]-official.pnl[1]) < 1e-9

    print("="*60)
    print("PHASE 2 TRACE REPLAY PARITY PASSED")
    print("="*60)
    print("Official PnL:", official.pnl)
    print("Research PnL:", research.pnl)
    print("Contracts:", len(research.contracts))

if __name__ == "__main__":
    main()