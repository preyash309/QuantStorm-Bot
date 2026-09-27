"""Smoke test proving the tracing layer does not change engine outcomes."""

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


def main():
    seed = 1234567

    # Deterministic 40-coin vector. The exact vector is supplied to both runs.
    coins = [
        1 if ((i * 17 + 3) % 7) < 3 else -1
        for i in range(DEFAULT_CONFIG.N_COINS)
    ]

    direct, _, _ = play_deal(
        NaiveEV(),
        Rational(),
        coins,
        DEFAULT_CONFIG,
        seed=seed,
        swap=False,
        invert_roles=False,
        verbose=False,
        bot_a_name="A",
        bot_b_name="B",
    )

    traced, trace_a, trace_b, _, _ = run_traced_deal(
        NaiveEV,
        Rational,
        seed=seed,
        coins=coins,
        swap=False,
        invert_roles=False,
        verbose=False,
        config=DEFAULT_CONFIG,
    )

    assert direct.pnl == traced.pnl
    assert direct.score == traced.score
    assert direct.te_left == traced.te_left
    assert direct.contracts == traced.contracts

    assert trace_a.events
    assert trace_b.events

    for event in trace_a.events + trace_b.events:
        s = event.state

        assert 1 <= s.round <= DEFAULT_CONFIG.N_ROUNDS
        assert s.n_turns == DEFAULT_CONFIG.N_TURNS
        assert s.te_mine >= 0
        assert s.te_theirs >= 0
        assert len(s.my_revealed) == (
            DEFAULT_CONFIG.REVEAL_PER_ROUND * s.round
        )
        assert s.n_unknown_both == DEFAULT_CONFIG.unknown_to_both(s.round)

    print("=" * 60)
    print("PHASE 1 ENGINE PARITY PASSED")
    print("=" * 60)
    print(f"PnL:          {direct.pnl}")
    print(f"Score:        {direct.score}")
    print(f"Contracts:    {len(direct.contracts)}")
    print(f"Trace events: A={len(trace_a.events)} B={len(trace_b.events)}")
    print("Official engine remains the execution oracle.")


if __name__ == "__main__":
    main()
