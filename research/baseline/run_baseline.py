"""Freeze a local AdaptiveFinal V1 baseline.

Run from the QuantStorm project root:

    python research/baseline/run_baseline.py
"""

from pathlib import Path
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import play_match
from game_config import DEFAULT_CONFIG
from research.config import ResearchConfig

from strategies.adaptive_standalone import Bot as AdaptiveFinal
from strategies.rational import Bot as Rational
from strategies.adaptive_bidder import Bot as AdaptiveBidder
from strategies.naive_ev import Bot as NaiveEV


def summarize(name, result):
    deals = len(result.deals)
    values = [d.pnl[0] for d in result.deals]

    return {
        "name": name,
        "deals": deals,
        "total_pnl": result.pnl[0],
        "pnl_per_deal": result.pnl[0] / deals if deals else 0.0,
        "std_deal_pnl": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "warnings": len(result.bot_a_warnings),
        "violations": result.bot_a_violations,
        "clamps": result.bot_a_clamps,
        "forfeited": bool(result.forfeits[0]),
        "avg_ms": (
            sum(result.bot_a_times) / len(result.bot_a_times)
            if result.bot_a_times else 0.0
        ),
        "max_ms": max(result.bot_a_times) if result.bot_a_times else 0.0,
    }


def main():
    settings = ResearchConfig()
    seed = settings.baseline_seed
    deals = settings.baseline_deals

    matchups = [
        ("AdaptiveFinal_vs_Rational", AdaptiveFinal, Rational),
        ("AdaptiveFinal_vs_Adaptive", AdaptiveFinal, AdaptiveBidder),
        ("AdaptiveFinal_vs_NaiveEV", AdaptiveFinal, NaiveEV),
    ]

    results = []

    for name, bot_a, bot_b in matchups:
        result = play_match(
            bot_a,
            bot_b,
            config=DEFAULT_CONFIG,
            seed=seed,
            mirror=settings.mirror,
            n_deals=deals,
            verbose=False,
            bot_a_name="AdaptiveFinal",
            bot_b_name=bot_b.__name__,
        )

        summary = summarize(name, result)
        results.append(summary)

        print(
            f"{name}: "
            f"PnL={summary['total_pnl']:+.3f} "
            f"/deal={summary['pnl_per_deal']:+.5f} "
            f"warnings={summary['warnings']} "
            f"violations={summary['violations']} "
            f"clamps={summary['clamps']} "
            f"avg={summary['avg_ms']:.5f}ms "
            f"max={summary['max_ms']:.5f}ms"
        )

    output = ROOT / "research" / "results" / "phase1_baseline_reproduced.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
