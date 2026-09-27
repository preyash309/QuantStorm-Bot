"""One-command Phase 3 runner.

Run from the repository root:

    python research\opponent\run_phase3.py

This produces:
    research/results/phase3_policy_dataset.json
    research/results/phase3_empirical_policy.json
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import json
from pathlib import Path

from strategies.rational import Bot as Rational
from strategies.naive_ev import Bot as NaiveEV
from strategies.adaptive_bidder import Bot as Adaptive

from research.opponent.collector import collect
from research.opponent.policy import EmpiricalPolicy


def main():
    out = Path("research/results/phase3_policy_dataset.json")

    pairs = [
        ("Rational", Rational, "NaiveEV", NaiveEV),
        ("Rational", Rational, "Adaptive", Adaptive),
        ("NaiveEV", NaiveEV, "Adaptive", Adaptive),
    ]

    path, n_rows = collect(
        None,
        opponent_pairs=pairs,
        n_deals=1000,
        seed=930000,
        mirror=True,
        output=out,
    )

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]

    policy = EmpiricalPolicy(alpha=1.0).fit_rows(rows)

    policy_path = Path("research/results/phase3_empirical_policy.json")
    policy.save(policy_path)

    print("=" * 68)
    print("PHASE 3 OPPONENT POLICY DATASET COMPLETE")
    print("=" * 68)
    print(f"Rows:       {n_rows:,}")
    print(f"Dataset:    {path}")
    print(f"Policy:     {policy_path}")
    print(f"States:     {len(policy.counts):,}")
    print(f"Actions:    {sorted(policy.actions)}")
    print()
    print("Next:")
    print("  python research\\opponent\\analyze.py")


if __name__ == "__main__":
    main()
