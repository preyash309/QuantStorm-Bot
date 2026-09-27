"""
research_harness.py

Parallel evaluation harness.

Run from the repository root:

    python research_harness.py
"""

from __future__ import annotations

import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

from engine import play_match
from game_config import GameConfig

from strategies.adaptive_bidder import (
    Bot as AdaptiveBidder,
)

from strategies.rational import (
    Bot as Rational,
)

from strategies.naive_ev import (
    Bot as NaiveEV,
)

from strategies.adaptive_plus import (
    Bot as AdaptivePlus,
)

from strategies.adaptive_exact import (
    Bot as AdaptiveExact,
)


# ============================================================
# Factories
# ============================================================

def make_adaptive():
    return AdaptiveBidder()


def make_adaptive_plus():
    return AdaptivePlus()


def make_rational():
    return Rational()


def make_naive():
    return NaiveEV()

def make_adaptive_exact():
    return AdaptiveExact()


# ============================================================
# Match result
# ============================================================

@dataclass
class MatchSummary:

    seed: int

    pnl_a: float
    pnl_b: float

    deals: int

    mean_time_a_ms: float
    mean_time_b_ms: float

    max_time_a_ms: float
    max_time_b_ms: float

    violations_a: int
    violations_b: int

    clamps_a: int
    clamps_b: int

    forfeits_a: bool
    forfeits_b: bool


# ============================================================
# One independent match
# ============================================================

def run_one_match(
    seed: int,
    bot_a_factory,
    bot_b_factory,
    n_deals: int,
):

    config = GameConfig()

    result = play_match(
        bot_a_factory,
        bot_b_factory,
        config=config,
        seed=seed,
        mirror=True,
        n_deals=n_deals,
        verbose=False,
    )

    times_a = result.bot_a_times
    times_b = result.bot_b_times

    return MatchSummary(

        seed=seed,

        pnl_a=result.pnl[0],
        pnl_b=result.pnl[1],

        deals=len(result.deals),

        mean_time_a_ms=(
            statistics.mean(times_a)
            if times_a
            else 0.0
        ),

        mean_time_b_ms=(
            statistics.mean(times_b)
            if times_b
            else 0.0
        ),

        max_time_a_ms=(
            max(times_a)
            if times_a
            else 0.0
        ),

        max_time_b_ms=(
            max(times_b)
            if times_b
            else 0.0
        ),

        violations_a=result.bot_a_violations,
        violations_b=result.bot_b_violations,

        clamps_a=result.bot_a_clamps,
        clamps_b=result.bot_b_clamps,

        forfeits_a=result.forfeits[0],
        forfeits_b=result.forfeits[1],
    )


# ============================================================
# Parallel evaluation
# ============================================================

def evaluate(
    bot_a_factory,
    bot_b_factory,
    *,
    n_matches: int,
    deals_per_match: int,
    start_seed: int,
    workers: int | None = None,
    label: str,
):

    if workers is None:
        workers = max(
            1,
            (os.cpu_count() or 2) - 1,
        )

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {label}")
    print("=" * 70)

    print(
        f"Matches:     {n_matches}"
    )

    print(
        f"Deals/match: {deals_per_match}"
    )

    print(
        f"Workers:     {workers}"
    )

    print("=" * 70)

    job = partial(
        run_one_match,
        bot_a_factory=bot_a_factory,
        bot_b_factory=bot_b_factory,
        n_deals=deals_per_match,
    )

    summaries = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                job,
                start_seed + i,
            )
            for i in range(n_matches)
        ]

        for index, future in enumerate(
            as_completed(futures),
            start=1,
        ):

            summary = future.result()

            summaries.append(
                summary
            )

            print(
                f"[{index:>3}/{n_matches}] "
                f"seed={summary.seed:<6} "
                f"PnL={summary.pnl_a:+9.2f} "
                f"avg={summary.mean_time_a_ms:.4f}ms "
                f"max={summary.max_time_a_ms:.4f}ms"
            )

    # ========================================================
    # Aggregate
    # ========================================================

    pnl = [
        summary.pnl_a
        for summary in summaries
    ]

    total_pnl = sum(pnl)

    total_deals = sum(
        summary.deals
        for summary in summaries
    )

    mean_pnl = (
        statistics.mean(pnl)
        if pnl
        else 0.0
    )

    std_pnl = (
        statistics.stdev(pnl)
        if len(pnl) >= 2
        else 0.0
    )

    avg_time = (
        statistics.mean(
            summary.mean_time_a_ms
            for summary in summaries
        )
        if summaries
        else 0.0
    )

    max_time = max(
        (
            summary.max_time_a_ms
            for summary in summaries
        ),
        default=0.0,
    )

    violations = sum(
        summary.violations_a
        for summary in summaries
    )

    clamps = sum(
        summary.clamps_a
        for summary in summaries
    )

    forfeits = sum(
        summary.forfeits_a
        for summary in summaries
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Total PnL:           "
        f"{total_pnl:+.3f}"
    )

    print(
        f"Total mirrored deals:"
        f" {total_deals}"
    )

    if total_deals:
        print(
            f"PnL / deal:          "
            f"{total_pnl / total_deals:+.5f}"
        )

    print(
        f"Mean PnL / match:    "
        f"{mean_pnl:+.5f}"
    )

    print(
        f"Std PnL / match:     "
        f"{std_pnl:.5f}"
    )

    print(
        f"Avg call time:       "
        f"{avg_time:.5f} ms"
    )

    print(
        f"Max call time:       "
        f"{max_time:.5f} ms"
    )

    print(
        f"Violations:          "
        f"{violations}"
    )

    print(
        f"Clamps:              "
        f"{clamps}"
    )

    print(
        f"Forfeits:            "
        f"{forfeits}"
    )

    print("-" * 70)

    return summaries


# ============================================================
# Current required evaluation
# ============================================================

# def main():

#     # --------------------------------------------------------
#     # 1. Verify AdaptivePlus against the reference Adaptive.
#     # --------------------------------------------------------

#     evaluate(
#         make_adaptive_plus,
#         make_adaptive,

#         n_matches=10,
#         deals_per_match=100,

#         start_seed=2000,

#         workers=4,

#         label=(
#             "AdaptivePlus vs Adaptive"
#         ),
#     )

#     evaluate(
#         make_adaptive_exact,
#         make_adaptive,
#         n_matches=20,
#         deals_per_match=200,
#         start_seed=3000,
#         workers=4,
#         label="AdaptiveExact vs Adaptive",
#     )

#     evaluate(
#     make_adaptive_exact,
#     make_rational,
#     n_matches=20,
#     deals_per_match=200,
#     start_seed=4000,
#     workers=4,
#     label="AdaptiveExact vs Rational",
#     )

# evaluate(
#     make_adaptive_exact,
#     make_naive,
#     n_matches=20,
#     deals_per_match=200,
#     start_seed=5000,
#     workers=4,
#     label="AdaptiveExact vs NaiveEV",
#     )


if __name__ == "__main__":
    evaluate(
        make_adaptive_plus,
        make_adaptive,

        n_matches=50,
        deals_per_match=500,

        start_seed=00000,

        workers=4,

        label=(
            "AdaptivePlus vs Adaptive"
        ),
    )

    evaluate(
        make_adaptive_exact,
        make_adaptive,
        n_matches=50,
        deals_per_match=500,
        start_seed=10000,
        workers=4,
        label="AdaptiveExact vs Adaptive",
    )

    evaluate(
        make_adaptive_exact,
        make_rational,
        n_matches=50,
        deals_per_match=500,
        start_seed=30000,
        workers=4,
        label="AdaptiveExact vs Rational",
    )

    evaluate(
            make_adaptive_exact,
            make_naive,
            n_matches=50,
            deals_per_match=500,
            start_seed=40000,
            workers=4,
            label="AdaptiveExact vs NaiveEV",
        )