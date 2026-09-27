"""
Final robustness evaluation for AdaptiveFinal.

No parameters are optimized here.

Fresh seeds are used so these results are independent
of the previous development sweeps.
"""

import multiprocessing

from strategies.adaptive_final import Bot as AdaptiveFinal
from strategies.adaptive_bidder import Bot as Adaptive
from strategies.rational import Bot as Rational
from strategies.naive_ev import Bot as NaiveEV

from research_harness import evaluate


# ============================================================
# Configuration
# ============================================================

MATCHES = 50
DEALS_PER_MATCH = 1000
WORKERS = 4


# ============================================================
# Pickle-safe factories
# ============================================================

class FinalFactory:

    def __call__(self):
        return AdaptiveFinal()


class AdaptiveFactory:

    def __call__(self):
        return Adaptive()


class RationalFactory:

    def __call__(self):
        return Rational()


class NaiveEVFactory:

    def __call__(self):
        return NaiveEV()


# ============================================================
# Evaluation helper
# ============================================================

def run_matchup(
    name,
    opponent_factory,
    seed,
):

    print()
    print("=" * 70)
    print(f"FINAL ROBUSTNESS: AdaptiveFinal vs {name}")
    print("=" * 70)

    print(
        f"Matches:     {MATCHES}"
    )

    print(
        f"Deals/match: {DEALS_PER_MATCH}"
    )

    print(
        f"Workers:     {WORKERS}"
    )

    print(
        "Frozen parameters:"
    )

    print(
        "  SHADE              = 0.60"
    )

    print(
        "  TRANSFORM_SHADE    = 0.60"
    )

    print(
        "  DENIAL_WEIGHT      = 1.10"
    )

    print("=" * 70)

    return evaluate(
        FinalFactory(),
        opponent_factory,

        n_matches=MATCHES,
        deals_per_match=DEALS_PER_MATCH,

        start_seed=seed,

        workers=WORKERS,

        label=(
            f"AdaptiveFinal vs {name}"
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL ADAPTIVEFINAL ROBUSTNESS SUITE")
    print("=" * 70)

    print(
        f"Total mirrored deals: "
        f"{3 * MATCHES * DEALS_PER_MATCH:,}"
    )

    print(
        "Fresh seed ranges:"
    )

    print(
        "  Adaptive : 200000+"
    )

    print(
        "  Rational : 210000+"
    )

    print(
        "  NaiveEV  : 220000+"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Adaptive
    # --------------------------------------------------------

    run_matchup(
        "Adaptive",
        AdaptiveFactory(),
        200000,
    )

    # --------------------------------------------------------
    # 2. Rational
    # --------------------------------------------------------

    run_matchup(
        "Rational",
        RationalFactory(),
        210000,
    )

    # --------------------------------------------------------
    # 3. NaiveEV
    # --------------------------------------------------------

    run_matchup(
        "NaiveEV",
        NaiveEVFactory(),
        220000,
    )

    print()
    print("=" * 70)
    print("FINAL ROBUSTNESS SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()