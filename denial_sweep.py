"""
denial_sweep.py

Automatically searches for the best TRANSFORM denial weight.

No manual candidate creation.
No manual result comparison.

The only thing varied is:

    DENIAL_WEIGHT

Everything else is frozen to AdaptiveExact.
"""

from __future__ import annotations

import multiprocessing

from strategies.adaptive_exact_denial import (
    Bot as DenialBot,
)

from strategies.adaptive_bidder import (
    Bot as AdaptiveBot,
)

from research_harness import evaluate


# ============================================================
# Search space
# ============================================================

WEIGHTS = (
    1.00,
    1.10,
    1.20,
    1.25,
    1.30,
    1.40,
    1.50,
)


MATCHES = 30
DEALS_PER_MATCH = 1000
WORKERS = 4


# ============================================================
# Factory
# ============================================================

class DenialFactory:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self):
        bot = DenialBot()
        bot.DENIAL_WEIGHT = self.weight
        return bot


class AdaptiveFactory:
    def __call__(self):
        return AdaptiveBot()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TRANSFORM DENIAL WEIGHT SWEEP")
    print("=" * 70)
    print(
        f"Weights:       {WEIGHTS}"
    )
    print(
        f"Matches/weight: {MATCHES}"
    )
    print(
        f"Deals/match:    {DEALS_PER_MATCH}"
    )
    print(
        f"Total deals:    "
        f"{len(WEIGHTS) * MATCHES * DEALS_PER_MATCH:,}"
    )
    print("=" * 70)

    results = []

    for i, weight in enumerate(WEIGHTS):

        print()
        print(
            f"\n### "
            f"[{i + 1}/{len(WEIGHTS)}] "
            f"DENIAL_WEIGHT = {weight}"
        )

        global CURRENT_DENIAL_WEIGHT
        CURRENT_DENIAL_WEIGHT = weight

        result = evaluate(
            DenialFactory(weight),
            AdaptiveFactory(),
            n_matches=MATCHES,
            deals_per_match=DEALS_PER_MATCH,
            start_seed=100000 + i * 1000,
            workers=WORKERS,
            label=f"Denial={weight:.2f} vs Adaptive",
        )

        # ----------------------------------------------------
        # research_harness versions may return a summary
        # dictionary. Keep this tolerant.
        # ----------------------------------------------------

        if isinstance(result, dict):

            pnl_per_deal = result.get(
                "pnl_per_deal",
                result.get(
                    "PnL / deal",
                    None,
                ),
            )

            total_pnl = result.get(
                "total_pnl",
                result.get(
                    "Total PnL",
                    None,
                ),
            )

        else:

            pnl_per_deal = None
            total_pnl = None

        results.append(
            {
                "weight": weight,
                "pnl_per_deal": pnl_per_deal,
                "total_pnl": total_pnl,
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("DENIAL SWEEP COMPLETE")
    print("=" * 70)

    usable = [
        r
        for r in results
        if r["pnl_per_deal"] is not None
    ]

    if not usable:

        print()
        print(
            "The harness did not return summary "
            "objects, so use the printed RESULT "
            "blocks above."
        )

        print()
        print(
            "Best candidate must be selected from "
            "the displayed PnL/deal values."
        )

        return

    usable.sort(
        key=lambda x: x["pnl_per_deal"],
        reverse=True,
    )

    print()
    print(
        f"{'RANK':<6}"
        f"{'WEIGHT':<10}"
        f"{'PNL/DEAL':<14}"
        f"{'TOTAL PNL':<14}"
    )

    print("-" * 50)

    for rank, row in enumerate(
        usable,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{row['weight']:<10.2f}"
            f"{row['pnl_per_deal']:<14.6f}"
            f"{row['total_pnl']:<14.3f}"
        )

    best = usable[0]

    print()
    print("=" * 70)
    print(
        "BEST DENIAL WEIGHT = "
        f"{best['weight']:.2f}"
    )
    print(
        "PNL / DEAL = "
        f"{best['pnl_per_deal']:.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()