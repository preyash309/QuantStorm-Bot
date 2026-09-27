"""
Automatically sweep power-auction SHADE.

Fixed:
    AdaptiveExact negotiation
    TRANSFORM denial = 1.10
    measured POWER_VALUES

Varied:
    SHADE only
"""

from __future__ import annotations

import multiprocessing

from strategies.adaptive_power import Bot as PowerBot
from strategies.adaptive_bidder import Bot as AdaptiveBot

from research_harness import evaluate


# ============================================================
# Search space
# ============================================================

SHADES = (
    0.45,
    0.50,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.75,
)


MATCHES = 20
DEALS_PER_MATCH = 500
WORKERS = 4


# ============================================================
# Pickle-safe factory
# ============================================================

class ShadeFactory:

    def __init__(self, shade):
        self.shade = shade

    def __call__(self):

        bot = PowerBot()

        bot.SHADE = self.shade
        bot.DENIAL_WEIGHT = 1.10

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
    print("POWER SHADE SWEEP")
    print("=" * 70)

    print(
        f"Shades:          {SHADES}"
    )

    print(
        f"Matches/shade:   {MATCHES}"
    )

    print(
        f"Deals/match:     {DEALS_PER_MATCH}"
    )

    print(
        f"Total mirrored:  "
        f"{len(SHADES) * MATCHES * DEALS_PER_MATCH:,}"
    )

    print(
        "Fixed denial:    1.10"
    )

    print("=" * 70)

    for i, shade in enumerate(SHADES):

        print()
        print(
            f"### [{i + 1}/{len(SHADES)}] "
            f"SHADE = {shade:.3f}"
        )

        evaluate(
            ShadeFactory(shade),
            AdaptiveFactory(),
            n_matches=MATCHES,
            deals_per_match=DEALS_PER_MATCH,
            start_seed=120000 + i * 1000,
            workers=WORKERS,
            label=(
                f"SHADE={shade:.3f} "
                f"vs Adaptive"
            ),
        )

    print()
    print("=" * 70)
    print("SHADE SWEEP COMPLETE")
    print("=" * 70)

    print()
    print(
        "Select the best SHADE from the printed "
        "PnL/deal values."
    )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()