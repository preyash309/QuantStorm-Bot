from strategies.adaptive_power import Bot as PowerBot
from strategies.adaptive_bidder import Bot as AdaptiveBot

from research_harness import evaluate


TRANSFORM_SHADES = (
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    0.90,
)

MATCHES = 20
DEALS_PER_MATCH = 500
WORKERS = 4


class TransformShadeBot(PowerBot):

    name = "AdaptiveTransformShade"

    SHADE = 0.60
    DENIAL_WEIGHT = 1.10

    def bid(self, obs, offered):

        if not offered or obs.te_mine <= 0:
            return {}

        out = {}

        for name in offered:

            if name == "TRANSFORM":
                value_ticks = self._transform_value(obs)

                # Only TRANSFORM uses the experimental shade.
                shade = self.TRANSFORM_SHADE

            else:
                value_ticks = self._power_value(obs, name)

                # Everything else stays at the proven 0.60.
                shade = self.SHADE

            if value_ticks <= 0:
                continue

            fair_te = (
                value_ticks /
                self.config.TE_SALVAGE
            )

            bid_amount = int(
                fair_te * shade
            )

            bid_amount = max(
                0,
                min(
                    bid_amount,
                    obs.te_mine,
                ),
            )

            out[name] = bid_amount

        return out


class CandidateFactory:

    def __init__(self, shade):
        self.shade = shade

    def __call__(self):

        bot = TransformShadeBot()

        bot.SHADE = 0.60
        bot.TRANSFORM_SHADE = self.shade
        bot.DENIAL_WEIGHT = 1.10

        return bot


class AdaptiveFactory:

    def __call__(self):
        return AdaptiveBot()


def main():

    print("=" * 70)
    print("TRANSFORM SHADE SWEEP")
    print("=" * 70)

    print(
        f"Candidates: {TRANSFORM_SHADES}"
    )

    print(
        f"Matches/candidate: {MATCHES}"
    )

    print(
        f"Deals/match: {DEALS_PER_MATCH}"
    )

    print(
        f"Normal SHADE: 0.60"
    )

    print(
        f"DENIAL_WEIGHT: 1.10"
    )

    print("=" * 70)

    for i, shade in enumerate(TRANSFORM_SHADES):

        print()
        print(
            f"### [{i + 1}/{len(TRANSFORM_SHADES)}] "
            f"TRANSFORM SHADE = {shade:.3f}"
        )

        evaluate(
            CandidateFactory(shade),
            AdaptiveFactory(),

            n_matches=MATCHES,
            deals_per_match=DEALS_PER_MATCH,

            start_seed=140000 + i * 1000,

            workers=WORKERS,

            label=(
                f"TRANSFORM_SHADE={shade:.3f} "
                f"vs Adaptive"
            ),
        )

    print()
    print("=" * 70)
    print("TRANSFORM SHADE SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()