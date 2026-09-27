"""
AdaptiveExact + optimized power auction shading.

Everything except the power auction is inherited from AdaptiveExact.

TRANSFORM denial is fixed at 1.10, based on the 210k-deal
confirmation sweep.

SHADE is supplied by the factory during research.
"""

from strategies.adaptive_exact import Bot as AdaptiveExact


class Bot(AdaptiveExact):

    name = "AdaptivePower"

    # Default value; research harness overrides this on construction.
    SHADE = 0.60
    DENIAL_WEIGHT = 1.10

    def _transform_value(self, obs):
        """
        Same TRANSFORM policy as the validated denial strategy.

        Flat hand:
            buy and use TRANSFORM.

        Decisive hand + opponent appears flat:
            buy to deny TRANSFORM.

        Decisive hand + no convincing read:
            do not contest.
        """

        swap = self._power_value(obs, "TRANSFORM")

        # Buy the actual swap for a flat hand.
        if abs(obs.k_mine) <= 1:
            return swap

        # Otherwise only pay for denial when the opponent
        # looks flat from an earlier quote.
        opp_k = self._opponent_k(obs)

        if (
            opp_k is not None
            and abs(opp_k) <= 2.0
        ):
            return swap * self.DENIAL_WEIGHT

        return 0.0

    def bid(self, obs, offered):
        """
        First-price power auction.

        Only SHADE is being optimized.
        """

        if not offered or obs.te_mine <= 0:
            return {}

        out = {}

        for name in offered:

            if name == "TRANSFORM":
                value_ticks = self._transform_value(obs)
            else:
                value_ticks = self._power_value(
                    obs,
                    name,
                )

            if value_ticks <= 0:
                continue

            fair_te = (
                value_ticks
                / self.config.TE_SALVAGE
            )

            bid_amount = int(
                fair_te * self.SHADE
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