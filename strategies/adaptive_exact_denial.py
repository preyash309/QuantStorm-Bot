"""
AdaptiveExact + TRANSFORM denial sweep candidate.

Everything is inherited from AdaptiveExact.

The ONLY changed quantity is DENIAL_WEIGHT.
"""

from strategies.adaptive_exact import Bot as AdaptiveExact


class Bot(AdaptiveExact):

    name = "AdaptiveExactDenial"

    # This is overwritten automatically by the sweep.
    DENIAL_WEIGHT = 0.0

    def _transform_value(self, obs):
        """
        Exact same TRANSFORM logic as AdaptiveExact,
        except denial weight is configurable.
        """

        swap = self._power_value(
            obs,
            "TRANSFORM",
        )

        # Flat hand:
        # buy TRANSFORM to use the swap.
        if abs(obs.k_mine) <= 1:
            return swap

        # Decisive hand:
        # only pay for denial when our historical quote
        # read says the opponent looks flat.
        opp_k = self._opponent_k(obs)

        if (
            opp_k is not None
            and abs(opp_k) <= 2.0
        ):
            return (
                swap
                * self.DENIAL_WEIGHT
            )

        return 0.0