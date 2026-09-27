"""
Final integrated QuantStorm candidate.

Frozen parameters from completed sweeps:

    Global power SHADE      = 0.60
    TRANSFORM SHADE         = 0.60
    TRANSFORM DENIAL_WEIGHT = 1.10

No other strategy behavior is changed.
"""

from strategies.adaptive_power import Bot as AdaptivePower


class Bot(AdaptivePower):

    name = "AdaptiveFinal"

    # Proven global power-auction shade.
    SHADE = 0.60

    # TRANSFORM-specific shade.
    # The dedicated sweep did not beat the global 0.60.
    TRANSFORM_SHADE = 0.60

    # Best confirmed region from the large denial sweep.
    DENIAL_WEIGHT = 1.10