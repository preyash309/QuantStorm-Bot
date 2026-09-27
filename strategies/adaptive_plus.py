"""
adaptive_plus.py

AdaptivePlus
============

Controlled improvement over the repository's AdaptiveBidder.

Changes from AdaptiveBidder:

    1. The underlying score posterior is represented exactly
       using the discrete +/-1 coin distribution.

    2. The posterior mean is calculated from the legal
       information state rather than being hard-coded as an
       informal expression.

Everything else remains aligned with the reference strategy:

    - POWER_VALUES
    - SHADE = 0.60
    - TRANSFORM logic
    - tight final-cap opening quote
    - SUBSTITUTE negotiation threshold
    - one-tick counter reduction

This file intentionally does NOT introduce:
    - RL
    - Monte Carlo inside the bot
    - neural networks
    - speculative Bayesian likelihood models
    - a new power-value surface
    - TRANSFORM denial tuning

Those are separate research questions.
"""

from __future__ import annotations

import random

from exact_value import (
    known_sum_from_obs,
    unknown_count_from_obs,
    expected_score,
    posterior_mean_from_obs,
    posterior_std_from_obs,
)

from strategies.adaptive_bidder import (
    POWER_VALUES,
    SHADE,
    FLAT_THRESHOLD,
    OPP_FLAT_THRESHOLD,
    DENIAL_WEIGHT,
)


class Bot:

    name = "AdaptivePlus"

    # ========================================================
    # Lifecycle
    # ========================================================

    def reset(
        self,
        seat,
        config,
        seed,
    ) -> None:

        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

        # Opening quote midpoints observed from the opponent.
        #
        # Key:
        #     round number
        #
        # Value:
        #     opening midpoint
        self._anchor = {}

        # Same information, explicitly named as opponent data.
        self._opp_anchor = {}

    # ========================================================
    # Opponent quote information
    # ========================================================

    def _get_anchor(
        self,
        obs,
        quote,
    ):
        """
        Capture the opponent's OPENING quote midpoint exactly once
        per round.

        We intentionally do not continuously re-anchor from counters.

        A counter is contaminated by both players' decisions, while
        the opening quote is the cleanest available public signal
        about the Maker's initial private information.
        """

        r = obs.round

        if r not in self._anchor:

            if (
                not obs.is_maker
                and quote is not None
            ):
                midpoint = (
                    quote[0]
                    + quote[1]
                ) / 2.0

                self._anchor[r] = midpoint
                self._opp_anchor[r] = midpoint

            else:
                self._anchor[r] = 0.0

        return self._anchor[r]

    def _opponent_k(
        self,
        obs,
    ):
        """
        Return the latest opponent opening-quote midpoint from
        an EARLIER round.

        We never use the current round because TRANSFORM denial
        needs a previous observable signal.
        """

        earlier_rounds = [
            r
            for r in self._opp_anchor
            if r < obs.round
        ]

        if not earlier_rounds:
            return None

        latest = max(
            earlier_rounds
        )

        return self._opp_anchor[
            latest
        ]

    # ========================================================
    # Exact information-state model
    # ========================================================

    def _exact_mean(
        self,
        obs,
    ) -> float:
        """
        Exact E[S | currently known information].

        Known contribution:

            own revealed coins
            +
            FORESIGHT-revealed opponent coins

        Unknown coins have expectation zero.
        """

        return posterior_mean_from_obs(
            obs,
            self.config,
        )

    def _exact_std(
        self,
        obs,
    ) -> float:
        """
        Exact posterior standard deviation.

        Not currently used in a decision rule, but kept as a
        cheap mathematical state variable for subsequent work.
        """

        return posterior_std_from_obs(
            obs,
            self.config,
        )

    # ========================================================
    # Score valuation
    # ========================================================

    def _value(
        self,
        obs,
        quote=None,
    ) -> float:
        """
        Estimate the final score S.

        Maker:
            exact posterior mean from legal information.

        Taker:
            exact posterior mean plus the opponent's opening
            quote signal.

        The opening quote is treated as an observed strategic
        signal rather than pretending it is an exact revelation.
        """

        exact_value = self._exact_mean(
            obs
        )

        # Maker has no opponent opening quote in the current
        # negotiation.
        if (
            obs.is_maker
            or quote is None
        ):
            return exact_value

        anchor = self._get_anchor(
            obs,
            quote,
        )

        # The opponent's midpoint is a noisy estimate of the
        # Maker's revealed score.
        #
        # We combine it with directly observed FORESIGHT exactly
        # as the reference bot does.
        if obs.foresight:

            leak_sum = sum(
                obs.foresight
            )

            anchor = (
                0.5 * anchor
                + 0.5 * leak_sum
            )

        # exact_value already contains our own revealed score
        # and FORESIGHT information. The opponent anchor estimates
        # the part of the opponent hand that we do not directly see.
        #
        # To avoid double-counting FORESIGHT, construct the score
        # from our own known contribution plus the inferred opponent
        # contribution.
        own_component = obs.k_mine

        return float(
            own_component
            + anchor
        )

    # ========================================================
    # Power valuation
    # ========================================================

    def _power_value(
        self,
        obs,
        name,
    ) -> float:
        """
        Existing calibrated per-round power surface.

        We deliberately retain this surface here.

        The exact posterior engine is not yet used to invent a
        new power-value table.
        """

        return POWER_VALUES.get(
            name,
            {},
        ).get(
            obs.round,
            0.5,
        )

    # ========================================================
    # TRANSFORM valuation
    # ========================================================

    def _transform_value(
        self,
        obs,
    ) -> float:
        """
        Value winning TRANSFORM.

        Flat hand:
            buy and use TRANSFORM.

        Decisive hand:
            normally don't pay to deny.

        Decisive + opponent appears flat:
            use DENIAL_WEIGHT.

        DENIAL_WEIGHT remains the repository's calibrated
        starting value of 0.0.
        """

        swap_value = self._power_value(
            obs,
            "TRANSFORM",
        )

        # Our hand is sufficiently flat:
        # TRANSFORM can improve it.
        if (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        ):
            return swap_value

        # Our hand is decisive.
        # Normally don't spend TE on denial.
        opponent_k = self._opponent_k(
            obs
        )

        if (
            opponent_k is not None
            and abs(opponent_k)
            <= OPP_FLAT_THRESHOLD
        ):
            return (
                swap_value
                * DENIAL_WEIGHT
            )

        return 0.0

    # ========================================================
    # Auction
    # ========================================================

    def bid(
        self,
        obs,
        offered,
    ):
        """
        Blind first-price auction.

        Convert tick value to TE:

            fair_TE = value / TE_SALVAGE

        Then shade:

            bid = floor(fair_TE * SHADE)

        Finally respect current remaining TE.
        """

        if (
            not offered
            or obs.te_mine <= 0
        ):
            return {}

        result = {}

        for name in offered:

            if name == "TRANSFORM":
                value = self._transform_value(
                    obs
                )
            else:
                value = self._power_value(
                    obs,
                    name,
                )

            if value <= 0:
                continue

            fair_te = (
                value
                / self.config.TE_SALVAGE
            )

            bid_amount = int(
                fair_te * SHADE
            )

            bid_amount = max(
                0,
                bid_amount,
            )

            bid_amount = min(
                bid_amount,
                obs.te_mine,
            )

            result[name] = bid_amount

        return result

    # ========================================================
    # Maker quote
    # ========================================================

    def quote(
        self,
        obs,
    ):
        """
        Open at the tightest legal width.

        Centre on the exact posterior mean, rounded to the game's
        integer price lattice.
        """

        value = round(
            self._exact_mean(obs)
        )

        width = obs.final_cap

        low = (
            value
            - width // 2
        )

        high = (
            low
            + width
        )

        return (
            low,
            high,
        )

    # ========================================================
    # Negotiation
    # ========================================================

    def respond(
        self,
        obs,
        quote,
        turn,
    ):
        """
        Decide whether to:

            ACCEPT_BUY
            ACCEPT_SELL
            COUNTER

        Positive edge means the quoted side is profitable under
        our current score estimate.
        """

        bid, ask = quote

        value = self._value(
            obs,
            quote,
        )

        edge_buy = (
            value
            - ask
        )

        edge_sell = (
            bid
            - value
        )

        # SUBSTITUTE makes downside less dangerous.
        threshold = 0.0

        if (
            "SUBSTITUTE"
            in obs.powers_mine
        ):
            threshold -= 1.0

        # Buy if it gives the strongest positive edge.
        if (
            edge_buy > threshold
            and edge_buy >= edge_sell
        ):
            return "ACCEPT_BUY"

        # Sell if it gives positive edge.
        if edge_sell > threshold:
            return "ACCEPT_SELL"

        # Otherwise shrink the spread by the required minimum.
        width = max(
            0,
            (ask - bid)
            - self.config.MIN_REDUCTION,
        )

        # Move the centre toward our value estimate while keeping
        # the resulting quote inside the previous quote.
        center = max(
            bid,
            min(
                round(value),
                ask - width,
            ),
        )

        return (
            "COUNTER",
            center,
            center + width,
        )

    # ========================================================
    # TRANSFORM execution
    # ========================================================

    def use_transform(
        self,
        obs,
    ):
        """
        Use TRANSFORM only when our revealed hand is flat.

        A decisive hand keeps its current hand and lets the
        purchased TRANSFORM be consumed defensively.
        """

        return (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        )