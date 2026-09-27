# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal v4 — standalone QuantStorm submission.

Built directly on test3.

Changes from test3:
  1. Confidence-weighted historical opponent estimate.
  2. Valid zero-valued opponent estimates are preserved.
  3. Historical quotes are blended instead of blindly trusting the latest quote.
  4. Late FORESIGHT auction uses adaptive minimum-winning logic rather
     than unconditional all-in bidding when behind.
  5. Existing test3 strategic mechanisms are preserved:
       - quote discounting
       - Maker historical information
       - exact FORESIGHT residual estimation
       - dynamic TRANSFORM valuation
       - TE pacing
       - late FORESIGHT prioritization
       - uncertainty-aware negotiation
       - power-aware stalling
       - mixed-strategy quote/counter perturbation
       - exact final-turn EV
"""

import random
import math


# ============================================================
# POWER VALUES
# ============================================================

POWER_VALUES = {
    "FORESIGHT": {
        1: 0.76,
        2: 1.16,
        3: 1.48,
        4: 1.97,
        5: 2.02,
    },

    "TRICK_ROOM": {
        1: 1.14,
        2: 0.00,
        3: 0.00,
        4: 0.60,
        5: 0.52,
    },

    "SUBSTITUTE": {
        1: 1.46,
        2: 1.15,
        3: 0.95,
        4: 0.57,
        5: 0.29,
    },

    "STEALTH_ROCK": {
        1: 1.51,
        2: 0.75,
        3: 0.75,
        4: 0.75,
        5: 0.00,
    },

    "TRANSFORM": {
        1: 1.58,
        2: 1.24,
        3: 1.31,
        4: 0.00,
        5: 0.00,
    },
}


# ============================================================
# CALIBRATED PARAMETERS
# ============================================================

SHADE = 0.60

DENIAL_WEIGHT = 0.40

FLAT_THRESHOLD = 1

OPP_FLAT_THRESHOLD = 2.0


# ============================================================
# BOT
# ============================================================

class Bot:

    name = "AdaptiveFinal_v4"

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(
        self,
        seat,
        config,
        seed,
    ):

        self.seat = seat
        self.config = config

        self.rng = random.Random(seed)

        # Opponent quote history.
        #
        # round -> discounted opponent anchor
        #
        self._opp_anchor = {}

        # Our cached anchor for current round.
        self._anchor = {}


    # ========================================================
    # OPPONENT QUOTE ESTIMATION
    # ========================================================

    def _quote_anchor(
        self,
        obs,
        quote,
    ):

        r = int(obs.round)

        if quote is None:
            return 0.0

        raw_midpoint = (
            float(quote[0])
            + float(quote[1])
        ) / 2.0

        # Early quotes are strategically noisy.
        #
        # Later quotes contain substantially more information.
        alpha = {
            1: 0.50,
            2: 0.65,
            3: 0.80,
            4: 0.90,
            5: 0.95,
        }.get(
            r,
            0.50,
        )

        discounted = (
            raw_midpoint
            * alpha
        )

        # Mathematically possible opponent hand range.
        max_revealed = 4 * r

        clamped = max(
            -max_revealed,
            min(
                max_revealed,
                discounted,
            ),
        )

        return float(clamped)


    def _get_anchor(
        self,
        obs,
        quote,
    ):
        """
        Obtain current-round opponent anchor.

        When we are responding to an opponent quote,
        update the historical record for this round.

        Important difference from test3:
        zero is a valid estimate and is retained.
        """

        r = int(obs.round)

        if r not in self._anchor:

            if (
                not obs.is_maker
                and quote is not None
            ):

                anchor = self._quote_anchor(
                    obs,
                    quote,
                )

                self._anchor[r] = anchor

                # Preserve zero as legitimate information.
                self._opp_anchor[r] = anchor

            else:

                self._anchor[r] = 0.0

        return self._anchor[r]


    def _historical_opponent_estimate(
        self,
        obs,
    ):
        """
        Estimate opponent hand from all available historical quotes.

        Instead of:

            latest quote wins completely

        use recency-weighted evidence.

        The latest round receives the strongest weight, but
        earlier observations prevent one noisy quote from
        completely replacing the historical estimate.
        """

        current_round = int(
            obs.round
        )

        history = [
            (
                r,
                value,
            )
            for r, value
            in self._opp_anchor.items()
            if r < current_round
        ]

        if not history:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for r, value in history:

            age = (
                current_round
                - r
            )

            # Recency weighting.
            #
            # Latest previous observation:
            #   age = 1 -> weight 1.00
            #
            # Older observations decay gradually.
            weight = 1.0 / (
                1.0
                + 0.75 * (age - 1)
            )

            weighted_sum += (
                weight
                * float(value)
            )

            total_weight += weight

        if total_weight <= 0:
            return 0.0

        estimate = (
            weighted_sum
            / total_weight
        )

        max_revealed = (
            4 * current_round
        )

        return max(
            -max_revealed,
            min(
                max_revealed,
                estimate,
            ),
        )


    def _latest_opp_k(
        self,
        obs,
    ):
        """
        Compatibility helper.

        Unlike test3, zero is not ignored.
        """

        return self._historical_opponent_estimate(
            obs
        )


    # ========================================================
    # VALUE ESTIMATION
    # ========================================================

    def _value(
        self,
        obs,
        quote=None,
    ):
        """
        Estimate current economic value.

        Taker:
            current quote + exact information

        Maker:
            historical opponent information + exact information
        """

        if (
            not obs.is_maker
            and quote is not None
        ):

            anchor = self._get_anchor(
                obs,
                quote,
            )

        else:

            anchor = (
                self._historical_opponent_estimate(
                    obs
                )
            )

        # ----------------------------------------------------
        # Exact FORESIGHT information
        # ----------------------------------------------------

        if obs.foresight:

            n_seen = len(
                obs.foresight
            )

            n_total = (
                4 * int(obs.round)
            )

            if n_total > 0:

                unseen_fraction = max(
                    0.0,
                    1.0
                    - (
                        n_seen
                        / n_total
                    ),
                )

            else:

                unseen_fraction = 0.0

            # Exact leaked coins +
            # estimated unseen remainder.
            opponent_estimate = (
                float(
                    sum(obs.foresight)
                )
                +
                unseen_fraction
                * anchor
            )

        else:

            opponent_estimate = (
                anchor
            )

        return float(
            obs.k_mine
            + opponent_estimate
        )


    # ========================================================
    # POWER VALUE
    # ========================================================

    def _power_value(
        self,
        obs,
        name,
    ):

        return (
            POWER_VALUES
            .get(
                name,
                {},
            )
            .get(
                int(obs.round),
                0.5,
            )
        )


    # ========================================================
    # TRANSFORM
    # ========================================================

    def _transform_value(
        self,
        obs,
    ):

        swap = self._power_value(
            obs,
            "TRANSFORM",
        )

        opp_k = (
            self._historical_opponent_estimate(
                obs
            )
        )

        # ----------------------------------------------------
        # We are flatter than opponent.
        #
        # Swapping becomes valuable because the opponent
        # has a more decisive hand.
        # ----------------------------------------------------

        if (
            abs(obs.k_mine)
            < abs(opp_k)
            or
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        ):

            return swap

        # ----------------------------------------------------
        # Defensive denial.
        # ----------------------------------------------------

        if (
            abs(opp_k)
            <= OPP_FLAT_THRESHOLD
            and
            abs(obs.k_mine)
            >= 3
        ):

            return (
                swap
                * DENIAL_WEIGHT
            )

        return 0.0


    # ========================================================
    # POWER BID
    # ========================================================

    def bid(
        self,
        obs,
        offered,
    ):

        if (
            not offered
            or obs.te_mine <= 0
        ):
            return {}

        # ----------------------------------------------------
        # Dynamic shading from TE parity.
        # ----------------------------------------------------

        te_advantage = (
            obs.te_mine
            - obs.te_theirs
        )

        dynamic_shade = min(
            0.85,
            max(
                0.35,
                SHADE
                + (
                    te_advantage
                    * 0.015
                ),
            ),
        )

        out = {}

        for name in offered:

            if name == "TRANSFORM":

                value = (
                    self._transform_value(
                        obs
                    )
                )

            else:

                value = (
                    self._power_value(
                        obs,
                        name,
                    )
                )

            if value <= 0:
                continue

            fair_te = (
                value
                / self.config.TE_SALVAGE
            )

            calculated_bid = int(
                fair_te
                * dynamic_shade
            )

            # =================================================
            # LATE FORESIGHT PRIORITY
            # =================================================

            if (
                name == "FORESIGHT"
                and obs.round >= 4
            ):

                # ------------------------------------------------
                # If we have TE advantage:
                #
                # win for the minimum necessary price.
                # ------------------------------------------------

                if (
                    obs.te_mine
                    > obs.te_theirs
                ):

                    calculated_bid = min(
                        obs.te_mine,
                        obs.te_theirs + 1,
                    )

                else:

                    # ------------------------------------------------
                    # TEST4 CHANGE:
                    #
                    # Test3 went all-in when behind.
                    #
                    # Instead, distinguish between:
                    #
                    #   modest disadvantage:
                    #       bid aggressively but preserve 1 TE
                    #
                    #   severe disadvantage:
                    #       all-in may actually be justified.
                    # ------------------------------------------------

                    te_gap = (
                        obs.te_theirs
                        - obs.te_mine
                    )

                    if te_gap <= 2:

                        calculated_bid = (
                            obs.te_mine
                        )

                    else:

                        # We cannot realistically win cheaply.
                        # Preserve one unit of optionality.
                        calculated_bid = max(
                            0,
                            obs.te_mine - 1,
                        )

            # ------------------------------------------------
            # General auction lockout.
            # ------------------------------------------------

            elif (
                calculated_bid
                > obs.te_theirs
                and
                obs.te_mine
                > obs.te_theirs
            ):

                calculated_bid = (
                    obs.te_theirs
                    + 1
                )

            bid_amount = max(
                0,
                min(
                    calculated_bid,
                    obs.te_mine,
                ),
            )

            out[name] = bid_amount

        # ====================================================
        # TE PACING
        # ====================================================

        if (
            obs.round <= 2
            and obs.te_mine > 8
        ):

            for name in out:

                # Stealth Rock retains some persistent value.
                if name != "STEALTH_ROCK":

                    out[name] = min(
                        out[name],
                        obs.te_mine - 8,
                    )

        elif (
            obs.round == 3
            and obs.te_mine > 6
        ):

            for name in out:

                out[name] = min(
                    out[name],
                    obs.te_mine - 6,
                )

        return out


    # ========================================================
    # QUOTE
    # ========================================================

    def quote(
        self,
        obs,
    ):

        v = round(
            self._value(obs)
        )

        # ----------------------------------------------------
        # Mixed strategy.
        # ----------------------------------------------------

        if obs.round > 1:

            v += self.rng.choice(
                [
                    -1,
                    0,
                    0,
                    1,
                ]
            )

        # ----------------------------------------------------
        # Early-round anti-sniping spread.
        # ----------------------------------------------------

        if obs.round <= 2:

            width = min(
                obs.spread_cap,
                obs.final_cap + 2,
            )

        else:

            width = (
                obs.final_cap
            )

        lo = (
            v
            - width // 2
        )

        return (
            lo,
            lo + width,
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    def respond(
        self,
        obs,
        quote,
        turn,
    ):

        bid, ask = quote

        v = self._value(
            obs,
            quote,
        )

        # ====================================================
        # FINAL TURN
        # ====================================================

        if (
            turn
            == self.config.N_TURNS
        ):

            my_shift = sum(
                int(
                    self.config.POWERS[pw][
                        "magnitude"
                    ]
                )
                for pw in (
                    "TRICK_ROOM",
                    "STEALTH_ROCK",
                )
                if (
                    pw in obs.powers_mine
                    and
                    pw in self.config.POWERS
                )
            )

            opp_shift = sum(
                int(
                    self.config.POWERS[pw][
                        "magnitude"
                    ]
                )
                for pw in (
                    "TRICK_ROOM",
                    "STEALTH_ROCK",
                )
                if (
                    pw in obs.powers_theirs
                    and
                    pw in self.config.POWERS
                )
            )

            net_shift = (
                my_shift
                - opp_shift
            )

            old_width = (
                ask - bid
            )

            max_width = min(
                old_width,
                max(
                    self.config.final_cap(
                        obs.round
                    ),
                    old_width
                    - self.config.MIN_REDUCTION,
                ),
            )

            new_ask = ask

            new_bid = (
                new_ask
                - max_width
            )

            forced_price = (
                (
                    new_bid
                    + new_ask
                )
                // 2
            ) + net_shift

            buy_ev = (
                v - ask
            )

            sell_ev = (
                bid - v
            )

            force_ev = (
                forced_price
                - v
                - self.config.FORCED_FILL_FEE
            )

            best = max(
                buy_ev,
                sell_ev,
                force_ev,
            )

            if best == buy_ev:
                return "ACCEPT_BUY"

            if best == sell_ev:
                return "ACCEPT_SELL"

            return (
                "COUNTER",
                new_bid,
                new_ask,
            )

        # ====================================================
        # TURNS 2-5
        # ====================================================

        edge_buy = (
            v - ask
        )

        edge_sell = (
            bid - v
        )

        # ----------------------------------------------------
        # Uncertainty margin.
        # ----------------------------------------------------

        unknown = (
            self.config.unknown_to_both(
                obs.round
            )
        )

        edge_margin = (
            0.25
            * math.sqrt(
                unknown
            )
            if unknown > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Substitute lowers downside.
        # ----------------------------------------------------

        if (
            "SUBSTITUTE"
            in obs.powers_mine
        ):

            edge_margin = max(
                0.0,
                edge_margin - 1.0,
            )

        # ----------------------------------------------------
        # Power-aware stalling.
        # ----------------------------------------------------

        my_shift = sum(
            int(
                self.config.POWERS[pw][
                    "magnitude"
                ]
            )
            for pw in (
                "TRICK_ROOM",
                "STEALTH_ROCK",
            )
            if (
                pw in obs.powers_mine
                and
                pw in self.config.POWERS
            )
        )

        if my_shift > 0:

            edge_margin += (
                my_shift
                * 0.75
            )

        # ----------------------------------------------------
        # Acceptance.
        # ----------------------------------------------------

        if (
            edge_buy
            >= edge_margin
            and
            edge_buy
            >= edge_sell
        ):

            return "ACCEPT_BUY"

        if (
            edge_sell
            >= edge_margin
        ):

            return "ACCEPT_SELL"

        # ====================================================
        # COUNTER
        # ====================================================

        w = min(
            ask - bid,
            max(
                self.config.final_cap(
                    obs.round
                ),
                (
                    ask - bid
                )
                - self.config.MIN_REDUCTION,
            ),
        )

        # Mixed strategy.
        blur = self.rng.choice(
            [
                -1,
                0,
                0,
                1,
            ]
        )

        center = max(
            bid,
            min(
                round(v) + blur,
                ask - w,
            ),
        )

        return (
            "COUNTER",
            center,
            center + w,
        )


    # ========================================================
    # TRANSFORM
    # ========================================================

    def use_transform(
        self,
        obs,
    ):

        opp_k = (
            self._historical_opponent_estimate(
                obs
            )
        )

        # If opponent is substantially more decisive,
        # Transform can neutralize that advantage.

        if (
            abs(opp_k)
            > abs(obs.k_mine)
        ):

            return True

        return (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        )