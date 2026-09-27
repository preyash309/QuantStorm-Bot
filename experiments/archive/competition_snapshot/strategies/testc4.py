# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal v7 — standalone QuantStorm submission.

BASE:
    TEST3

ONLY CHANGE:
    Adaptive quote width.

Strategy:
    - Strong + certain position -> narrower spread
    - Weak + uncertain position -> wider spread
    - Otherwise -> TEST3 width

Everything else is intentionally unchanged from TEST3.
"""

import random
import math


# ============================================================
# Calibrated Strategy Parameters
# ============================================================

POWER_VALUES = {
    "FORESIGHT":    {
        1: 0.76,
        2: 1.16,
        3: 1.48,
        4: 1.97,
        5: 2.02,
    },

    "TRICK_ROOM":   {
        1: 1.14,
        2: 0.00,
        3: 0.00,
        4: 0.60,
        5: 0.52,
    },

    "SUBSTITUTE":   {
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

    "TRANSFORM":    {
        1: 1.58,
        2: 1.24,
        3: 1.31,
        4: 0.00,
        5: 0.00,
    },
}


SHADE = 0.60
DENIAL_WEIGHT = 0.40
FLAT_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0


# ============================================================
# BOT
# ============================================================

class Bot:

    name = "AdaptiveFinal_v7"


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seat,
        config,
        seed,
    ):

        self.seat = seat
        self.config = config

        self.rng = random.Random(seed)

        self._anchor = {}
        self._opp_anchor = {}


    # ========================================================
    # OPPONENT ANCHOR
    # ========================================================

    def _get_anchor(
        self,
        obs,
        quote,
    ):

        r = obs.round

        if r not in self._anchor:

            if (
                not obs.is_maker
                and quote is not None
            ):

                raw_midpoint = (
                    quote[0]
                    + quote[1]
                ) / 2.0

                # Trust late-round quotes heavily,
                # discount early quotes.
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

                # Clamp implied coins to mathematically
                # possible bounds.
                max_revealed = (
                    4 * r
                )

                clamped = max(
                    -max_revealed,
                    min(
                        max_revealed,
                        discounted,
                    ),
                )

                self._anchor[r] = clamped
                self._opp_anchor[r] = clamped

            else:

                self._anchor[r] = 0.0

        return self._anchor[r]


    # ========================================================
    # LATEST OPPONENT ESTIMATE
    # ========================================================

    def _latest_opp_k(
        self,
        obs,
    ):
        """
        Retrieve the most recent clean read of the
        opponent's hand.
        """

        earlier = [
            k
            for k in self._opp_anchor
            if (
                k <= obs.round
                and
                self._opp_anchor[k] != 0.0
            )
        ]

        if not earlier:

            return 0.0

        return self._opp_anchor[
            max(earlier)
        ]


    # ========================================================
    # VALUE
    # ========================================================

    def _value(
        self,
        obs,
        quote=None,
    ):

        # ----------------------------------------------------
        # 1. Base opponent hand estimate from historical quotes
        # ----------------------------------------------------

        if (
            not obs.is_maker
            and
            quote is not None
        ):

            anchor = self._get_anchor(
                obs,
                quote,
            )

        else:

            # When Maker, use historical anchor.
            anchor = self._latest_opp_k(
                obs
            )

        # ----------------------------------------------------
        # 2. Exact FORESIGHT residual calculation
        # ----------------------------------------------------

        if obs.foresight:

            n_seen = len(
                obs.foresight
            )

            n_total = (
                4 * obs.round
            )

            unseen_fraction = (
                max(
                    0.0,
                    1.0
                    - (
                        n_seen
                        / n_total
                    ),
                )
                if n_total > 0
                else 0.0
            )

            # Exact sum of leaked coins +
            # Bayesian estimate of remaining unseen coins.
            opp_coins_estimate = (
                sum(obs.foresight)
                +
                (
                    unseen_fraction
                    * anchor
                )
            )

        else:

            opp_coins_estimate = anchor

        return float(
            obs.k_mine
            + opp_coins_estimate
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
                obs.round,
                0.5,
            )
        )


    # ========================================================
    # TRANSFORM VALUE
    # ========================================================

    def _transform_value(
        self,
        obs,
    ):

        swap = self._power_value(
            obs,
            "TRANSFORM",
        )

        opp_k = self._latest_opp_k(
            obs
        )

        # High value if our hand is flatter than
        # opponent's estimated hand.
        if (
            abs(obs.k_mine)
            < abs(opp_k)
            or
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        ):

            return swap

        # Defense value against opponent taking
        # our decisive hand.
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
    # BID
    # ========================================================

    def bid(
        self,
        obs,
        offered,
    ):

        if (
            not offered
            or
            obs.te_mine <= 0
        ):

            return {}

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

                v = self._transform_value(
                    obs
                )

            else:

                v = self._power_value(
                    obs,
                    name,
                )

            if v <= 0:
                continue

            fair_te = (
                v
                / self.config.TE_SALVAGE
            )

            calculated_bid = int(
                fair_te
                * dynamic_shade
            )

            # ------------------------------------------------
            # Auction Lockout Snipe
            # R4 & R5 FORESIGHT
            # ------------------------------------------------

            if (
                name == "FORESIGHT"
                and
                obs.round >= 4
            ):

                if (
                    obs.te_mine
                    > obs.te_theirs
                ):

                    # Guarantee win at minimum necessary price.
                    calculated_bid = min(
                        obs.te_mine,
                        obs.te_theirs + 1,
                    )

                else:

                    # Go all-in if behind on crucial information.
                    calculated_bid = (
                        obs.te_mine
                    )

            # ------------------------------------------------
            # General lockout snipe.
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

        # ----------------------------------------------------
        # Strict TE pacing:
        # Reserve budget for R4/R5 FORESIGHT.
        # ----------------------------------------------------

        if (
            obs.round <= 2
            and
            obs.te_mine > 8
        ):

            for name in out:

                if name != "STEALTH_ROCK":

                    # Persistent powers are worth
                    # buying early.
                    out[name] = min(
                        out[name],
                        obs.te_mine - 8,
                    )

        elif (
            obs.round == 3
            and
            obs.te_mine > 6
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
        """
        TEST7 CHANGE:
        Adaptive quote width.

        TEST3:
            R1/R2 -> min(spread_cap, final_cap + 2)
            R3+    -> final_cap

        TEST7:
            Start from the exact TEST3 width.

            Then estimate:
                1. information uncertainty
                2. strength of our current position

            Strong + certain:
                narrow by 1

            Weak + uncertain:
                widen by 1

            Otherwise:
                leave TEST3 unchanged.

        The quote center and random perturbation remain
        identical to TEST3.
        """

        # ----------------------------------------------------
        # EXACT TEST3 VALUE.
        # ----------------------------------------------------

        v = round(
            self._value(obs)
        )

        # ----------------------------------------------------
        # EXACT TEST3 MIXED STRATEGY.
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
        # EXACT TEST3 BASE WIDTH.
        # ----------------------------------------------------

        if obs.round <= 2:

            base_width = min(
                obs.spread_cap,
                obs.final_cap + 2,
            )

        else:

            base_width = (
                obs.final_cap
            )

        width = int(
            base_width
        )

        # ====================================================
        # TEST7 ADAPTIVE WIDTH
        # ====================================================

        # ----------------------------------------------------
        # Information uncertainty.
        #
        # unknown_to_both() is already used by TEST3's
        # response policy, so this is an existing game
        # quantity rather than a new model.
        # ----------------------------------------------------

        unknown = (
            self.config.unknown_to_both(
                obs.round
            )
        )

        # ----------------------------------------------------
        # Estimate our informational confidence.
        #
        # More revealed FORESIGHT coins -> more confidence.
        # ----------------------------------------------------

        if obs.foresight:

            n_seen = len(
                obs.foresight
            )

            n_total = (
                4 * obs.round
            )

            if n_total > 0:

                revealed_fraction = (
                    n_seen
                    / n_total
                )

            else:

                revealed_fraction = 0.0

        else:

            revealed_fraction = 0.0

        # ----------------------------------------------------
        # Position strength.
        #
        # We compare our estimated value to zero.
        #
        # Large absolute value means a more decisive
        # position.
        # ----------------------------------------------------

        position_strength = abs(
            float(v)
        )

        # ----------------------------------------------------
        # Normalize uncertainty.
        #
        # This is deliberately coarse.
        # We don't want another fitted model.
        # ----------------------------------------------------

        uncertainty_high = (
            unknown >= 8
        )

        uncertainty_low = (
            unknown <= 3
        )

        # ----------------------------------------------------
        # Information confidence.
        #
        # Either:
        #   - little unknown information
        #   - substantial FORESIGHT information
        #
        # counts as relatively high confidence.
        # ----------------------------------------------------

        confidence_high = (
            uncertainty_low
            or
            revealed_fraction >= 0.50
        )

        confidence_low = (
            uncertainty_high
            and
            revealed_fraction < 0.50
        )

        # ----------------------------------------------------
        # Strong position:
        #
        # Narrow the quote.
        #
        # We require both strength AND confidence so that
        # random early estimates don't cause aggressive
        # quote compression.
        # ----------------------------------------------------

        if (
            position_strength >= 3
            and
            confidence_high
        ):

            width -= 1

        # ----------------------------------------------------
        # Weak / uncertain position:
        #
        # Widen the quote.
        #
        # This gives the opponent more room to find a
        # mutually acceptable deal.
        # ----------------------------------------------------

        elif (
            position_strength <= 1
            and
            confidence_low
        ):

            width += 1

        # ----------------------------------------------------
        # LEGALITY CLAMP
        #
        # Never exceed spread_cap.
        # Never go below the minimum meaningful spread.
        # ----------------------------------------------------

        width = max(
            0,
            min(
                width,
                obs.spread_cap,
            ),
        )

        # ----------------------------------------------------
        # Keep the quote centered exactly as TEST3.
        # ----------------------------------------------------

        lo = (
            v
            - width // 2
        )

        return (
            lo,
            lo + width,
        )


    # ========================================================
    # RESPOND
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
        # EXACT TURN 6 EVALUATION
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
                ask
                - bid
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
                v
                - ask
            )

            sell_ev = (
                bid
                - v
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
        # TURNS 2 TO 5
        # ====================================================

        edge_buy = (
            v
            - ask
        )

        edge_sell = (
            bid
            - v
        )

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
        # SUBSTITUTE.
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
        # Accept BUY.
        # ----------------------------------------------------

        if (
            edge_buy
            >= edge_margin
            and
            edge_buy
            >= edge_sell
        ):

            return "ACCEPT_BUY"

        # ----------------------------------------------------
        # Accept SELL.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TEST3 blurred center.
        # ----------------------------------------------------

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
                round(v)
                + blur,
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

        # TEST3 EXACTLY.

        opp_k = self._latest_opp_k(
            obs
        )

        # Fire swap if opponent's hand is more
        # decisive than ours.
        if (
            abs(opp_k)
            > abs(obs.k_mine)
        ):

            return True

        return (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        )