# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal v9 — TEST3 + uncertainty-aware tail-risk control.

BASE:
    AdaptiveFinal v3 / test3

UNCHANGED:
    - Historical quote carryover
    - FORESIGHT residual estimation
    - TRANSFORM valuation
    - TE pacing
    - Auction lockout
    - Quote generation
    - Turn-6 evaluation
    - Counter generation

NEW:
    A small uncertainty premium is added to the acceptance
    margin when our estimate of the opponent's hidden value
    is poorly informed.

    The goal is NOT to change the valuation itself.

    Instead:

        point estimate says:
            "this deal is slightly profitable"

        uncertainty says:
            "but the estimate could be wrong"

    Therefore require a little more edge before accepting.

    If uncertainty is low, TEST3 behaves identically.
"""

import random
import math


# ============================================================
# Calibrated Strategy Parameters
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


SHADE = 0.60
DENIAL_WEIGHT = 0.40
FLAT_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0


# ============================================================
# TEST9 PARAMETERS
# ============================================================

# Below this estimated uncertainty, behave EXACTLY like test3.
UNCERTAINTY_THRESHOLD = 2.0

# How strongly uncertainty increases the acceptance margin.
UNCERTAINTY_WEIGHT = 0.12

# Hard cap on the additional safety margin.
MAX_UNCERTAINTY_PREMIUM = 0.60


# ============================================================
# BOT
# ============================================================

class Bot:

    name = "AdaptiveFinal_v9"


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
                and
                quote is not None
            ):

                raw_midpoint = (
                    quote[0]
                    +
                    quote[1]
                ) / 2.0

                # Trust late-round quotes more heavily.
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

                # Clamp implied coins to possible bounds.
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
    # LATEST OPPONENT K
    # ========================================================

    def _latest_opp_k(
        self,
        obs,
    ):

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
        # Base opponent estimate.
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

            anchor = self._latest_opp_k(
                obs
            )

        # ----------------------------------------------------
        # Exact FORESIGHT residual calculation.
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
                    -
                    (
                        n_seen
                        /
                        n_total
                    ),
                )
                if n_total > 0
                else 0.0
            )

            opp_coins_estimate = (
                sum(obs.foresight)
                +
                (
                    unseen_fraction
                    *
                    anchor
                )
            )

        else:

            opp_coins_estimate = anchor

        return float(
            obs.k_mine
            +
            opp_coins_estimate
        )


    # ========================================================
    # TEST9 — ESTIMATE UNCERTAINTY
    # ========================================================

    def _value_uncertainty(
        self,
        obs,
        quote,
    ):
        """
        Estimate how uncertain our scalar value estimate is.

        This is intentionally NOT a statistical model.

        It uses only quantities already available to test3:

            - round
            - FORESIGHT observations
            - opponent quote
            - historical anchor

        The uncertainty is used only as a small risk premium.
        It never changes the value estimate itself.
        """

        # ----------------------------------------------------
        # Current opponent anchor.
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

            anchor = self._latest_opp_k(
                obs
            )

        # ----------------------------------------------------
        # How much information has FORESIGHT revealed?
        # ----------------------------------------------------

        n_total = max(
            1,
            4 * obs.round,
        )

        n_seen = len(
            obs.foresight
        )

        unseen_fraction = max(
            0.0,
            1.0
            -
            (
                n_seen
                /
                n_total
            ),
        )

        # ----------------------------------------------------
        # Residual uncertainty.
        #
        # If we know almost nothing, the hidden component
        # can still move our estimate substantially.
        #
        # If FORESIGHT has revealed much of the hand, this
        # term naturally collapses.
        # ----------------------------------------------------

        hidden_uncertainty = (
            unseen_fraction
            *
            min(
                4.0,
                1.0
                +
                abs(anchor),
            )
        )

        # ----------------------------------------------------
        # Quote disagreement.
        #
        # If the current quote midpoint is far from the
        # historical anchor, the opponent's behavior is less
        # consistent with our previous estimate.
        #
        # We treat only a fraction of this as uncertainty.
        # ----------------------------------------------------

        quote_disagreement = 0.0

        if quote is not None:

            quote_mid = (
                quote[0]
                +
                quote[1]
            ) / 2.0

            quote_disagreement = min(
                3.0,
                abs(
                    quote_mid
                    -
                    anchor
                )
                *
                0.25,
            )

        # ----------------------------------------------------
        # Combined uncertainty.
        # ----------------------------------------------------

        return (
            hidden_uncertainty
            +
            quote_disagreement
        )


    # ========================================================
    # TEST9 — UNCERTAINTY PREMIUM
    # ========================================================

    def _uncertainty_premium(
        self,
        obs,
        quote,
    ):
        """
        Convert uncertainty into a small additional
        acceptance threshold.

        IMPORTANT:

        The premium is zero for low uncertainty.

        Therefore a large fraction of TEST3 decisions are
        completely untouched.
        """

        uncertainty = (
            self._value_uncertainty(
                obs,
                quote,
            )
        )

        excess = max(
            0.0,
            uncertainty
            -
            UNCERTAINTY_THRESHOLD,
        )

        premium = (
            excess
            *
            UNCERTAINTY_WEIGHT
        )

        return min(
            MAX_UNCERTAINTY_PREMIUM,
            premium,
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
            <
            abs(opp_k)
            or
            abs(obs.k_mine)
            <=
            FLAT_THRESHOLD
        ):

            return swap

        # Defensive value.
        if (
            abs(opp_k)
            <=
            OPP_FLAT_THRESHOLD
            and
            abs(obs.k_mine)
            >=
            3
        ):

            return (
                swap
                *
                DENIAL_WEIGHT
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
            -
            obs.te_theirs
        )

        dynamic_shade = min(
            0.85,
            max(
                0.35,
                SHADE
                +
                (
                    te_advantage
                    *
                    0.015
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
                /
                self.config.TE_SALVAGE
            )

            calculated_bid = int(
                fair_te
                *
                dynamic_shade
            )

            # ------------------------------------------------
            # R4/R5 FORESIGHT lockout.
            # ------------------------------------------------

            if (
                name == "FORESIGHT"
                and
                obs.round >= 4
            ):

                if (
                    obs.te_mine
                    >
                    obs.te_theirs
                ):

                    calculated_bid = min(
                        obs.te_mine,
                        obs.te_theirs + 1,
                    )

                else:

                    calculated_bid = (
                        obs.te_mine
                    )

            # ------------------------------------------------
            # General lockout.
            # ------------------------------------------------

            elif (
                calculated_bid
                >
                obs.te_theirs
                and
                obs.te_mine
                >
                obs.te_theirs
            ):

                calculated_bid = (
                    obs.te_theirs
                    +
                    1
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
        # Strict TE pacing.
        # ----------------------------------------------------

        if (
            obs.round <= 2
            and
            obs.te_mine > 8
        ):

            for name in out:

                if name != "STEALTH_ROCK":

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

        # EXACT TEST3 QUOTE LOGIC.

        v = round(
            self._value(obs)
        )

        if obs.round > 1:

            v += self.rng.choice(
                [
                    -1,
                    0,
                    0,
                    1,
                ]
            )

        if obs.round <= 2:

            width = min(
                obs.spread_cap,
                obs.final_cap + 2,
            )

        else:

            width = obs.final_cap

        lo = (
            v
            -
            width // 2
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
        # EXACT TEST3 TURN 6
        # ====================================================

        if (
            turn
            ==
            self.config.N_TURNS
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
                    pw
                    in obs.powers_mine
                    and
                    pw
                    in self.config.POWERS
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
                    pw
                    in obs.powers_theirs
                    and
                    pw
                    in self.config.POWERS
                )
            )

            net_shift = (
                my_shift
                -
                opp_shift
            )

            old_width = (
                ask
                -
                bid
            )

            max_width = min(
                old_width,
                max(
                    self.config.final_cap(
                        obs.round
                    ),
                    old_width
                    -
                    self.config.MIN_REDUCTION,
                ),
            )

            new_ask = ask

            new_bid = (
                new_ask
                -
                max_width
            )

            forced_price = (
                (
                    new_bid
                    +
                    new_ask
                )
                //
                2
            ) + net_shift

            buy_ev = (
                v
                -
                ask
            )

            sell_ev = (
                bid
                -
                v
            )

            force_ev = (
                forced_price
                -
                v
                -
                self.config.FORCED_FILL_FEE
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
            -
            ask
        )

        edge_sell = (
            bid
            -
            v
        )

        unknown = (
            self.config.unknown_to_both(
                obs.round
            )
        )

        # ----------------------------------------------------
        # EXACT TEST3 BASE MARGIN.
        # ----------------------------------------------------

        edge_margin = (
            0.25
            *
            math.sqrt(
                unknown
            )
            if unknown > 0
            else 0.0
        )

        # ----------------------------------------------------
        # TEST3 SUBSTITUTE adjustment.
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
        # TEST3 power-aware stalling.
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
                pw
                in obs.powers_mine
                and
                pw
                in self.config.POWERS
            )
        )

        if my_shift > 0:

            edge_margin += (
                my_shift
                *
                0.75
            )

        # ====================================================
        # TEST9 ONLY CHANGE
        # ====================================================

        uncertainty_premium = (
            self._uncertainty_premium(
                obs,
                quote,
            )
        )

        edge_margin += (
            uncertainty_premium
        )

        # ----------------------------------------------------
        # ACCEPT BUY
        # ----------------------------------------------------

        if (
            edge_buy
            >=
            edge_margin
            and
            edge_buy
            >=
            edge_sell
        ):

            return "ACCEPT_BUY"

        # ----------------------------------------------------
        # ACCEPT SELL
        # ----------------------------------------------------

        if (
            edge_sell
            >=
            edge_margin
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
                -
                self.config.MIN_REDUCTION,
            ),
        )

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
                +
                blur,
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

        # EXACT TEST3.

        opp_k = self._latest_opp_k(
            obs
        )

        if (
            abs(opp_k)
            >
            abs(obs.k_mine)
        ):

            return True

        return (
            abs(obs.k_mine)
            <=
            FLAT_THRESHOLD
        )