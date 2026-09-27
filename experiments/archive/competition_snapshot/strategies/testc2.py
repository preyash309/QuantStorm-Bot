# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal v5

TEST5 CHANGE FROM TEST3:
    Only the TRANSFORM decision has been changed.

    TEST3:
        use TRANSFORM whenever
            abs(opponent_estimate) > abs(my_hand)

    TEST5:
        use TRANSFORM only when
            abs(opponent_estimate) >= abs(my_hand) + 2

    Flat-hand Transform behavior is unchanged.

Everything else is intentionally identical to TEST3.
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

    name = "AdaptiveFinal_v5"

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

        self._anchor = {}
        self._opp_anchor = {}


    # ========================================================
    # OPPONENT QUOTE ANCHOR
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

                # Trust late-round quotes more.
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

                # Mathematically possible range.
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
    # HISTORICAL OPPONENT ESTIMATE
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
        # Current quote if we are responding.
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

            # Historical opponent information
            # when we are Maker.
            anchor = self._latest_opp_k(
                obs
            )

        # ----------------------------------------------------
        # Exact FORESIGHT residual.
        # ----------------------------------------------------

        if obs.foresight:

            n_seen = len(
                obs.foresight
            )

            n_total = (
                4 * obs.round
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

        # ----------------------------------------------------
        # TEST3 BEHAVIOR:
        #
        # Transform valuable when our hand is flatter.
        #
        # Kept unchanged because this is valuation.
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

            calculated_bid = int(
                fair_te
                * dynamic_shade
            )

            # ------------------------------------------------
            # R4/R5 FORESIGHT LOCKOUT
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
        # TE PACING
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
        # NORMAL NEGOTIATION
        # ====================================================

        edge_buy = (
            v - ask
        )

        edge_sell = (
            bid - v
        )

        unknown = (
            self.config.unknown_to_both(
                obs.round
            )
        )

        edge_margin = (
            0.25
            * math.sqrt(unknown)
            if unknown > 0
            else 0.0
        )

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
        # Accept.
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

        # ----------------------------------------------------
        # Counter.
        # ----------------------------------------------------

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
    # TRANSFORM DECISION
    # ========================================================

    def use_transform(
        self,
        obs,
    ):

        opp_k = (
            self._latest_opp_k(
                obs
            )
        )

        # ====================================================
        # TEST5 CHANGE
        # ====================================================
        #
        # TEST3:
        #
        #     if abs(opp_k) > abs(my):
        #         Transform
        #
        # TEST5:
        #
        #     Require a meaningful 2-point advantage.
        #
        # This prevents spending Transform merely because
        # the opponent appears one coin more decisive.
        #
        # ====================================================

        if (
            abs(opp_k)
            >=
            abs(obs.k_mine) + 2
        ):

            return True

        # Flat hands retain the original test3 behavior.
        return (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        )