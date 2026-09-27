# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal v8 — TEST3 + cautious online opponent exploitation.

BASE:
    AdaptiveFinal v3 / test3

UNCHANGED:
    - historical quote carryover
    - FORESIGHT residual estimation
    - TRANSFORM valuation
    - TE bidding / pacing
    - R4/R5 FORESIGHT lockout
    - quote generation
    - Turn-6 deterministic evaluation
    - counter generation

NEW:
    Online opponent-response profiling within the current deal.

    We observe:
        1. how often the opponent counters our offers
        2. whether their counter moves toward our previous offer
        3. whether they repeatedly concede

    We then make a SMALL adjustment to the normal
    acceptance margin.

    The default behavior is exactly TEST3 until
    enough evidence exists.

IMPORTANT:
    This is deliberately not a learned model.
    It is a small poker-style exploitation layer.
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
# TEST8 EXPLOITATION PARAMETERS
# ============================================================

# Minimum number of observed opponent responses before
# changing TEST3's acceptance behavior.
MIN_OBSERVATIONS = 2


# If opponent repeatedly counters and does not concede,
# we become somewhat more willing to accept.
STUBBORN_RELIEF = 0.35


# If opponent repeatedly concedes toward our offers,
# we become somewhat more demanding.
CONCESSION_PENALTY = 0.25


# Maximum total adaptation.
#
# This is deliberately small so TEST8 cannot destroy TEST3
# because of one noisy observation.
MAX_ADAPTATION = 0.50


# ============================================================
# BOT
# ============================================================

class Bot:

    name = "AdaptiveFinal_v8"


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

        # ----------------------------------------------------
        # TEST8 opponent-response statistics.
        # ----------------------------------------------------

        self._opp_responses = 0

        self._opp_counters = 0

        self._opp_concessions = 0

        self._opp_nonconcessions = 0

        # ----------------------------------------------------
        # Our previous counter.
        #
        # If a later respond() call occurs, we know the
        # opponent did not accept our previous counter.
        # ----------------------------------------------------

        self._last_my_counter = None

        # ----------------------------------------------------
        # Previous opponent quote.
        # ----------------------------------------------------

        self._last_opp_quote = None


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
                    + quote[1]
                ) / 2.0

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
        # Exact FORESIGHT residual.
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
            +
            opp_coins_estimate
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

        if (
            abs(obs.k_mine)
            < abs(opp_k)
            or
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        ):

            return swap

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
    # TEST8 — OBSERVE OPPONENT RESPONSE
    # ========================================================

    def _observe_opponent(
        self,
        quote,
    ):
        """
        Called at the beginning of respond().

        If we made a counter on the previous turn and
        we are now being asked to respond again, then:

            previous counter
                    ↓
            opponent produced another quote

        Therefore the opponent countered us.

        We compare their new quote to our previous counter
        to determine whether they moved toward us.
        """

        if (
            self._last_my_counter
            is None
        ):

            self._last_opp_quote = (
                quote
            )

            return

        # ----------------------------------------------------
        # A new response after our previous counter means
        # the opponent rejected our previous counter and
        # supplied another quote.
        # ----------------------------------------------------

        self._opp_responses += 1

        self._opp_counters += 1

        previous_bid, previous_ask = (
            self._last_my_counter
        )

        current_bid, current_ask = (
            quote
        )

        previous_mid = (
            previous_bid
            + previous_ask
        ) / 2.0

        current_mid = (
            current_bid
            + current_ask
        ) / 2.0

        # ----------------------------------------------------
        # If we have an earlier opponent quote, measure
        # whether they moved toward our previous counter.
        #
        # Otherwise use a conservative comparison.
        # ----------------------------------------------------

        if (
            self._last_opp_quote
            is not None
        ):

            old_opp_bid, old_opp_ask = (
                self._last_opp_quote
            )

            old_opp_mid = (
                old_opp_bid
                + old_opp_ask
            ) / 2.0

            old_distance = abs(
                old_opp_mid
                - previous_mid
            )

            new_distance = abs(
                current_mid
                - previous_mid
            )

            if (
                new_distance
                < old_distance
            ):

                self._opp_concessions += 1

            else:

                self._opp_nonconcessions += 1

        # ----------------------------------------------------
        # Current quote becomes the previous opponent quote
        # for the next observation.
        # ----------------------------------------------------

        self._last_opp_quote = (
            quote
        )


    # ========================================================
    # TEST8 — EXPLOITATION SCORE
    # ========================================================

    def _exploitation_adjustment(
        self,
    ):
        """
        Return a small additive adjustment to TEST3's
        acceptance margin.

        Positive:
            demand MORE edge.

        Negative:
            accept with LESS edge.

        Before enough evidence:
            exactly 0.
        """

        if (
            self._opp_responses
            < MIN_OBSERVATIONS
        ):

            return 0.0

        total = (
            self._opp_concessions
            +
            self._opp_nonconcessions
        )

        if total <= 0:

            return 0.0

        concession_rate = (
            self._opp_concessions
            / total
        )

        # ----------------------------------------------------
        # Strongly concessive opponent.
        #
        # They are moving toward us repeatedly.
        #
        # Become more demanding.
        # ----------------------------------------------------

        if concession_rate >= 0.67:

            return min(
                MAX_ADAPTATION,
                CONCESSION_PENALTY,
            )

        # ----------------------------------------------------
        # Strongly stubborn opponent.
        #
        # They counter without moving toward us.
        #
        # Accept a little more readily rather than
        # repeatedly burning negotiation turns.
        # ----------------------------------------------------

        if concession_rate <= 0.25:

            return max(
                -MAX_ADAPTATION,
                -STUBBORN_RELIEF,
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
                /
                self.config.TE_SALVAGE
            )

            calculated_bid = int(
                fair_te
                * dynamic_shade
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

        # ----------------------------------------------------
        # TEST8 observation happens before we make the next
        # decision.
        # ----------------------------------------------------

        self._observe_opponent(
            quote
        )

        bid, ask = quote

        v = self._value(
            obs,
            quote,
        )

        # ====================================================
        # TURN 6
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
                // 2
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

            # No adaptation to Turn 6.
            # Keep TEST3 deterministic behavior.

            self._last_my_counter = (
                new_bid,
                new_ask,
            )

            return (
                "COUNTER",
                new_bid,
                new_ask,
            )

        # ====================================================
        # TURNS 2–5
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
        # SUBSTITUTE exactly as TEST3.
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
        # Power-aware stalling exactly as TEST3.
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
        # TEST8 ONLY CHANGE
        # ====================================================

        edge_margin += (
            self._exploitation_adjustment()
        )

        # Never allow the exploitation layer to make the
        # margin negative.
        edge_margin = max(
            0.0,
            edge_margin,
        )

        # ----------------------------------------------------
        # ACCEPT BUY
        # ----------------------------------------------------

        if (
            edge_buy
            >= edge_margin
            and
            edge_buy
            >= edge_sell
        ):

            self._last_my_counter = None

            return "ACCEPT_BUY"

        # ----------------------------------------------------
        # ACCEPT SELL
        # ----------------------------------------------------

        if (
            edge_sell
            >= edge_margin
        ):

            self._last_my_counter = None

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
                + blur,
                ask - w,
            ),
        )

        new_counter = (
            center,
            center + w,
        )

        # Remember our offer so that if the opponent
        # counters next turn, we can classify the response.

        self._last_my_counter = (
            new_counter
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
            <= FLAT_THRESHOLD
        )