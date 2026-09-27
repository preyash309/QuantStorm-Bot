"""
adaptive_exact.py

AdaptiveExact

Next generation of AdaptivePlus.

Changes:

    1. Exact posterior score distribution.
    2. Exact final-turn EV optimizer.
    3. Exact treatment of SUBSTITUTE.
    4. Exact treatment of TRICK_ROOM / STEALTH_ROCK
       on forced fills.
    5. Existing calibrated auction strategy retained.
    6. Existing TRANSFORM strategy retained.
    7. Existing quote strategy retained.

Only the FINAL negotiation turn is changed.

Earlier turns retain the proven Adaptive policy because an
earlier counter introduces a strategic response from the
opponent and therefore cannot be solved exactly without an
opponent policy model.
"""

from __future__ import annotations

import random

from exact_value import (
    known_sum_from_obs,
    unknown_count_from_obs,
)

from exact_negotiation import (
    best_final_action,
    best_forcing_counter,
)

from strategies.adaptive_bidder import (
    POWER_VALUES,
    SHADE,
    FLAT_THRESHOLD,
    OPP_FLAT_THRESHOLD,
    DENIAL_WEIGHT,
)


class Bot:

    name = "AdaptiveExact"

    # ========================================================
    # Reset
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
    # Opponent quote read
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

        earlier = [
            r
            for r in self._opp_anchor
            if r < obs.round
        ]

        if not earlier:
            return None

        return self._opp_anchor[
            max(earlier)
        ]

    # ========================================================
    # Exact posterior
    # ========================================================

    def _known_sum(
        self,
        obs,
    ):
        return known_sum_from_obs(
            obs
        )

    def _unknown_count(
        self,
        obs,
    ):
        return unknown_count_from_obs(
            obs,
            self.config,
        )

    def _exact_value(
        self,
        obs,
    ):
        """
        Exact E[S | direct information].

        Unknown fair coins have expectation zero.
        """

        return float(
            self._known_sum(obs)
        )

    # ========================================================
    # Taker valuation
    # ========================================================

    def _value(
        self,
        obs,
        quote=None,
    ):

        exact_value = (
            self._exact_value(obs)
        )

        if (
            obs.is_maker
            or quote is None
        ):
            return exact_value

        anchor = self._get_anchor(
            obs,
            quote,
        )

        if obs.foresight:

            anchor = (
                0.5 * anchor
                + 0.5 * sum(
                    obs.foresight
                )
            )

        return float(
            obs.k_mine
            + anchor
        )

    # ========================================================
    # Power valuation
    # ========================================================

    def _power_value(
        self,
        obs,
        name,
    ):

        return POWER_VALUES.get(
            name,
            {},
        ).get(
            obs.round,
            0.5,
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

        if (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        ):
            return swap

        opp_k = self._opponent_k(
            obs
        )

        if (
            opp_k is not None
            and abs(opp_k)
            <= OPP_FLAT_THRESHOLD
        ):

            return (
                swap
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

        if (
            not offered
            or obs.te_mine <= 0
        ):
            return {}

        result = {}

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

        value = round(
            self._exact_value(obs)
        )

        width = obs.final_cap

        low = (
            value
            - width // 2
        )

        return (
            low,
            low + width,
        )

    # ========================================================
    # Earlier negotiation
    # ========================================================

    def _normal_response(
        self,
        obs,
        quote,
    ):

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

        threshold = 0.0

        if (
            "SUBSTITUTE"
            in obs.powers_mine
        ):
            threshold = -1.0

        if (
            edge_buy > threshold
            and edge_buy >= edge_sell
        ):

            return "ACCEPT_BUY"

        if edge_sell > threshold:

            return "ACCEPT_SELL"

        width = max(
            0,
            (
                ask
                - bid
            )
            - self.config.MIN_REDUCTION,
        )

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
    # Response
    # ========================================================

    def respond(
        self,
        obs,
        quote,
        turn,
    ):

        # ----------------------------------------------------
        # EXACT FINAL-TURN OPTIMIZER
        # ----------------------------------------------------

        if turn == self.config.N_TURNS:

            known_sum = (
                self._known_sum(obs)
            )

            unknown_count = (
                self._unknown_count(obs)
            )

            best = best_final_action(
                bid=quote[0],
                ask=quote[1],
                final_cap=obs.final_cap,
                known_sum=known_sum,
                unknown_count=unknown_count,
                my_powers=obs.powers_mine,
                opponent_powers=obs.powers_theirs,
                config=self.config,
            )

            # ------------------------------------------------
            # Acceptance actions are directly executable.
            # ------------------------------------------------

            if (
                best.action
                == "ACCEPT_BUY"
            ):
                return "ACCEPT_BUY"

            if (
                best.action
                == "ACCEPT_SELL"
            ):
                return "ACCEPT_SELL"

            # ------------------------------------------------
            # FORCE
            #
            # Construct the exact legal counter used in the
            # EV calculation.
            # ------------------------------------------------

            new_bid, new_ask = (
                best_forcing_counter(
                    bid=quote[0],
                    ask=quote[1],
                    final_cap=obs.final_cap,
                )
            )

            return (
                "COUNTER",
                new_bid,
                new_ask,
            )

        # ----------------------------------------------------
        # Turns before the final one:
        #
        # retain the established Adaptive policy.
        # ----------------------------------------------------

        return self._normal_response(
            obs,
            quote,
        )

    # ========================================================
    # TRANSFORM execution
    # ========================================================

    def use_transform(
        self,
        obs,
    ):

        return (
            abs(obs.k_mine)
            <= FLAT_THRESHOLD
        )