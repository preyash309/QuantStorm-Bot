# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal — standalone QuantStorm submission.

Flattened from the validated research stack. Frozen parameters:
    SHADE = 0.60
    TRANSFORM_SHADE = 0.60
    DENIAL_WEIGHT = 1.10

No project-local imports are required.
"""

import random
from functools import lru_cache
from math import comb


# ============================================================
# Frozen final strategy parameters
# ============================================================

POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

SHADE = 0.60
TRANSFORM_SHADE = 0.60
DENIAL_WEIGHT = 1.10
FLAT_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0


# ============================================================
# Exact posterior mathematics
# ============================================================

"""
exact_value.py

Exact discrete mathematics for the Divided Oracle game.

The hidden coins are independent fair Rademacher variables:

    X_i in {-1, +1}
    P(X_i = +1) = P(X_i = -1) = 1/2

Therefore the sum of n unknown coins has the exact lattice distribution

    X = 2K - n
    K ~ Binomial(n, 1/2)

This module contains only deterministic mathematical calculations.

No simulation.
No random numbers.
No model fitting.
"""




# ============================================================
# Exact Rademacher distribution
# ============================================================

@lru_cache(maxsize=None)
def rademacher_pmf(
    n: int,
) -> tuple[tuple[int, float], ...]:
    """
    Exact probability mass function of the sum of n fair
    {-1, +1} variables.

    Example:

        n = 2

        possible sums:
            -2, 0, +2

        probabilities:
            1/4, 1/2, 1/4

    Returns:
        tuple of (sum_value, probability)
    """

    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        return ((0, 1.0),)

    denominator = 1 << n

    result = []

    for k in range(n + 1):

        # k positive coins, n-k negative coins
        #
        # Sum = k - (n-k)
        #     = 2k - n
        score = 2 * k - n

        probability = comb(n, k) / denominator

        result.append(
            (score, probability)
        )

    return tuple(result)


# ============================================================
# Shifted score distribution
# ============================================================

@lru_cache(maxsize=None)
def score_pmf(
    known_sum: int,
    unknown_count: int,
) -> tuple[tuple[int, float], ...]:
    """
    Exact distribution of:

        S = known_sum + X

    where X is the sum of `unknown_count` independent fair
    +/-1 coins.
    """

    if unknown_count < 0:
        raise ValueError(
            "unknown_count must be >= 0"
        )

    return tuple(
        (
            known_sum + residual,
            probability,
        )
        for residual, probability
        in rademacher_pmf(unknown_count)
    )


# ============================================================
# Exact moments
# ============================================================

def expected_score(
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Exact E[S].

    Since each unknown fair coin has expectation 0:

        E[S] = known_sum
    """

    if unknown_count < 0:
        raise ValueError(
            "unknown_count must be >= 0"
        )

    return float(known_sum)


def score_variance(
    unknown_count: int,
) -> float:
    """
    Exact Var[S].

    Every independent +/-1 coin has variance 1, therefore:

        Var[S] = unknown_count
    """

    if unknown_count < 0:
        raise ValueError(
            "unknown_count must be >= 0"
        )

    return float(unknown_count)


def score_std(
    unknown_count: int,
) -> float:
    """
    Exact standard deviation of the hidden-score posterior.
    """

    return score_variance(
        unknown_count
    ) ** 0.5


# ============================================================
# Distribution statistics
# ============================================================

def distribution_mean(
    pmf: tuple[tuple[int, float], ...],
) -> float:
    """
    Calculate the expectation directly from a PMF.

    This is primarily useful for testing the mathematics.
    """

    return sum(
        score * probability
        for score, probability in pmf
    )


def distribution_variance(
    pmf: tuple[tuple[int, float], ...],
) -> float:
    """
    Calculate variance directly from a PMF.
    """

    mean = distribution_mean(pmf)

    return sum(
        probability * (score - mean) ** 2
        for score, probability in pmf
    )


# ============================================================
# Interval probability
# ============================================================

def probability_between(
    low: int,
    high: int,
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Exact probability that:

        low <= S <= high
    """

    if low > high:
        return 0.0

    pmf = score_pmf(
        known_sum,
        unknown_count,
    )

    return sum(
        probability
        for score, probability in pmf
        if low <= score <= high
    )


# ============================================================
# Tail probabilities
# ============================================================

def probability_at_least(
    threshold: int,
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Exact P(S >= threshold).
    """

    return sum(
        probability
        for score, probability
        in score_pmf(
            known_sum,
            unknown_count,
        )
        if score >= threshold
    )


def probability_at_most(
    threshold: int,
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Exact P(S <= threshold).
    """

    return sum(
        probability
        for score, probability
        in score_pmf(
            known_sum,
            unknown_count,
        )
        if score <= threshold
    )


# ============================================================
# Contract EV
# ============================================================

def buy_ev(
    price: int,
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Expected PnL of being LONG at price.

        PnL = S - price

    Therefore:

        EV = E[S] - price
    """

    return (
        expected_score(
            known_sum,
            unknown_count,
        )
        - price
    )


def sell_ev(
    price: int,
    known_sum: int,
    unknown_count: int,
) -> float:
    """
    Expected PnL of being SHORT at price.

        PnL = price - S

    Therefore:

        EV = price - E[S]
    """

    return (
        price
        - expected_score(
            known_sum,
            unknown_count,
        )
    )


# ============================================================
# Exact SUBSTITUTE mathematics
# ============================================================

def substitute_buy_ev(
    price: int,
    known_sum: int,
    unknown_count: int,
    loss_cap: int = 2,
) -> float:
    """
    Expected PnL of a long position protected by SUBSTITUTE.

    Without SUBSTITUTE:

        payoff = S - price

    With SUBSTITUTE:

        payoff = max(S - price, -loss_cap)

    NOTE:
    This is the mathematical payoff calculation only.

    It does NOT attempt to decide whether buying SUBSTITUTE is
    optimal in the auction. Auction opportunity cost and
    remaining TE must be handled separately.
    """

    if loss_cap < 0:
        raise ValueError(
            "loss_cap must be >= 0"
        )

    return sum(
        max(
            score - price,
            -loss_cap,
        ) * probability
        for score, probability
        in score_pmf(
            known_sum,
            unknown_count,
        )
    )


def substitute_value(
    price: int,
    known_sum: int,
    unknown_count: int,
    loss_cap: int = 2,
) -> float:
    """
    Incremental mathematical value of SUBSTITUTE relative to
    an unprotected long position at the same price.
    """

    ordinary = buy_ev(
        price=price,
        known_sum=known_sum,
        unknown_count=unknown_count,
    )

    protected = substitute_buy_ev(
        price=price,
        known_sum=known_sum,
        unknown_count=unknown_count,
        loss_cap=loss_cap,
    )

    return protected - ordinary


# ============================================================
# Information-state helpers
# ============================================================

def known_sum_from_obs(obs) -> int:
    """
    Sum of all opponent/own coin information legally visible
    to the bot that contributes directly to the score estimate.

    Current information:

        own revealed coins
        +
        FORESIGHT-revealed opponent coins
    """

    return (
        obs.k_mine
        + sum(obs.foresight)
    )


def known_count_from_obs(obs) -> int:
    """
    Number of individually known coins contributing to the
    current score posterior.
    """

    return (
        len(obs.my_revealed)
        + len(obs.foresight)
    )


def unknown_count_from_obs(
    obs,
    config,
) -> int:
    """
    Number of coins whose signs remain unknown to this bot.
    """

    unknown = (
        config.N_COINS
        - known_count_from_obs(obs)
    )

    if unknown < 0:
        raise ValueError(
            "Information state contains more known coins "
            "than N_COINS."
        )

    return unknown


def posterior_from_obs(
    obs,
    config,
) -> tuple[tuple[int, float], ...]:
    """
    Exact posterior distribution of the final score S given
    the legally observable information in Obs.
    """

    known_sum = known_sum_from_obs(obs)

    unknown_count = unknown_count_from_obs(
        obs,
        config,
    )

    return score_pmf(
        known_sum,
        unknown_count,
    )


def posterior_mean_from_obs(
    obs,
    config,
) -> float:
    """
    Exact posterior mean E[S | information in Obs].
    """

    known_sum = known_sum_from_obs(obs)

    unknown_count = unknown_count_from_obs(
        obs,
        config,
    )

    return expected_score(
        known_sum,
        unknown_count,
    )


def posterior_std_from_obs(
    obs,
    config,
) -> float:
    """
    Exact posterior standard deviation.
    """

    unknown_count = unknown_count_from_obs(
        obs,
        config,
    )

    return score_std(
        unknown_count
    )


# ============================================================
# Debug helper
# ============================================================

def print_posterior(
    known_sum: int,
    unknown_count: int,
) -> None:
    """
    Human-readable posterior dump for development/testing.
    """

    pmf = score_pmf(
        known_sum,
        unknown_count,
    )

    print(
        f"known_sum={known_sum}, "
        f"unknown_count={unknown_count}"
    )

    print()

    for score, probability in pmf:
        print(
            f"S={score:+4d}    "
            f"P={probability:.10f}"
        )

    print()
    print(
        f"Probability total: "
        f"{sum(p for _, p in pmf):.12f}"
    )

    print(
        f"Mean: "
        f"{distribution_mean(pmf):.12f}"
    )

    print(
        f"Variance: "
        f"{distribution_variance(pmf):.12f}"
    )

# ============================================================
# Exact final-turn negotiation mathematics
# ============================================================

"""
exact_negotiation.py

Exact EV calculations for the Divided Oracle negotiation.

This module is used ONLY for the final negotiation turn.

On the final turn, the three meaningful actions are:

    ACCEPT_BUY
        Long at ask.

    ACCEPT_SELL
        Short at bid.

    COUNTER
        The negotiation ends in a forced fill.
        The final quoter becomes short.
        The final quoter pays the forcing fee.

Because the score distribution is a finite +/-1 lattice,
every EV below is calculated exactly.

No Monte Carlo.
No approximation.
No learned parameters.
"""





# ============================================================
# Result
# ============================================================

class ActionEV:
    """Exact expected value of one final-turn action."""

    __slots__ = ("action", "ev", "price", "details")

    def __init__(self, action: str, ev: float, price: int, details: str = ""):
        self.action = action
        self.ev = ev
        self.price = price
        self.details = details


# ============================================================
# Settlement payoff
# ============================================================

def protected_payoff(
    raw_pnl: float,
    has_substitute: bool,
    cap: float = 2.0,
) -> float:
    """
    Apply SUBSTITUTE to one contract payoff.

    Without SUBSTITUTE:

        payoff = raw_pnl

    With SUBSTITUTE:

        payoff = max(raw_pnl, -cap)

    Profit is uncapped.
    """

    if not has_substitute:
        return raw_pnl

    return max(
        raw_pnl,
        -cap,
    )


def expected_contract_payoff(
    *,
    price: int,
    long: bool,
    known_sum: int,
    unknown_count: int,
    has_substitute: bool,
    substitute_cap: float = 2.0,
) -> float:
    """
    Exact expected settlement PnL of a contract.

    Long:

        raw = S - price

    Short:

        raw = price - S

    SUBSTITUTE is applied after S is known.
    """

    total = 0.0

    for score, probability in score_pmf(
        known_sum,
        unknown_count,
    ):

        if long:
            raw = score - price
        else:
            raw = price - score

        payoff = protected_payoff(
            raw,
            has_substitute,
            substitute_cap,
        )

        total += (
            probability
            * payoff
        )

    return total


# ============================================================
# Fill-shift calculation
# ============================================================

def shift_magnitude(
    active_powers,
    config,
) -> int:
    """
    Total fill-shift magnitude controlled by a seat.

    TRICK_ROOM:
        current-round forced fill shift.

    STEALTH_ROCK:
        persistent forced-fill shift.
    """

    total = 0

    for name in (
        "TRICK_ROOM",
        "STEALTH_ROCK",
    ):

        if name in active_powers:

            if name in config.POWERS:

                total += int(
                    config.POWERS[name][
                        "magnitude"
                    ]
                )

    return total


def forced_fill_shift(
    *,
    my_powers,
    opponent_powers,
    config,
) -> int:
    """
    The bot using this function becomes SHORT on a forced fill.

    Therefore:

        own shift       -> positive
        opponent shift  -> negative
    """

    mine = shift_magnitude(
        my_powers,
        config,
    )

    theirs = shift_magnitude(
        opponent_powers,
        config,
    )

    return mine - theirs


# ============================================================
# Legal final counter
# ============================================================

def best_forcing_counter(
    *,
    bid: int,
    ask: int,
    final_cap: int,
) -> tuple[int, int]:
    """
    Construct the final-turn counter that maximizes the forced
    midpoint price for the countering player.

    The final quoter becomes SHORT.

    Therefore, all else equal, we want the highest possible
    midpoint.

    The new range must:

        1. remain inside [bid, ask]
        2. have legal width
        3. never be narrower than final_cap
        4. shrink by at least one tick unless already at floor

    The highest possible midpoint is obtained by placing the
    legal range against the old ASK.
    """

    old_width = ask - bid

    max_width = max(
        final_cap,
        old_width - 1,
    )

    max_width = min(
        old_width,
        max_width,
    )

    new_ask = ask

    new_bid = (
        new_ask
        - max_width
    )

    return (
        new_bid,
        new_ask,
    )


# ============================================================
# Exact final-turn optimizer
# ============================================================

def final_turn_actions(
    *,
    bid: int,
    ask: int,
    final_cap: int,
    known_sum: int,
    unknown_count: int,
    my_powers,
    opponent_powers,
    config,
) -> tuple[ActionEV, ActionEV, ActionEV]:
    """
    Calculate exact EV for:

        ACCEPT_BUY
        ACCEPT_SELL
        FORCE

    FORCE means:

        COUNTER on the final turn
        -> forced midpoint
        -> this bot becomes short
        -> this bot pays forcing fee
    """

    has_substitute = (
        "SUBSTITUTE"
        in my_powers
    )

    # --------------------------------------------------------
    # ACCEPT BUY
    # --------------------------------------------------------

    buy_ev = expected_contract_payoff(
        price=ask,
        long=True,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    buy_action = ActionEV(
        action="ACCEPT_BUY",
        ev=buy_ev,
        price=ask,
        details=(
            f"long at ask={ask}"
        ),
    )

    # --------------------------------------------------------
    # ACCEPT SELL
    # --------------------------------------------------------

    sell_ev = expected_contract_payoff(
        price=bid,
        long=False,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    sell_action = ActionEV(
        action="ACCEPT_SELL",
        ev=sell_ev,
        price=bid,
        details=(
            f"short at bid={bid}"
        ),
    )

    # --------------------------------------------------------
    # FORCE
    # --------------------------------------------------------

    new_bid, new_ask = (
        best_forcing_counter(
            bid=bid,
            ask=ask,
            final_cap=final_cap,
        )
    )

    midpoint = (
        new_bid
        + new_ask
    ) // 2

    shift = forced_fill_shift(
        my_powers=my_powers,
        opponent_powers=opponent_powers,
        config=config,
    )

    forced_price = (
        midpoint
        + shift
    )

    force_ev = expected_contract_payoff(
        price=forced_price,
        long=False,
        known_sum=known_sum,
        unknown_count=unknown_count,
        has_substitute=has_substitute,
        substitute_cap=float(
            config.POWERS["SUBSTITUTE"][
                "magnitude"
            ]
        ) if "SUBSTITUTE" in config.POWERS else 2.0,
    )

    # Final-turn counter incurs the forcing fee.
    force_ev -= (
        config.FORCED_FILL_FEE
    )

    force_action = ActionEV(
        action="FORCE",
        ev=force_ev,
        price=forced_price,
        details=(
            f"counter=[{new_bid},{new_ask}], "
            f"midpoint={midpoint}, "
            f"shift={shift}, "
            f"forced_price={forced_price}, "
            f"fee={config.FORCED_FILL_FEE}"
        ),
    )

    return (
        buy_action,
        sell_action,
        force_action,
    )


def best_final_action(
    **kwargs,
) -> ActionEV:
    """
    Return the exact EV-maximizing final-turn action.
    """

    actions = final_turn_actions(
        **kwargs
    )

    return max(
        actions,
        key=lambda action: action.ev,
    )


# ============================================================
# Debugging / standalone tests
# ============================================================

# ============================================================
# Final Bot
# ============================================================

class Bot:
    # Frozen research parameters. These must be class attributes because the
    # validated bid/transform code accesses them through self.<NAME>.
    SHADE = 0.60
    TRANSFORM_SHADE = 0.60
    DENIAL_WEIGHT = 1.10


    name = "AdaptiveFinal"

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
