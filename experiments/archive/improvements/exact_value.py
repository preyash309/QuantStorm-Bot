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

from __future__ import annotations

from functools import lru_cache
from math import comb


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


if __name__ == "__main__":
    print_posterior(
        known_sum=4,
        unknown_count=4,
    )