"""
test_exact_value.py

Deterministic tests for exact_value.py.
"""

from exact_value import (
    rademacher_pmf,
    score_pmf,
    expected_score,
    score_variance,
    score_std,
    distribution_mean,
    distribution_variance,
    probability_between,
    probability_at_least,
    probability_at_most,
    buy_ev,
    sell_ev,
    substitute_buy_ev,
    substitute_value,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-12,
) -> None:

    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Expected {expected}, got {actual}"
        )


def test_rademacher_distribution():

    # --------------------------------------------------------
    # Zero unknown coins
    # --------------------------------------------------------

    assert rademacher_pmf(0) == (
        (0, 1.0),
    )

    # --------------------------------------------------------
    # One unknown coin
    # --------------------------------------------------------

    assert rademacher_pmf(1) == (
        (-1, 0.5),
        (1, 0.5),
    )

    # --------------------------------------------------------
    # Two unknown coins
    # --------------------------------------------------------

    pmf = dict(
        rademacher_pmf(2)
    )

    assert_close(
        pmf[-2],
        0.25,
    )

    assert_close(
        pmf[0],
        0.50,
    )

    assert_close(
        pmf[2],
        0.25,
    )

    assert_close(
        sum(pmf.values()),
        1.0,
    )


def test_score_distribution():

    pmf = score_pmf(
        known_sum=4,
        unknown_count=4,
    )

    assert_close(
        sum(
            probability
            for _, probability in pmf
        ),
        1.0,
    )

    assert_close(
        distribution_mean(pmf),
        4.0,
    )

    assert_close(
        distribution_variance(pmf),
        4.0,
    )


def test_moments():

    assert_close(
        expected_score(
            known_sum=5,
            unknown_count=20,
        ),
        5.0,
    )

    assert_close(
        score_variance(20),
        20.0,
    )

    assert_close(
        score_std(20),
        20.0 ** 0.5,
    )


def test_interval_probability():

    probability = probability_between(
        low=3,
        high=7,
        known_sum=5,
        unknown_count=4,
    )

    print(
        "P(3 <= S <= 7) =",
        probability,
    )

    assert_close(
        probability,
        0.875,
    )


def test_tail_probabilities():

    # known_sum=0, n=2:
    #
    # S ∈ {-2, 0, 2}
    # P(S >= 0) = 0.75
    # P(S <= 0) = 0.75

    assert_close(
        probability_at_least(
            threshold=0,
            known_sum=0,
            unknown_count=2,
        ),
        0.75,
    )

    assert_close(
        probability_at_most(
            threshold=0,
            known_sum=0,
            unknown_count=2,
        ),
        0.75,
    )


def test_contract_ev():

    # E[S] = 5

    assert_close(
        buy_ev(
            price=5,
            known_sum=5,
            unknown_count=10,
        ),
        0.0,
    )

    assert_close(
        sell_ev(
            price=5,
            known_sum=5,
            unknown_count=10,
        ),
        0.0,
    )

    assert_close(
        buy_ev(
            price=3,
            known_sum=5,
            unknown_count=10,
        ),
        2.0,
    )

    assert_close(
        sell_ev(
            price=7,
            known_sum=5,
            unknown_count=10,
        ),
        2.0,
    )


def test_substitute():

    normal_ev = buy_ev(
        price=5,
        known_sum=0,
        unknown_count=4,
    )

    protected_ev = substitute_buy_ev(
        price=5,
        known_sum=0,
        unknown_count=4,
        loss_cap=2,
    )

    value = substitute_value(
        price=5,
        known_sum=0,
        unknown_count=4,
        loss_cap=2,
    )

    print(
        "SUBSTITUTE"
    )

    print(
        "normal EV =",
        normal_ev,
    )

    print(
        "protected EV =",
        protected_ev,
    )

    print(
        "power value =",
        value,
    )

    assert_close(
        normal_ev,
        -5.0,
    )

    assert_close(
        protected_ev,
        -1.9375,
    )

    assert_close(
        value,
        3.0625,
    )


def main():

    test_rademacher_distribution()
    test_score_distribution()
    test_moments()
    test_interval_probability()
    test_tail_probabilities()
    test_contract_ev()
    test_substitute()

    print()
    print("=" * 60)
    print("ALL EXACT-VALUE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()