from __future__ import annotations
from fractions import Fraction
from math import comb
from functools import lru_cache

@lru_cache(maxsize=None)
def score_pmf_unknown(n: int):
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    den = 1 << n
    return tuple(
        (2 * plus - n, Fraction(comb(n, plus), den))
        for plus in range(n + 1)
    )

def score_pmf(known_sum: int, unknown_count: int):
    for residual, p in score_pmf_unknown(int(unknown_count)):
        yield int(known_sum) + residual, p

def expected_score(known_sum: int, unknown_count: int) -> Fraction:
    return Fraction(int(known_sum), 1)

def probability_between(known_sum, unknown_count, lo, hi):
    return sum(
        p for score, p in score_pmf(known_sum, unknown_count)
        if lo <= score <= hi
    )

def expected_payoff(
    *, price, long, known_sum, unknown_count,
    substitute=False, substitute_cap=2
):
    total = Fraction(0, 1)
    for score, p in score_pmf(known_sum, unknown_count):
        raw = score - price if long else price - score
        if substitute:
            raw = max(raw, -int(substitute_cap))
        total += p * raw
    return total
