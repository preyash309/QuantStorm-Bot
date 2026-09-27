import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fractions import Fraction
from research.simulator.score import score_pmf_unknown, expected_score, probability_between

def main():
    assert sum(p for _, p in score_pmf_unknown(4)) == 1
    assert dict(score_pmf_unknown(4)) == {
        -4: Fraction(1,16), -2: Fraction(4,16),
        0: Fraction(6,16), 2: Fraction(4,16), 4: Fraction(1,16)
    }
    assert expected_score(7, 10) == 7
    assert probability_between(0, 4, -2, 2) == Fraction(14,16)
    print("="*60)
    print("PHASE 2 SCORE TEST PASSED")
    print("="*60)

if __name__ == "__main__":
    main()
