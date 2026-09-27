import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from game_config import DEFAULT_CONFIG
from research.simulator.mechanics import sanitize_counter, fill_shift

def main():
    c = DEFAULT_CONFIG
    nb, na = sanitize_counter(
        0, 8, -100, 100,
        c.final_cap(1), c.MIN_REDUCTION
    )
    assert 0 <= nb <= na <= 8
    assert na-nb <= max(c.final_cap(1), 8-c.MIN_REDUCTION)

    shift = fill_shift(
        0,
        ({"TRICK_ROOM"}, {"STEALTH_ROCK"}),
        c
    )
    assert shift == 1

    print("="*60)
    print("PHASE 2 MECHANICS TEST PASSED")
    print("="*60)

if __name__ == "__main__":
    main()
