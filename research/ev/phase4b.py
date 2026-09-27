"""
PHASE 4B — Public-state opponent-conditioned EV analysis.

Fits a public-information response model from the Phase 3
dataset and demonstrates one-step negotiation EV.

IMPORTANT:
    The Phase 3 dataset does not contain the actual foresight
    coin tuple. It stores:

        foresight_sum
        foresight_n

    Therefore this research runner uses foresight_sum directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

ROOT = Path(
    __file__
).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from game_config import DEFAULT_CONFIG

from research.ev.public_opponent import (
    PublicOpponentModel,
)

from research.ev.continuation import (
    evaluate_all_actions,
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_rows(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Dataset root is not a dictionary."
        )

    if "rows" not in data:
        raise RuntimeError(
            "Could not find top-level 'rows' array."
        )

    rows = data["rows"]

    if not isinstance(rows, list):
        raise RuntimeError(
            "'rows' is not a list."
        )

    return rows


# ============================================================
# SPLIT
# ============================================================

def split_by_seed(
    rows,
    validation_fraction=0.30,
):

    seeds = sorted({
        int(row["seed"])
        for row in rows
    })

    n_validation = max(
        1,
        int(
            len(seeds)
            * validation_fraction
        ),
    )

    validation_seeds = set(
        seeds[-n_validation:]
    )

    train = []
    validation = []

    for row in rows:

        if int(row["seed"]) in validation_seeds:
            validation.append(row)
        else:
            train.append(row)

    return (
        train,
        validation,
    )


# ============================================================
# QUOTE NORMALIZATION
# ============================================================

def normalize_quote(
    quote,
):
    """
    Convert the serialized dataset quote into:

        (bid, ask)

    The Phase 3 dataset should contain a two-element quote,
    but this function validates it explicitly so that a bad
    row fails clearly rather than producing a confusing error
    later.
    """

    if quote is None:
        raise ValueError(
            "Encountered respond row with quote=None."
        )

    if not isinstance(
        quote,
        (list, tuple),
    ):
        raise ValueError(
            f"Invalid quote type: "
            f"{type(quote).__name__}"
        )

    if len(quote) != 2:
        raise ValueError(
            f"Quote must contain exactly "
            f"two values, got {quote!r}"
        )

    bid = int(
        quote[0]
    )

    ask = int(
        quote[1]
    )

    if bid > ask:
        raise ValueError(
            f"Invalid quote: "
            f"bid={bid} > ask={ask}"
        )

    return (
        bid,
        ask,
    )


# ============================================================
# BUILD REPRESENTATIVE OBS
# ============================================================

class ResearchObs:
    """
    Minimal research observation reconstructed from the
    serialized Phase 3 dataset.

    IMPORTANT:
        This uses only fields actually present in the dataset.

    The dataset schema is:

        seat
        round
        k_mine
        te_mine
        te_theirs
        spread_cap
        final_cap
        is_maker
        powers_mine
        powers_theirs
        foresight_sum
        foresight_n
        n_unknown_both
        n_turns
        auction_log
    """

    def __init__(
        self,
        row,
    ):

        obs = row["obs"]

        # ----------------------------------------------------
        # Basic game state
        # ----------------------------------------------------

        self.seat = int(
            obs["seat"]
        )

        self.round = int(
            obs["round"]
        )

        self.is_maker = bool(
            obs["is_maker"]
        )

        # ----------------------------------------------------
        # Our information
        # ----------------------------------------------------

        self.k_mine = int(
            obs["k_mine"]
        )

        # The dataset stores only the aggregate foresight
        # information, not the individual coins.
        self.foresight_sum = int(
            obs.get(
                "foresight_sum",
                0,
            )
        )

        self.foresight_n = int(
            obs.get(
                "foresight_n",
                0,
            )
        )

        # ----------------------------------------------------
        # TE
        # ----------------------------------------------------

        self.te_mine = int(
            obs["te_mine"]
        )

        self.te_theirs = int(
            obs["te_theirs"]
        )

        # ----------------------------------------------------
        # Quote constraints
        # ----------------------------------------------------

        self.spread_cap = int(
            obs["spread_cap"]
        )

        self.final_cap = int(
            obs["final_cap"]
        )

        # ----------------------------------------------------
        # Public powers
        # ----------------------------------------------------

        self.powers_mine = frozenset(
            str(x)
            for x in obs[
                "powers_mine"
            ]
        )

        self.powers_theirs = frozenset(
            str(x)
            for x in obs[
                "powers_theirs"
            ]
        )

        # ----------------------------------------------------
        # Other state fields retained for completeness
        # ----------------------------------------------------

        self.n_unknown_both = int(
            obs.get(
                "n_unknown_both",
                0,
            )
        )

        self.n_turns = int(
            obs.get(
                "n_turns",
                6,
            )
        )


# ============================================================
# EXACT CURRENT VALUE
# ============================================================

def exact_value(
    obs: ResearchObs,
) -> float:
    """
    Exact research value available from this serialized Obs.

    Unknown fair coins have expectation zero.

        E[S]
        =
        k_mine
        +
        foresight_sum
    """

    return (
        float(obs.k_mine)
        +
        float(obs.foresight_sum)
    )


# ============================================================
# FORMAT ACTION
# ============================================================

def format_action(
    action,
):

    if action.kind in (
        "ACCEPT_BUY",
        "ACCEPT_SELL",
    ):
        return action.kind

    return (
        f"COUNTER "
        f"[{action.bid},{action.ask}]"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "research/opponent/"
            "phase3_policy_dataset.json"
        ),
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 4B — "
        "PUBLIC OPPONENT-CONDITIONED EV"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_rows(
        args.input
    )

    train, validation = (
        split_by_seed(rows)
    )

    print(
        f"Total rows:       {len(rows):,}"
    )

    print(
        f"Training rows:    {len(train):,}"
    )

    print(
        f"Validation rows:  {len(validation):,}"
    )

    # --------------------------------------------------------
    # Fit public model
    # --------------------------------------------------------

    print()
    print(
        "FITTING PUBLIC RESPONSE MODEL..."
    )

    model = (
        PublicOpponentModel()
        .fit(train)
    )

    print(
        f"Public states: "
        f"{len(model.tables):,}"
    )

    print(
        f"Global response counts: "
        f"{dict(model.global_counts)}"
    )

    # --------------------------------------------------------
    # Find representative validation states
    # --------------------------------------------------------

    examples = []

    seen = set()

    for row in validation:

        if row["method"] != "respond":
            continue

        quote = normalize_quote(
            row["quote"]
        )

        key = (
            int(row["seed"]),
            int(row["obs"]["round"]),
            int(row["turn"]),
            int(quote[0]),
            int(quote[1]),
        )

        if key in seen:
            continue

        seen.add(key)

        examples.append(
            row
        )

        if len(examples) >= args.examples:
            break

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for i, row in enumerate(
        examples,
        start=1,
    ):

        obs = ResearchObs(
            row
        )

        quote = normalize_quote(
            row["quote"]
        )

        turn = int(
            row["turn"]
        )

        ranked = evaluate_all_actions(
            obs=obs,
            current_quote=quote,
            opponent_model=model,
            turn=turn,
            final_cap=obs.final_cap,
            min_reduction=DEFAULT_CONFIG.MIN_REDUCTION,
            forcing_fee=DEFAULT_CONFIG.FORCED_FILL_FEE,
        )

        if not ranked:
            raise RuntimeError(
                "No legal actions generated "
                f"for quote={quote}"
            )

        best = ranked[0]

        print()
        print("-" * 72)

        print(
            f"EXAMPLE {i}"
        )

        print(
            f"Seed: {row['seed']}   "
            f"Round: {obs.round}   "
            f"Turn: {turn}   "
            f"Quote: {quote}"
        )

        print(
            f"Exact value: "
            f"{exact_value(obs):+.2f}"
        )

        print(
            f"Foresight: "
            f"sum={obs.foresight_sum:+d}, "
            f"n={obs.foresight_n}"
        )

        print()

        print(
            f"{'Action':<25}"
            f"{'EV':>12}"
            f"{'P(B)':>10}"
            f"{'P(S)':>10}"
            f"{'P(C)':>10}"
        )

        print("-" * 72)

        for result in ranked[:10]:

            print(
                f"{format_action(result.action):<25}"
                f"{result.ev:>+12.6f}"
                f"{result.p_accept_buy:>10.4f}"
                f"{result.p_accept_sell:>10.4f}"
                f"{result.p_counter:>10.4f}"
            )

        print()

        print(
            "OBSERVED ACTION:",
            row["action"],
        )

        print(
            "BEST:",
            format_action(
                best.action
            ),
            f"EV={best.ev:+.6f}",
        )

        results.append({
            "seed": int(
                row["seed"]
            ),

            "round": obs.round,

            "turn": turn,

            "quote": list(
                quote
            ),

            "observed_action":
                row["action"],

            "best_action":
                format_action(
                    best.action
                ),

            "best_ev":
                best.ev,

            "exact_value":
                exact_value(obs),

            "foresight_sum":
                obs.foresight_sum,

            "foresight_n":
                obs.foresight_n,

            "ranking": [
                {
                    "action":
                        format_action(
                            x.action
                        ),

                    "ev":
                        x.ev,

                    "p_accept_buy":
                        x.p_accept_buy,

                    "p_accept_sell":
                        x.p_accept_sell,

                    "p_counter":
                        x.p_counter,
                }
                for x in ranked
            ],
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        ROOT
        / "research"
        / "results"
        / "phase4b_examples.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "phase": "4B",
                "description":
                    "Public-state opponent-conditioned "
                    "one-step negotiation EV",
                "n_examples":
                    len(results),
                "examples":
                    results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 72)

    print(
        f"Saved: {output}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()