"""
PHASE 4D — Quote-conditioned negotiation EV.

Uses the Phase 4C quote-conditioned opponent model inside the
existing exact negotiation EV engine.

Evaluation is performed on held-out seeds.

Important:
    This measures model-implied EV, not counterfactual realized PnL.
    True counterfactual PnL requires simulator replay and comes later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from game_config import DEFAULT_CONFIG

from research.ev.quote_opponent import (
    QuoteOpponentModel,
)

from research.ev.continuation import (
    evaluate_all_actions,
)


# ============================================================
# LOAD
# ============================================================

def load_rows(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if "rows" not in data:
        raise RuntimeError(
            "Dataset does not contain top-level 'rows'."
        )

    return data["rows"]


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
        validation_seeds,
    )


# ============================================================
# REPRESENTATIVE OBS
# ============================================================

class ResearchObs:

    def __init__(
        self,
        row,
    ):

        obs = row["obs"]

        self.round = int(
            obs["round"]
        )

        self.is_maker = bool(
            obs["is_maker"]
        )

        self.k_mine = int(
            obs["k_mine"]
        )

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

        # Compatibility with older research EV code.
        self.foresight = ()

        self.te_mine = int(
            obs["te_mine"]
        )

        self.te_theirs = int(
            obs["te_theirs"]
        )

        self.powers_mine = frozenset(
            str(x)
            for x in obs.get(
                "powers_mine",
                [],
            )
        )

        self.powers_theirs = frozenset(
            str(x)
            for x in obs.get(
                "powers_theirs",
                [],
            )
        )

        self.final_cap = int(
            obs["final_cap"]
        )

        self.spread_cap = int(
            obs["spread_cap"]
        )


# ============================================================
# ACTION NORMALIZATION
# ============================================================

def normalize_action(action):

    # --------------------------------------------------------
    # NegotiationAction object
    # --------------------------------------------------------
    #
    # Objects returned by evaluate_all_actions() have:
    #
    #   action.kind
    #   action.bid
    #   action.ask
    #
    # Handle these BEFORE list/tuple handling.
    # --------------------------------------------------------

    if hasattr(action, "kind"):

        kind = str(
            action.kind
        )

        if kind == "COUNTER":

            return (
                kind,
                int(action.bid),
                int(action.ask),
            )

        return (
            kind,
            None,
            None,
        )

    # --------------------------------------------------------
    # String action
    # --------------------------------------------------------

    if isinstance(
        action,
        str,
    ):

        return (
            action,
            None,
            None,
        )

    # --------------------------------------------------------
    # Dataset list / tuple action
    # --------------------------------------------------------

    if isinstance(
        action,
        (list, tuple),
    ):

        if len(action) == 0:

            return (
                "",
                None,
                None,
            )

        kind = str(
            action[0]
        )

        if kind == "COUNTER":

            if len(action) >= 3:

                return (
                    kind,
                    int(action[1]),
                    int(action[2]),
                )

        return (
            kind,
            None,
            None,
        )

    # --------------------------------------------------------
    # Unknown action representation
    # --------------------------------------------------------

    return (
        str(action),
        None,
        None,
    )

def same_action(
    a,
    b,
):

    ak, ab, aa = normalize_action(
        a
    )

    bk, bb, ba = normalize_action(
        b
    )

    if ak != bk:
        return False

    if ak == "COUNTER":

        return (
            ab == bb
            and
            aa == ba
        )

    return True


def format_action(
    action,
):

    kind, bid, ask = normalize_action(
        action
    )

    if kind == "COUNTER":

        return (
            f"COUNTER "
            f"[{bid},{ask}]"
        )

    return kind


# ============================================================
# QUOTE WIDTH
# ============================================================

def get_quote(
    row,
):

    quote = row.get(
        "quote"
    )

    if quote is None:
        return None

    if len(quote) != 2:
        return None

    return (
        int(quote[0]),
        int(quote[1]),
    )


def quote_width(
    row,
):

    quote = get_quote(
        row
    )

    if quote is None:
        return None

    return (
        quote[1]
        - quote[0]
    )


# ============================================================
# METRICS
# ============================================================

class Bucket:

    def __init__(self):

        self.n = 0

        self.best_ev = 0.0

        self.observed_ev = 0.0

        self.improvement = 0.0

        self.policy_changed = 0

    def add(
        self,
        best_ev,
        observed_ev,
        changed,
    ):

        self.n += 1

        self.best_ev += float(
            best_ev
        )

        self.observed_ev += float(
            observed_ev
        )

        self.improvement += (
            float(best_ev)
            - float(observed_ev)
        )

        if changed:
            self.policy_changed += 1

    def summary(self):

        if self.n == 0:

            return {
                "n": 0,
                "mean_best_ev": 0.0,
                "mean_observed_ev": 0.0,
                "mean_ev_improvement": 0.0,
                "policy_change_rate": 0.0,
            }

        return {
            "n": self.n,

            "mean_best_ev":
                self.best_ev / self.n,

            "mean_observed_ev":
                self.observed_ev / self.n,

            "mean_ev_improvement":
                self.improvement / self.n,

            "policy_change_rate":
                self.policy_changed / self.n,
        }


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
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional limit for debugging. "
            "Default: use all validation rows."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 4D — "
        "QUOTE-CONDITIONED NEGOTIATION EV"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    rows = load_rows(
        args.input
    )

    train, validation, validation_seeds = (
        split_by_seed(rows)
    )

    train_respond = [
        row
        for row in train
        if row.get("method") == "respond"
        and get_quote(row) is not None
    ]

    validation_respond = [
        row
        for row in validation
        if row.get("method") == "respond"
        and get_quote(row) is not None
    ]

    if args.max_rows is not None:

        validation_respond = (
            validation_respond[
                :args.max_rows
            ]
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

    print(
        f"Training respond: {len(train_respond):,}"
    )

    print(
        f"Validation respond:{len(validation_respond):,}"
    )

    # --------------------------------------------------------
    # FIT 4C MODEL
    # --------------------------------------------------------

    print()
    print(
        "FITTING PHASE 4C QUOTE MODEL..."
    )

    model = (
        QuoteOpponentModel()
        .fit(train_respond)
    )

    print(
        f"Quote-conditioned states: "
        f"{model.n_states:,}"
    )

    # --------------------------------------------------------
    # BUCKETS
    # --------------------------------------------------------

    overall = Bucket()

    by_turn = defaultdict(
        Bucket
    )

    by_width = defaultdict(
        Bucket
    )

    seen = Bucket()
    unseen = Bucket()

    policy_changes = 0

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print()
    print(
        "EVALUATING HELD-OUT NEGOTIATIONS..."
    )

    for index, row in enumerate(
        validation_respond,
        start=1,
    ):

        obs = ResearchObs(
            row
        )

        current_quote = get_quote(
            row
        )

        turn = int(
            row["turn"]
        )

        # ----------------------------------------------------
        # Rank all legal actions using the 4C model.
        #
        # evaluate_all_actions will query the opponent model
        # for the candidate quote being evaluated.
        # ----------------------------------------------------

        ranked = evaluate_all_actions(
            obs=obs,
            current_quote=current_quote,
            opponent_model=model,
            turn=turn,
            final_cap=obs.final_cap,
            min_reduction=DEFAULT_CONFIG.MIN_REDUCTION,
            forcing_fee=DEFAULT_CONFIG.FORCED_FILL_FEE,
        )

        if not ranked:
            continue

        best = ranked[0]

        # ----------------------------------------------------
        # Find the EV assigned to the historical action.
        # ----------------------------------------------------

        historical_action = row[
            "action"
        ]

        observed_result = None

        for result in ranked:

            if same_action(
                result.action,
                historical_action,
            ):

                observed_result = result
                break

        # Historical action may not be among the legal
        # candidates generated by the research engine.
        if observed_result is None:
            continue

        best_ev = float(
            best.ev
        )

        observed_ev = float(
            observed_result.ev
        )

        changed = not same_action(
            best.action,
            historical_action,
        )

        overall.add(
            best_ev,
            observed_ev,
            changed,
        )

        by_turn[
            turn
        ].add(
            best_ev,
            observed_ev,
            changed,
        )

        width = quote_width(
            row
        )

        if width is not None:

            by_width[
                width
            ].add(
                best_ev,
                observed_ev,
                changed,
            )

        # ----------------------------------------------------
        # Seen / unseen
        # ----------------------------------------------------

        key = model.state_key(
            row
        )

        if key in model.tables:

            seen.add(
                best_ev,
                observed_ev,
                changed,
            )

        else:

            unseen.add(
                best_ev,
                observed_ev,
                changed,
            )

        if changed:
            policy_changes += 1

        # ----------------------------------------------------
        # Occasional progress
        # ----------------------------------------------------

        if index % 5000 == 0:

            print(
                f"  processed "
                f"{index:,}/"
                f"{len(validation_respond):,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    result = overall.summary()

    print()
    print("=" * 72)
    print(
        "PHASE 4D RESULT"
    )
    print("=" * 72)

    print(
        f"Evaluated observations: "
        f"{result['n']:,}"
    )

    print()

    print(
        f"Mean EV of best 4C action: "
        f"{result['mean_best_ev']:+.6f}"
    )

    print(
        f"Mean EV of historical action: "
        f"{result['mean_observed_ev']:+.6f}"
    )

    print(
        f"Mean EV improvement: "
        f"{result['mean_ev_improvement']:+.6f}"
    )

    print(
        f"Policy changed: "
        f"{policy_changes:,} / "
        f"{result['n']:,}"
        f" = "
        f"{result['policy_change_rate'] * 100:.4f}%"
    )

    # ========================================================
    # SEEN / UNSEEN
    # ========================================================

    print()
    print(
        "SEEN / UNSEEN"
    )

    print(
        f"{'Bucket':<12}"
        f"{'N':>10}"
        f"{'Best EV':>14}"
        f"{'Observed EV':>16}"
        f"{'Improvement':>16}"
        f"{'Change %':>12}"
    )

    print("-" * 80)

    for name, bucket in (
        ("Seen", seen),
        ("Unseen", unseen),
    ):

        x = bucket.summary()

        if x["n"] == 0:
            continue

        print(
            f"{name:<12}"
            f"{x['n']:>10,}"
            f"{x['mean_best_ev']:>+14.6f}"
            f"{x['mean_observed_ev']:>+16.6f}"
            f"{x['mean_ev_improvement']:>+16.6f}"
            f"{x['policy_change_rate'] * 100:>11.3f}%"
        )

    # ========================================================
    # BY TURN
    # ========================================================

    print()
    print(
        "BY TURN"
    )

    print(
        f"{'Turn':<8}"
        f"{'N':>10}"
        f"{'Best EV':>14}"
        f"{'Observed EV':>16}"
        f"{'Improvement':>16}"
        f"{'Change %':>12}"
    )

    print("-" * 80)

    for turn in sorted(
        by_turn
    ):

        x = by_turn[
            turn
        ].summary()

        print(
            f"{turn:<8}"
            f"{x['n']:>10,}"
            f"{x['mean_best_ev']:>+14.6f}"
            f"{x['mean_observed_ev']:>+16.6f}"
            f"{x['mean_ev_improvement']:>+16.6f}"
            f"{x['policy_change_rate'] * 100:>11.3f}%"
        )

    # ========================================================
    # BY QUOTE WIDTH
    # ========================================================

    print()
    print(
        "BY QUOTE WIDTH"
    )

    print(
        f"{'Width':<8}"
        f"{'N':>10}"
        f"{'Best EV':>14}"
        f"{'Observed EV':>16}"
        f"{'Improvement':>16}"
        f"{'Change %':>12}"
    )

    print("-" * 80)

    for width in sorted(
        by_width
    ):

        x = by_width[
            width
        ].summary()

        print(
            f"{width:<8}"
            f"{x['n']:>10,}"
            f"{x['mean_best_ev']:>+14.6f}"
            f"{x['mean_observed_ev']:>+16.6f}"
            f"{x['mean_ev_improvement']:>+16.6f}"
            f"{x['policy_change_rate'] * 100:>11.3f}%"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = (
        ROOT
        / "research"
        / "results"
        / "phase4d_validation.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "phase": "4D",

        "total_rows":
            len(rows),

        "training_rows":
            len(train),

        "validation_rows":
            len(validation),

        "evaluated_observations":
            result["n"],

        "validation_seeds":
            sorted(
                validation_seeds
            ),

        "overall":
            result,

        "seen":
            seen.summary(),

        "unseen":
            unseen.summary(),

        "by_turn": {
            str(k):
                v.summary()
            for k, v
            in by_turn.items()
        },

        "by_quote_width": {
            str(k):
                v.summary()
            for k, v
            in by_width.items()
        },

        "method":
            "quote_conditioned_EV",

        "warning":
            (
                "EV improvement is model-implied. "
                "It is not counterfactual realized PnL."
            ),
    }

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_data,
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