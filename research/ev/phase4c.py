"""
PHASE 4C — Held-out quote-conditioned opponent validation.

Compares:

    Phase 4B:
        public state -> response probabilities

against:

    Phase 4C:
        public state + exact quote + turn -> response probabilities

Evaluation is performed on held-out seeds.

Research only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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


from research.ev.public_opponent import (
    PublicOpponentModel,
)

from research.ev.quote_opponent import (
    QuoteOpponentModel,
)


ACTIONS = (
    "ACCEPT_BUY",
    "ACCEPT_SELL",
    "COUNTER",
)


# ============================================================
# DATA
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
# SEED SPLIT
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
# HELPERS
# ============================================================

def action_name(
    row,
):

    action = row["action"]

    if isinstance(
        action,
        list,
    ):
        action = action[0]

    action = str(action)

    if action in ACTIONS:
        return action

    return None


def quote_from_row(
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


def log_loss(
    probabilities,
    actual,
):

    p = float(
        probabilities.get(
            actual,
            0.0,
        )
    )

    p = max(
        1e-15,
        min(
            1.0,
            p,
        ),
    )

    return -math.log(p)


def brier(
    probabilities,
    actual,
):

    value = 0.0

    for action in ACTIONS:

        target = (
            1.0
            if action == actual
            else 0.0
        )

        p = float(
            probabilities.get(
                action,
                0.0,
            )
        )

        value += (
            p - target
        ) ** 2

    return value


def accuracy(
    probabilities,
    actual,
):

    predicted = max(
        ACTIONS,
        key=lambda x:
            probabilities.get(
                x,
                0.0,
            ),
    )

    return (
        predicted == actual
    )


# ============================================================
# QUOTE WIDTH
# ============================================================

def quote_width(
    row,
):

    quote = quote_from_row(
        row
    )

    if quote is None:
        return None

    return (
        quote[1]
        - quote[0]
    )


# ============================================================
# EVALUATION BUCKET
# ============================================================

class Metrics:

    def __init__(self):

        self.n = 0

        self.correct = 0

        self.logloss = 0.0

        self.brier = 0.0

    def add(
        self,
        probabilities,
        actual,
    ):

        self.n += 1

        if accuracy(
            probabilities,
            actual,
        ):
            self.correct += 1

        self.logloss += log_loss(
            probabilities,
            actual,
        )

        self.brier += brier(
            probabilities,
            actual,
        )

    def summary(self):

        if self.n == 0:

            return {
                "n": 0,
                "accuracy": None,
                "log_loss": None,
                "brier": None,
            }

        return {
            "n": self.n,

            "accuracy":
                self.correct
                / self.n,

            "log_loss":
                self.logloss
                / self.n,

            "brier":
                self.brier
                / self.n,
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

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 4C — "
        "HELD-OUT QUOTE-CONDITIONED OPPONENT VALIDATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_rows(
        args.input
    )

    train, validation, validation_seeds = (
        split_by_seed(rows)
    )

    response_train = [
        row
        for row in train
        if row.get("method") == "respond"
        and action_name(row) is not None
        and quote_from_row(row) is not None
    ]

    response_validation = [
        row
        for row in validation
        if row.get("method") == "respond"
        and action_name(row) is not None
        and quote_from_row(row) is not None
    ]

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
        f"Training respond: {len(response_train):,}"
    )

    print(
        f"Validation respond:{len(response_validation):,}"
    )

    # --------------------------------------------------------
    # Fit Phase 4B model
    # --------------------------------------------------------

    print()
    print(
        "FITTING PHASE 4B PUBLIC MODEL..."
    )

    public_model = (
        PublicOpponentModel()
        .fit(train)
    )

    print(
        f"Public states: "
        f"{len(public_model.tables):,}"
    )

    # --------------------------------------------------------
    # Fit Phase 4C model
    # --------------------------------------------------------

    print()
    print(
        "FITTING PHASE 4C QUOTE MODEL..."
    )

    quote_model = (
        QuoteOpponentModel()
        .fit(response_train)
    )

    print(
        f"Quote-conditioned states: "
        f"{quote_model.n_states:,}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    public_metrics = Metrics()
    quote_metrics = Metrics()

    by_method = {
        "public": defaultdict(Metrics),
        "quote": defaultdict(Metrics),
    }

    by_turn = {
        "public": defaultdict(Metrics),
        "quote": defaultdict(Metrics),
    }

    by_width = {
        "public": defaultdict(Metrics),
        "quote": defaultdict(Metrics),
    }

    unseen_quote = 0
    seen_quote = 0

    policy_changed = 0

    total = 0

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    for row in response_validation:

        actual = action_name(
            row
        )

        quote = quote_from_row(
            row
        )

        if actual is None:
            continue

        if quote is None:
            continue

        total += 1

        obs = row["obs"]

        round_no = int(
            obs["round"]
        )

        turn = int(
            row["turn"]
        )

        # ----------------------------------------------------
        # Phase 4B
        # ----------------------------------------------------

        public_prob = public_model.predict(
            round_no=round_no,
            is_maker=bool(
                obs["is_maker"]
            ),
            te_mine=int(
                obs["te_mine"]
            ),
            te_theirs=int(
                obs["te_theirs"]
            ),
            powers_mine=obs.get(
                "powers_mine",
                [],
            ),
            powers_theirs=obs.get(
                "powers_theirs",
                [],
            ),
            quote=quote,
            turn=turn,
        )

        # ----------------------------------------------------
        # Phase 4C
        # ----------------------------------------------------

        quote_prob = quote_model.predict_from_row(
            row
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        public_metrics.add(
            public_prob,
            actual,
        )

        quote_metrics.add(
            quote_prob,
            actual,
        )

        by_method[
            "public"
        ][
            actual
        ].add(
            public_prob,
            actual,
        )

        by_method[
            "quote"
        ][
            actual
        ].add(
            quote_prob,
            actual,
        )

        by_turn[
            "public"
        ][
            turn
        ].add(
            public_prob,
            actual,
        )

        by_turn[
            "quote"
        ][
            turn
        ].add(
            quote_prob,
            actual,
        )

        width = quote_width(
            row
        )

        if width is not None:

            by_width[
                "public"
            ][
                width
            ].add(
                public_prob,
                actual,
            )

            by_width[
                "quote"
            ][
                width
            ].add(
                quote_prob,
                actual,
            )

        # ----------------------------------------------------
        # Seen/unseen quote state
        # ----------------------------------------------------

        key = quote_model.state_key(
            row
        )

        if key in quote_model.tables:

            seen_quote += 1

        else:

            unseen_quote += 1

        # ----------------------------------------------------
        # Policy change
        # ----------------------------------------------------

        public_best = max(
            ACTIONS,
            key=lambda x:
                public_prob.get(
                    x,
                    0.0,
                ),
        )

        quote_best = max(
            ACTIONS,
            key=lambda x:
                quote_prob.get(
                    x,
                    0.0,
                ),
        )

        if public_best != quote_best:

            policy_changed += 1

    # ========================================================
    # RESULT
    # ========================================================

    public_result = (
        public_metrics.summary()
    )

    quote_result = (
        quote_metrics.summary()
    )

    print()
    print("=" * 72)
    print(
        "PHASE 4C RESULT"
    )
    print("=" * 72)

    print(
        f"Validation observations: "
        f"{total:,}"
    )

    print()

    print(
        f"{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'LogLoss':>12}"
        f"{'Brier':>12}"
    )

    print("-" * 58)

    print(
        f"{'Phase 4B public':<22}"
        f"{public_result['accuracy'] * 100:>11.4f}%"
        f"{public_result['log_loss']:>12.6f}"
        f"{public_result['brier']:>12.6f}"
    )

    print(
        f"{'Phase 4C quote':<22}"
        f"{quote_result['accuracy'] * 100:>11.4f}%"
        f"{quote_result['log_loss']:>12.6f}"
        f"{quote_result['brier']:>12.6f}"
    )

    print()
    print("IMPROVEMENT")

    accuracy_delta = (
        quote_result["accuracy"]
        - public_result["accuracy"]
    )

    logloss_delta = (
        quote_result["log_loss"]
        - public_result["log_loss"]
    )

    brier_delta = (
        quote_result["brier"]
        - public_result["brier"]
    )

    print(
        f"Accuracy delta: "
        f"{accuracy_delta * 100:+.4f} "
        f"percentage points"
    )

    print(
        f"LogLoss delta: "
        f"{logloss_delta:+.6f}"
    )

    print(
        f"Brier delta: "
        f"{brier_delta:+.6f}"
    )

    print()

    print(
        f"Seen quote states: "
        f"{seen_quote:,}"
    )

    print(
        f"Unseen quote states: "
        f"{unseen_quote:,}"
    )

    unseen_rate = (
        unseen_quote
        / max(1, total)
    )

    print(
        f"Unseen rate: "
        f"{unseen_rate * 100:.4f}%"
    )

    print()

    policy_change_rate = (
        policy_changed
        / max(1, total)
    )

    print(
        f"Predicted policy changed: "
        f"{policy_changed:,} / {total:,}"
        f" = {policy_change_rate * 100:.4f}%"
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
        f"{'4B Acc':>12}"
        f"{'4C Acc':>12}"
        f"{'4B LL':>12}"
        f"{'4C LL':>12}"
    )

    print("-" * 58)

    turns = sorted(
        set(
            by_turn["public"].keys()
        )
        |
        set(
            by_turn["quote"].keys()
        )
    )

    for turn in turns:

        a = by_turn[
            "public"
        ][turn].summary()

        b = by_turn[
            "quote"
        ][turn].summary()

        print(
            f"{turn:<8}"
            f"{a['accuracy'] * 100:>11.3f}%"
            f"{b['accuracy'] * 100:>11.3f}%"
            f"{a['log_loss']:>12.6f}"
            f"{b['log_loss']:>12.6f}"
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
        f"{'N':>8}"
        f"{'4B Acc':>12}"
        f"{'4C Acc':>12}"
        f"{'4B LL':>12}"
        f"{'4C LL':>12}"
    )

    print("-" * 64)

    widths = sorted(
        set(
            by_width["public"].keys()
        )
        |
        set(
            by_width["quote"].keys()
        )
    )

    for width in widths:

        a = by_width[
            "public"
        ][width].summary()

        b = by_width[
            "quote"
        ][width].summary()

        print(
            f"{width:<8}"
            f"{a['n']:>8}"
            f"{a['accuracy'] * 100:>11.3f}%"
            f"{b['accuracy'] * 100:>11.3f}%"
            f"{a['log_loss']:>12.6f}"
            f"{b['log_loss']:>12.6f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = (
        ROOT
        / "research"
        / "results"
        / "phase4c_validation.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {

        "phase": "4C",

        "training_rows":
            len(train),

        "validation_rows":
            len(validation),

        "validation_respond_rows":
            total,

        "validation_seeds":
            sorted(
                validation_seeds
            ),

        "phase4b":
            public_result,

        "phase4c":
            quote_result,

        "accuracy_delta":
            (
                quote_result["accuracy"]
                -
                public_result["accuracy"]
            ),

        "log_loss_delta":
            (
                quote_result["log_loss"]
                -
                public_result["log_loss"]
            ),

        "brier_delta":
            (
                quote_result["brier"]
                -
                public_result["brier"]
            ),

        "seen_quote_states":
            seen_quote,

        "unseen_quote_states":
            unseen_quote,

        "unseen_rate":
            unseen_quote
            / max(
                1,
                total,
            ),

        "policy_changed":
            policy_changed,

        "policy_change_rate":
            policy_changed
            / max(
                1,
                total,
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