"""
PHASE 4E.0 — Exact opponent counter-policy model.

Phase 3 intentionally collapsed all COUNTER actions into the
single label "COUNTER".

For counterfactual replay we need the exact counter:

    COUNTER [bid, ask]

This phase learns:

    P(exact opponent action | public information state)

using the same deterministic state representation as Phase 3.

This phase DOES NOT perform counterfactual replay yet.
It only validates whether we can predict the opponent's exact
response action on held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys


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


from research.opponent.features import (
    state_key,
)

from research.ev.negotiation_ev import (
    NegotiationAction,
    enumerate_actions,
)


# ============================================================
# DATASET
# ============================================================

def load_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data["rows"]


# ============================================================
# SEED SPLIT
# ============================================================

def split_by_seed(
    rows,
    validation_fraction=0.30,
):
    """
    Exact same deal-level split used by Phase 3.
    """

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
# EXACT ACTION LABEL
# ============================================================

def exact_action_label(row):
    """
    Preserve exact negotiation action.

    Examples:

        ACCEPT_BUY
        ACCEPT_SELL
        COUNTER|-1|2
        COUNTER|0|1
    """

    action = row["action"]

    if isinstance(action, str):
        return str(action)

    if not isinstance(action, list):
        return str(action)

    if not action:
        return "UNKNOWN"

    kind = str(
        action[0]
    )

    if kind == "COUNTER":

        if len(action) < 3:
            return "COUNTER|INVALID"

        return (
            f"COUNTER|"
            f"{int(action[1])}|"
            f"{int(action[2])}"
        )

    return kind


# ============================================================
# ACTION DECODING
# ============================================================

def parse_action_label(label):
    """
    Convert an exact action label back into NegotiationAction.
    """

    label = str(label)

    if label == "ACCEPT_BUY":

        return NegotiationAction(
            kind="ACCEPT_BUY"
        )

    if label == "ACCEPT_SELL":

        return NegotiationAction(
            kind="ACCEPT_SELL"
        )

    parts = label.split("|")

    if (
        len(parts) == 3
        and parts[0] == "COUNTER"
    ):

        return NegotiationAction(
            kind="COUNTER",
            bid=int(parts[1]),
            ask=int(parts[2]),
        )

    return None


# ============================================================
# STATE KEY
# ============================================================

def make_key(row):
    """
    Reuse the exact Phase-3 state abstraction.

    This is important: Phase 4E.0 must not accidentally
    introduce a different information state.
    """

    obs = row["obs"]

    return state_key(
        obs,
        method="respond",
        quote=row["quote"],
        turn=int(row["turn"]),
        offered=row.get(
            "offered",
            (),
        ),
    )


# ============================================================
# MODEL
# ============================================================

class ExactCounterPolicy:

    def __init__(
        self,
        alpha=1.0,
    ):
        self.alpha = float(alpha)

        self.counts = defaultdict(
            Counter
        )

        self.global_counts = Counter()

        self.actions = set()

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(
        self,
        rows,
    ):

        for row in rows:

            if row["method"] != "respond":
                continue

            label = exact_action_label(
                row
            )

            key = make_key(
                row
            )

            self.counts[key][
                label
            ] += 1

            self.global_counts[
                label
            ] += 1

            self.actions.add(
                label
            )

        return self

    # --------------------------------------------------------
    # RAW PROBABILITIES
    # --------------------------------------------------------

    def probabilities(
        self,
        key,
    ):
        """
        Smoothed empirical distribution.
        """

        labels = sorted(
            self.actions
        )

        if not labels:
            return {}

        if key in self.counts:

            counts = self.counts[
                key
            ]

        else:

            counts = self.global_counts

        total = sum(
            counts.get(
                action,
                0,
            )
            for action in labels
        )

        denominator = (
            total
            + self.alpha
            * len(labels)
        )

        return {
            action:
                (
                    counts.get(
                        action,
                        0,
                    )
                    + self.alpha
                )
                / denominator

            for action in labels
        }

    # --------------------------------------------------------
    # LEGAL ACTION FILTER
    # --------------------------------------------------------

    def legal_probabilities(
        self,
        key,
        quote,
        final_cap,
        min_reduction,
    ):
        """
        Restrict the learned distribution to actions that are
        actually legal for the current quote.

        This is essential for counterfactual replay.

        A model prediction learned from one quote must never
        produce an illegal counter for another quote.
        """

        probabilities = self.probabilities(
            key
        )

        legal_actions = enumerate_actions(
            bid=int(quote[0]),
            ask=int(quote[1]),
            final_cap=int(final_cap),
            min_reduction=int(
                min_reduction
            ),
        )

        legal_labels = {
            self.action_to_label(
                action
            )
            for action in legal_actions
        }

        filtered = {
            label: probability
            for label, probability
            in probabilities.items()
            if label in legal_labels
        }

        # ----------------------------------------------------
        # If no learned action survives the legal filter,
        # fall back to uniform over the legal action set.
        # ----------------------------------------------------

        if not filtered:

            if not legal_labels:
                return {}

            uniform = (
                1.0
                / len(legal_labels)
            )

            return {
                label: uniform
                for label in sorted(
                    legal_labels
                )
            }

        total = sum(
            filtered.values()
        )

        if total <= 0:

            uniform = (
                1.0
                / len(filtered)
            )

            return {
                label: uniform
                for label in filtered
            }

        return {
            label:
                probability / total

            for label, probability
            in filtered.items()
        }

    # --------------------------------------------------------
    # LABEL CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def action_to_label(
        action,
    ):

        if action.kind == "ACCEPT_BUY":
            return "ACCEPT_BUY"

        if action.kind == "ACCEPT_SELL":
            return "ACCEPT_SELL"

        return (
            f"COUNTER|"
            f"{int(action.bid)}|"
            f"{int(action.ask)}"
        )

    # --------------------------------------------------------
    # TOP ACTION
    # --------------------------------------------------------

    def top_action(
        self,
        key,
        quote,
        final_cap,
        min_reduction,
    ):

        probabilities = (
            self.legal_probabilities(
                key=key,
                quote=quote,
                final_cap=final_cap,
                min_reduction=min_reduction,
            )
        )

        if not probabilities:
            return None

        label = max(
            probabilities.items(),
            key=lambda x: x[1],
        )[0]

        return (
            parse_action_label(
                label
            ),
            probabilities[label],
        )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    rows,
):
    """
    Evaluate exact action prediction on held-out rows.
    """

    total = 0
    correct = 0

    counter_total = 0
    counter_correct = 0

    unseen = 0

    log_loss_sum = 0.0

    action_counts = Counter()
    predicted_counts = Counter()

    confidence_sum = 0.0

    for row in rows:

        if row["method"] != "respond":
            continue

        total += 1

        key = make_key(
            row
        )

        if key not in model.counts:
            unseen += 1

        actual = exact_action_label(
            row
        )

        action_counts[
            actual
        ] += 1

        # ----------------------------------------------------
        # Need the public quote limits.
        #
        # Dataset obs already contains final_cap.
        # ----------------------------------------------------

        obs = row["obs"]

        quote = row["quote"]

        probabilities = (
            model.legal_probabilities(
                key=key,
                quote=quote,
                final_cap=int(
                    obs["final_cap"]
                ),
                min_reduction=1,
            )
        )

        if not probabilities:
            continue

        prediction = max(
            probabilities.items(),
            key=lambda x: x[1],
        )[0]

        predicted_counts[
            prediction
        ] += 1

        p_actual = probabilities.get(
            actual,
            1e-12,
        )

        log_loss_sum -= math.log(
            max(
                p_actual,
                1e-12,
            )
        )

        confidence_sum += (
            probabilities[
                prediction
            ]
        )

        if prediction == actual:
            correct += 1

        if actual.startswith(
            "COUNTER|"
        ):

            counter_total += 1

            if prediction == actual:
                counter_correct += 1

    return {
        "total": total,

        "correct": correct,

        "accuracy": (
            correct / total
            if total
            else 0.0
        ),

        "log_loss": (
            log_loss_sum / total
            if total
            else 0.0
        ),

        "unseen": unseen,

        "unseen_rate": (
            unseen / total
            if total
            else 0.0
        ),

        "counter_total":
            counter_total,

        "counter_correct":
            counter_correct,

        "counter_accuracy": (
            counter_correct
            / counter_total
            if counter_total
            else 0.0
        ),

        "mean_confidence": (
            confidence_sum
            / total
            if total
            else 0.0
        ),

        "actual_actions":
            dict(action_counts),

        "predicted_actions":
            dict(predicted_counts),
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
        "--output",
        type=Path,
        default=Path(
            "research/results/"
            "phase4e0_counter_policy.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 4E.0 — "
        "EXACT OPPONENT COUNTER POLICY"
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

    train_respond = [
        row
        for row in train
        if row["method"] == "respond"
    ]

    validation_respond = [
        row
        for row in validation
        if row["method"] == "respond"
    ]

    print(
        f"Total rows:       "
        f"{len(rows):,}"
    )

    print(
        f"Training rows:    "
        f"{len(train):,}"
    )

    print(
        f"Validation rows:  "
        f"{len(validation):,}"
    )

    print()

    print(
        f"Training respond: "
        f"{len(train_respond):,}"
    )

    print(
        f"Validation respond:"
        f"{len(validation_respond):,}"
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    print()
    print(
        "FITTING EXACT COUNTER POLICY..."
    )

    model = (
        ExactCounterPolicy(
            alpha=1.0
        )
        .fit(
            train_respond
        )
    )

    print(
        f"Policy states: "
        f"{len(model.counts):,}"
    )

    print(
        f"Exact actions: "
        f"{len(model.actions):,}"
    )

    # --------------------------------------------------------
    # Count counters
    # --------------------------------------------------------

    counter_labels = sorted(
        label
        for label in model.actions
        if label.startswith(
            "COUNTER|"
        )
    )

    print(
        f"Exact counter actions: "
        f"{len(counter_labels):,}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print(
        "VALIDATING..."
    )

    result = validate(
        model,
        validation_respond,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "PHASE 4E.0 RESULT"
    )
    print("=" * 72)

    print(
        f"Validation observations: "
        f"{result['total']:,}"
    )

    print(
        f"Exact action accuracy:    "
        f"{result['accuracy']:.4%}"
    )

    print(
        f"Exact action log loss:     "
        f"{result['log_loss']:.6f}"
    )

    print(
        f"Unseen states:             "
        f"{result['unseen']:,}"
    )

    print(
        f"Unseen rate:               "
        f"{result['unseen_rate']:.4%}"
    )

    print()

    print(
        "COUNTER PREDICTION"
    )

    print(
        f"Counter observations:      "
        f"{result['counter_total']:,}"
    )

    print(
        f"Exact counter accuracy:    "
        f"{result['counter_accuracy']:.4%}"
    )

    print()

    print(
        f"Mean prediction confidence:"
        f" {result['mean_confidence']:.4f}"
    )

    print()

    print(
        "TOP ACTUAL ACTIONS"
    )

    for action, count in (
        Counter(
            result["actual_actions"]
        )
        .most_common(20)
    ):

        print(
            f"  {action:25s}"
            f" {count:8,}"
        )

    print()

    print(
        "TOP PREDICTED ACTIONS"
    )

    for action, count in (
        Counter(
            result["predicted_actions"]
        )
        .most_common(20)
    ):

        print(
            f"  {action:25s}"
            f" {count:8,}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "phase": "4E.0",

        "description":
            "Exact opponent action policy "
            "for counterfactual replay",

        "train_rows":
            len(train_respond),

        "validation_rows":
            len(validation_respond),

        "states":
            len(model.counts),

        "actions":
            sorted(model.actions),

        "result":
            result,
    }

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved: {args.output}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()