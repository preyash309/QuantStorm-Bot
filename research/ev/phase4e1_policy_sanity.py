"""
PHASE 4E.1 — Exact opponent-policy class consistency.

Phase 4E.0 predicts exact actions:

    ACCEPT_BUY
    ACCEPT_SELL
    COUNTER|bid|ask

This phase collapses the exact-action distribution back into:

    ACCEPT_BUY
    ACCEPT_SELL
    COUNTER

and verifies that the exact model remains compatible with the
validated Phase 4C three-class opponent model.

This is a sanity check before counterfactual replay.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from research.opponent.features import (
    state_key,
)

from research.ev.phase4e0_counter_policy import (
    ExactCounterPolicy,
    exact_action_label,
    load_rows,
    split_by_seed,
    make_key,
)


# ============================================================
# CONSTANTS — PHASE 4C BENCHMARK
# ============================================================

PHASE4C_ACCURACY = 0.819042
PHASE4C_LOGLOSS = 0.549016
PHASE4C_BRIER = 0.296497


# ============================================================
# CLASS COLLAPSE
# ============================================================

def action_class(
    label: str,
) -> str:
    """
    Collapse an exact action into the Phase 4C class.
    """

    label = str(label)

    if label == "ACCEPT_BUY":
        return "ACCEPT_BUY"

    if label == "ACCEPT_SELL":
        return "ACCEPT_SELL"

    if label.startswith(
        "COUNTER|"
    ):
        return "COUNTER"

    return "UNKNOWN"


# ============================================================
# BRIER SCORE
# ============================================================

def multiclass_brier(
    probabilities,
    actual,
    labels,
):
    """
    Multiclass Brier score:

        sum_k (p_k - y_k)^2
    """

    score = 0.0

    for label in labels:

        p = float(
            probabilities.get(
                label,
                0.0,
            )
        )

        y = (
            1.0
            if label == actual
            else 0.0
        )

        score += (
            p - y
        ) ** 2

    return score


# ============================================================
# NORMALIZE
# ============================================================

def normalize(
    probabilities,
):
    total = sum(
        probabilities.values()
    )

    if total <= 0:
        return {}

    return {
        key:
            value / total

        for key, value
        in probabilities.items()
    }


# ============================================================
# COLLAPSE EXACT DISTRIBUTION
# ============================================================

def collapse_probabilities(
    exact_probabilities,
):
    """
    Convert:

        COUNTER|-1|2
        COUNTER|-2|2
        COUNTER|0|1
        ...

    into:

        COUNTER
    """

    result = {
        "ACCEPT_BUY": 0.0,
        "ACCEPT_SELL": 0.0,
        "COUNTER": 0.0,
    }

    for label, probability in (
        exact_probabilities.items()
    ):

        cls = action_class(
            label
        )

        if cls in result:
            result[cls] += float(
                probability
            )

    return normalize(
        result
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    rows,
):
    labels = (
        "ACCEPT_BUY",
        "ACCEPT_SELL",
        "COUNTER",
    )

    total = 0

    correct = 0

    log_loss = 0.0

    brier = 0.0

    unseen = 0

    policy_changes = 0

    confidence_sum = 0.0

    exact_counter_rows = 0

    exact_counter_top1 = 0

    actual_counts = Counter()

    predicted_counts = Counter()

    # --------------------------------------------------------
    # Confidence calibration buckets
    # --------------------------------------------------------

    buckets = {
        "<0.50": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },

        "0.50-0.60": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },

        "0.60-0.70": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },

        "0.70-0.80": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },

        "0.80-0.90": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },

        ">=0.90": {
            "n": 0,
            "correct": 0,
            "confidence": 0.0,
        },
    }

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for row in rows:

        if row.get(
            "method"
        ) != "respond":
            continue

        total += 1

        key = make_key(
            row
        )

        if key not in model.counts:
            unseen += 1

        # ----------------------------------------------------
        # Exact distribution.
        #
        # We intentionally use the same empirical model that
        # Phase 4E.0 validated.
        # ----------------------------------------------------

        exact_probs = (
            model.probabilities(
                key
            )
        )

        class_probs = (
            collapse_probabilities(
                exact_probs
            )
        )

        if not class_probs:
            continue

        actual_exact = (
            exact_action_label(
                row
            )
        )

        actual = action_class(
            actual_exact
        )

        actual_counts[
            actual
        ] += 1

        predicted = max(
            class_probs.items(),
            key=lambda x: x[1],
        )[0]

        predicted_counts[
            predicted
        ] += 1

        confidence = float(
            class_probs[
                predicted
            ]
        )

        confidence_sum += (
            confidence
        )

        if predicted == actual:
            correct += 1

        # ----------------------------------------------------
        # Log loss.
        # ----------------------------------------------------

        p_actual = max(
            float(
                class_probs.get(
                    actual,
                    0.0,
                )
            ),
            1e-12,
        )

        log_loss -= math.log(
            p_actual
        )

        # ----------------------------------------------------
        # Brier.
        # ----------------------------------------------------

        brier += (
            multiclass_brier(
                class_probs,
                actual,
                labels,
            )
        )

        # ----------------------------------------------------
        # Exact counter accuracy.
        # ----------------------------------------------------

        if actual_exact.startswith(
            "COUNTER|"
        ):

            exact_counter_rows += 1

            # Exact MAP prediction.
            exact_prediction = max(
                exact_probs.items(),
                key=lambda x: x[1],
            )[0]

            if (
                exact_prediction
                == actual_exact
            ):
                exact_counter_top1 += 1

        # ----------------------------------------------------
        # Confidence bucket.
        # ----------------------------------------------------

        if confidence < 0.50:
            bucket = buckets[
                "<0.50"
            ]

        elif confidence < 0.60:
            bucket = buckets[
                "0.50-0.60"
            ]

        elif confidence < 0.70:
            bucket = buckets[
                "0.60-0.70"
            ]

        elif confidence < 0.80:
            bucket = buckets[
                "0.70-0.80"
            ]

        elif confidence < 0.90:
            bucket = buckets[
                "0.80-0.90"
            ]

        else:
            bucket = buckets[
                ">=0.90"
            ]

        bucket["n"] += 1

        if predicted == actual:
            bucket["correct"] += 1

        bucket["confidence"] += (
            confidence
        )

    # --------------------------------------------------------
    # Return.
    # --------------------------------------------------------

    return {
        "total": total,

        "accuracy": (
            correct / total
            if total
            else 0.0
        ),

        "log_loss": (
            log_loss / total
            if total
            else 0.0
        ),

        "brier": (
            brier / total
            if total
            else 0.0
        ),

        "unseen": unseen,

        "unseen_rate": (
            unseen / total
            if total
            else 0.0
        ),

        "mean_confidence": (
            confidence_sum / total
            if total
            else 0.0
        ),

        "exact_counter_rows":
            exact_counter_rows,

        "exact_counter_top1":
            exact_counter_top1,

        "exact_counter_accuracy": (
            exact_counter_top1
            / exact_counter_rows
            if exact_counter_rows
            else 0.0
        ),

        "actual_counts":
            dict(actual_counts),

        "predicted_counts":
            dict(predicted_counts),

        "buckets": buckets,
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
            "phase4e1_policy_sanity.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)

    print(
        "PHASE 4E.1 — "
        "EXACT POLICY CLASS CONSISTENCY"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Load.
    # --------------------------------------------------------

    rows = load_rows(
        args.input
    )

    train, validation = (
        split_by_seed(
            rows
        )
    )

    train_respond = [
        row
        for row in train
        if row.get(
            "method"
        ) == "respond"
    ]

    validation_respond = [
        row
        for row in validation
        if row.get(
            "method"
        ) == "respond"
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
    # Fit exact model.
    # --------------------------------------------------------

    print()

    print(
        "FITTING PHASE 4E.0 MODEL..."
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
    # Evaluate.
    # --------------------------------------------------------

    print()

    print(
        "EVALUATING HELD-OUT "
        "CLASS DISTRIBUTIONS..."
    )

    result = evaluate(
        model,
        validation_respond,
    )

    # --------------------------------------------------------
    # Print result.
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "PHASE 4E.1 RESULT"
    )

    print("=" * 72)

    print(
        f"Validation observations: "
        f"{result['total']:,}"
    )

    print()

    print(
        "COLLAPSED THREE-CLASS MODEL"
    )

    print(
        f"Accuracy:    "
        f"{result['accuracy']:.4%}"
    )

    print(
        f"Log loss:    "
        f"{result['log_loss']:.6f}"
    )

    print(
        f"Brier score: "
        f"{result['brier']:.6f}"
    )

    print()

    print(
        "PHASE 4C BENCHMARK"
    )

    print(
        f"Accuracy:    "
        f"{PHASE4C_ACCURACY:.4%}"
    )

    print(
        f"Log loss:    "
        f"{PHASE4C_LOGLOSS:.6f}"
    )

    print(
        f"Brier score: "
        f"{PHASE4C_BRIER:.6f}"
    )

    print()

    print(
        "DELTA VS PHASE 4C"
    )

    print(
        f"Accuracy delta: "
        f"{result['accuracy'] - PHASE4C_ACCURACY:+.4%}"
    )

    print(
        f"LogLoss delta:  "
        f"{result['log_loss'] - PHASE4C_LOGLOSS:+.6f}"
    )

    print(
        f"Brier delta:    "
        f"{result['brier'] - PHASE4C_BRIER:+.6f}"
    )

    print()

    print(
        "EXACT COUNTER"
    )

    print(
        f"Counter observations: "
        f"{result['exact_counter_rows']:,}"
    )

    print(
        f"Exact counter top-1:   "
        f"{result['exact_counter_accuracy']:.4%}"
    )

    print()

    print(
        "UNSEEN"
    )

    print(
        f"Unseen states: "
        f"{result['unseen']:,}"
    )

    print(
        f"Unseen rate:   "
        f"{result['unseen_rate']:.4%}"
    )

    print()

    print(
        "ACTUAL CLASS DISTRIBUTION"
    )

    for label, count in (
        Counter(
            result[
                "actual_counts"
            ]
        ).most_common()
    ):

        print(
            f"  {label:15s}"
            f" {count:8,}"
        )

    print()

    print(
        "PREDICTED CLASS DISTRIBUTION"
    )

    for label, count in (
        Counter(
            result[
                "predicted_counts"
            ]
        ).most_common()
    ):

        print(
            f"  {label:15s}"
            f" {count:8,}"
        )

    print()

    print(
        "CONFIDENCE BUCKETS"
    )

    print(
        f"{'Bucket':<12}"
        f"{'N':>8}"
        f"{'Accuracy':>12}"
        f"{'MeanConf':>12}"
    )

    print(
        "-" * 44
    )

    for label, bucket in (
        result[
            "buckets"
        ].items()
    ):

        n = bucket["n"]

        if n:

            accuracy = (
                bucket["correct"]
                / n
            )

            confidence = (
                bucket["confidence"]
                / n
            )

        else:

            accuracy = 0.0
            confidence = 0.0

        print(
            f"{label:<12}"
            f"{n:>8,}"
            f"{accuracy:>11.4%}"
            f"{confidence:>12.4f}"
        )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "phase": "4E.1",

        "description":
            "Consistency check between "
            "exact-action opponent model "
            "and Phase 4C three-class policy",

        "phase4c_benchmark": {
            "accuracy":
                PHASE4C_ACCURACY,

            "log_loss":
                PHASE4C_LOGLOSS,

            "brier":
                PHASE4C_BRIER,
        },

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