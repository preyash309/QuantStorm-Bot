from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ALPHA = 1.0


# ============================================================
# ACTION LABEL
# ============================================================

def action_label(row):
    method = row["method"]
    action = row["action"]

    if method == "respond":
        if isinstance(action, list) and action:
            return str(action[0])
        return str(action)

    if method == "use_transform":
        return "USE_TRANSFORM" if bool(action) else "NO_TRANSFORM"

    if method == "quote":
        return f"QUOTE_WIDTH_{int(action[1]) - int(action[0])}"

    if method == "bid":
        if not action:
            return "NO_BID"

        return "BID|" + "|".join(
            f"{power}={amount}"
            for power, amount in sorted(action.items())
        )

    return "UNKNOWN"


# ============================================================
# SAME COARSE STATE USED IN PHASE 3A / 3B
# ============================================================

def coarse_state(row):
    obs = row["obs"]
    method = row["method"]

    k = int(obs["k_mine"])
    te = int(obs["te_mine"])
    opp_te = int(obs["te_theirs"])

    if method == "respond":

        quote = row["quote"]

        bid = int(quote[0])
        ask = int(quote[1])

        width_band = (ask - bid) // 2

        midpoint = (bid + ask) / 2.0

        midpoint_band = round(
            midpoint - k
        ) // 2

        turn = int(row["turn"])

    else:

        width_band = -1
        midpoint_band = -1
        turn = -1

    return (
        method,
        int(obs["round"]),
        bool(obs["is_maker"]),

        k // 4,
        te // 4,
        opp_te // 4,

        bool(
            int(obs["foresight_n"]) > 0
        ),

        int(obs["foresight_sum"]) // 2,

        tuple(
            sorted(
                str(x)
                for x in obs["powers_mine"]
            )
        ),

        tuple(
            sorted(
                str(x)
                for x in obs["powers_theirs"]
            )
        ),

        tuple(
            sorted(
                str(x)
                for x in row.get(
                    "offered",
                    []
                )
            )
        ),

        width_band,
        midpoint_band,
        turn,
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_by_seed(rows):

    seeds = sorted({
        int(row["seed"])
        for row in rows
    })

    n_validation = max(
        1,
        int(len(seeds) * 0.30)
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
        sorted(
            set(seeds) - validation_seeds
        ),
        sorted(validation_seeds),
    )


# ============================================================
# FIT EMPIRICAL POLICY
# ============================================================

def fit_policy(rows):

    table = defaultdict(Counter)
    global_counts = Counter()

    for row in rows:

        state = coarse_state(row)
        action = action_label(row)

        table[state][action] += 1
        global_counts[action] += 1

    return table, global_counts


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def probabilities(counts, action_space):

    total = sum(
        counts.values()
    )

    denominator = (
        total
        + ALPHA * len(action_space)
    )

    return {
        action: (
            counts.get(action, 0)
            + ALPHA
        ) / denominator
        for action in action_space
    }


# ============================================================
# BRIER SCORE
# ============================================================

def brier_score(
    probs,
    actual,
    action_space,
):

    return sum(
        (
            probs.get(action, 0.0)
            - (
                1.0
                if action == actual
                else 0.0
            )
        ) ** 2
        for action in action_space
    )


# ============================================================
# CALIBRATION BUCKET
# ============================================================

def confidence_bucket(p):

    if p < 0.50:
        return "<0.50"

    if p < 0.60:
        return "0.50-0.60"

    if p < 0.70:
        return "0.60-0.70"

    if p < 0.80:
        return "0.70-0.80"

    if p < 0.90:
        return "0.80-0.90"

    return ">=0.90"


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    policy,
    global_counts,
    rows,
):

    # --------------------------------------------------------
    # Global action universe
    # --------------------------------------------------------

    action_space = sorted(
        global_counts.keys()
    )

    total = 0

    correct = 0

    log_loss_total = 0.0
    brier_total = 0.0

    unseen = 0

    method_total = Counter()
    method_correct = Counter()

    method_log_loss = Counter()
    method_brier = Counter()

    calibration = defaultdict(
        lambda: {
            "n": 0,
            "probability_sum": 0.0,
            "correct": 0,
        }
    )

    # --------------------------------------------------------
    # Validation loop
    # --------------------------------------------------------

    for row in rows:

        state = coarse_state(row)

        actual = action_label(row)

        method = row["method"]

        total += 1
        method_total[method] += 1

        if state in policy:

            counts = policy[state]

            probs = probabilities(
                counts,
                action_space,
            )

        else:

            unseen += 1

            probs = probabilities(
                global_counts,
                action_space,
            )

        prediction = max(
            probs.items(),
            key=lambda x: x[1],
        )[0]

        p_actual = max(
            probs.get(actual, 0.0),
            1e-12,
        )

        ll = -math.log(
            p_actual
        )

        bs = brier_score(
            probs,
            actual,
            action_space,
        )

        log_loss_total += ll
        brier_total += bs

        method_log_loss[method] += ll
        method_brier[method] += bs

        if prediction == actual:

            correct += 1
            method_correct[method] += 1

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        bucket = confidence_bucket(
            p_actual
        )

        calibration[bucket]["n"] += 1

        calibration[bucket][
            "probability_sum"
        ] += p_actual

        calibration[bucket][
            "correct"
        ] += int(
            prediction == actual
        )

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    method_results = {}

    for method, n in method_total.items():

        method_results[method] = {
            "n": n,

            "accuracy": (
                method_correct[method]
                / n
            ),

            "log_loss": (
                method_log_loss[method]
                / n
            ),

            "brier": (
                method_brier[method]
                / n
            ),
        }

    calibration_results = {}

    for bucket, data in calibration.items():

        n = data["n"]

        calibration_results[bucket] = {

            "n": n,

            "mean_predicted_probability": (
                data["probability_sum"]
                / n
            ),

            "empirical_accuracy": (
                data["correct"]
                / n
            ),
        }

    return {

        "n": total,

        "accuracy": (
            correct / total
        ),

        "log_loss": (
            log_loss_total / total
        ),

        "brier": (
            brier_total / total
        ),

        "unseen": unseen,

        "unseen_rate": (
            unseen / total
        ),

        "method": method_results,

        "calibration": calibration_results,
    }


# ============================================================
# COMPACT POLICY EXPORT
# ============================================================

def build_compact_policy(
    policy,
    min_observations=3,
):

    result = {}

    for state, counts in policy.items():

        n = sum(
            counts.values()
        )

        if n < min_observations:
            continue

        probs = probabilities(
            counts,
            sorted(counts.keys()),
        )

        # JSON-compatible state representation.
        key = json.dumps(
            state,
            separators=(",", ":"),
            default=list,
        )

        result[key] = {
            "n": n,
            "counts": dict(counts),
            "probabilities": probs,
            "best_action": max(
                probs.items(),
                key=lambda x: x[1],
            )[0],
        }

    return result


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
            "phase3c_calibration.json"
        ),
    )

    parser.add_argument(
        "--policy-output",
        type=Path,
        default=Path(
            "research/results/"
            "phase3c_policy.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 3C — OPPONENT PROBABILITY "
        "CALIBRATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:

        dataset = json.load(f)

    rows = dataset["rows"]

    print(
        f"Total observations: {len(rows):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train,
        validation,
        train_seeds,
        validation_seeds,
    ) = split_by_seed(rows)

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Validation observations: "
        f"{len(validation):,}"
    )

    print(
        f"Training seeds: "
        f"{train_seeds[0]} → "
        f"{train_seeds[-1]}"
    )

    print(
        f"Validation seeds: "
        f"{validation_seeds[0]} → "
        f"{validation_seeds[-1]}"
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    print()
    print("FITTING COARSE POLICY...")

    policy, global_counts = fit_policy(
        train
    )

    print(
        f"Policy states: {len(policy):,}"
    )

    print(
        f"Global actions: "
        f"{len(global_counts)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("CALIBRATING ON HELD-OUT SEEDS...")

    result = evaluate(
        policy,
        global_counts,
        validation,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 3C RESULT")
    print("=" * 72)

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

    print(
        f"Unseen:      "
        f"{result['unseen']:,} "
        f"({result['unseen_rate']:.4%})"
    )

    # --------------------------------------------------------
    # Method results
    # --------------------------------------------------------

    print()
    print("BY DECISION TYPE")

    print(
        f"{'Method':<18}"
        f"{'N':>10}"
        f"{'Accuracy':>14}"
        f"{'LogLoss':>14}"
        f"{'Brier':>14}"
    )

    print("-" * 72)

    for method, data in result[
        "method"
    ].items():

        print(
            f"{method:<18}"
            f"{data['n']:>10,}"
            f"{data['accuracy']:>13.4%}"
            f"{data['log_loss']:>14.6f}"
            f"{data['brier']:>14.6f}"
        )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print()
    print("CALIBRATION")

    print(
        f"{'Bucket':<14}"
        f"{'N':>10}"
        f"{'Predicted':>16}"
        f"{'Actual':>16}"
    )

    print("-" * 72)

    order = [
        "<0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        ">=0.90",
    ]

    for bucket in order:

        data = result[
            "calibration"
        ].get(bucket)

        if data is None:
            continue

        print(
            f"{bucket:<14}"
            f"{data['n']:>10,}"
            f"{data['mean_predicted_probability']:>15.4f}"
            f"{data['empirical_accuracy']:>15.4f}"
        )

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    analysis = {
        "phase": "3C",
        "input": str(args.input),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "train_observations": len(train),
        "validation_observations": len(validation),
        "policy_states": len(policy),
        "result": result,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            analysis,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save compact policy
    # --------------------------------------------------------

    compact = build_compact_policy(
        policy
    )

    policy_output = {
        "phase": "3C",
        "representation": "coarse",
        "states": compact,
    }

    with args.policy_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            policy_output,
            f,
            separators=(",", ":"),
        )

    print()
    print(
        f"Saved analysis: "
        f"{args.output}"
    )

    print(
        f"Saved policy:   "
        f"{args.policy_output}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()