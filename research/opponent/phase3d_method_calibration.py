from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ALPHA = 1.0


# ============================================================
# ACTION LABELS
# ============================================================

def action_label(row):
    method = row["method"]
    action = row["action"]

    if method == "respond":
        if isinstance(action, list) and action:
            return str(action[0])
        return str(action)

    if method == "use_transform":
        return (
            "USE_TRANSFORM"
            if bool(action)
            else "NO_TRANSFORM"
        )

    if method == "quote":
        bid = int(action[0])
        ask = int(action[1])
        return f"QUOTE_WIDTH_{ask - bid}"

    if method == "bid":
        if not action:
            return "NO_BID"

        return "BID|" + "|".join(
            f"{power}={amount}"
            for power, amount in sorted(action.items())
        )

    return "UNKNOWN"


# ============================================================
# COARSE STATE
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
# SPLIT BY SEED
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
# FIT METHOD-SPECIFIC POLICY
# ============================================================

def fit_policy(rows):

    tables = defaultdict(
        lambda: defaultdict(Counter)
    )

    global_counts = defaultdict(Counter)

    for row in rows:

        method = str(row["method"])

        state = coarse_state(row)

        action = action_label(row)

        tables[method][state][action] += 1

        global_counts[method][action] += 1

    return tables, global_counts


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def make_probabilities(
    counts,
    action_space,
):

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
# BRIER
# ============================================================

def brier(
    probabilities,
    actual,
    action_space,
):

    return sum(
        (
            probabilities.get(
                action,
                0.0,
            )
            -
            (
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

def probability_bucket(p):

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
# EVALUATION
# ============================================================

def evaluate_method(
    method,
    policy,
    global_counts,
    rows,
):

    # --------------------------------------------------------
    # Action space belongs ONLY to this method.
    # --------------------------------------------------------

    action_space = sorted(
        global_counts.keys()
    )

    total = 0
    correct = 0
    unseen = 0

    log_loss_total = 0.0
    brier_total = 0.0

    calibration = defaultdict(
        lambda: {
            "n": 0,
            "predicted_sum": 0.0,
            "correct": 0,
        }
    )

    for row in rows:

        if row["method"] != method:
            continue

        total += 1

        state = coarse_state(row)

        actual = action_label(row)

        if state in policy:

            counts = policy[state]

            probabilities = make_probabilities(
                counts,
                action_space,
            )

        else:

            unseen += 1

            probabilities = make_probabilities(
                global_counts,
                action_space,
            )

        prediction = max(
            probabilities.items(),
            key=lambda x: x[1],
        )[0]

        p_actual = max(
            probabilities.get(
                actual,
                0.0,
            ),
            1e-12,
        )

        log_loss_total -= math.log(
            p_actual
        )

        brier_total += brier(
            probabilities,
            actual,
            action_space,
        )

        if prediction == actual:
            correct += 1

        bucket = probability_bucket(
            p_actual
        )

        calibration[bucket]["n"] += 1

        calibration[bucket][
            "predicted_sum"
        ] += p_actual

        calibration[bucket][
            "correct"
        ] += int(
            prediction == actual
        )

    if total == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
            "unseen": 0,
            "unseen_rate": 0.0,
            "calibration": {},
            "action_space": action_space,
        }

    calibration_result = {}

    for bucket, data in calibration.items():

        n = data["n"]

        calibration_result[bucket] = {
            "n": n,
            "mean_predicted_probability": (
                data["predicted_sum"] / n
            ),
            "empirical_accuracy": (
                data["correct"] / n
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

        "action_space": action_space,

        "calibration": calibration_result,
    }


# ============================================================
# SPECIAL RESPONSE ANALYSIS
# ============================================================

def response_summary(
    policy,
    global_counts,
    rows,
):

    method = "respond"

    action_space = sorted(
        global_counts[method].keys()
    )

    result = {
        "n": 0,
        "accept_buy": 0,
        "accept_sell": 0,
        "counter": 0,
        "probability_sum": {
            "ACCEPT_BUY": 0.0,
            "ACCEPT_SELL": 0.0,
            "COUNTER": 0.0,
        },
    }

    for row in rows:

        if row["method"] != method:
            continue

        result["n"] += 1

        state = coarse_state(row)

        if state in policy:
            counts = policy[state]
        else:
            counts = global_counts[method]

        probs = make_probabilities(
            counts,
            action_space,
        )

        for action in result[
            "probability_sum"
        ]:

            result[
                "probability_sum"
            ][action] += probs.get(
                action,
                0.0,
            )

        actual = action_label(row)

        if actual == "ACCEPT_BUY":
            result["accept_buy"] += 1

        elif actual == "ACCEPT_SELL":
            result["accept_sell"] += 1

        elif actual == "COUNTER":
            result["counter"] += 1

    if result["n"] > 0:

        result["empirical_frequency"] = {
            "ACCEPT_BUY":
                result["accept_buy"]
                / result["n"],

            "ACCEPT_SELL":
                result["accept_sell"]
                / result["n"],

            "COUNTER":
                result["counter"]
                / result["n"],
        }

        result["mean_predicted_probability"] = {
            action:
                value / result["n"]
            for action, value
            in result["probability_sum"].items()
        }

    return result


# ============================================================
# BUILD COMPACT POLICY
# ============================================================

def build_compact_policy(
    policy,
    minimum_observations=3,
):

    result = {}

    for state, counts in policy.items():

        n = sum(
            counts.values()
        )

        if n < minimum_observations:
            continue

        action_space = sorted(
            counts.keys()
        )

        probs = make_probabilities(
            counts,
            action_space,
        )

        state_key = json.dumps(
            state,
            separators=(",", ":"),
            default=list,
        )

        result[state_key] = {
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
            "phase3d_calibration.json"
        ),
    )

    parser.add_argument(
        "--policy-output",
        type=Path,
        default=Path(
            "research/results/"
            "phase3d_policy.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 3D — METHOD-SPECIFIC "
        "OPPONENT CALIBRATION"
    )
    print("=" * 72)

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:

        dataset = json.load(f)

    rows = dataset["rows"]

    print(
        f"Total observations: "
        f"{len(rows):,}"
    )

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

    print()
    print("FITTING METHOD-SPECIFIC POLICIES...")

    tables, global_counts = fit_policy(
        train
    )

    methods = [
        "respond",
        "quote",
        "bid",
        "use_transform",
    ]

    results = {}

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        f"{'Method':<18}"
        f"{'N':>10}"
        f"{'Accuracy':>14}"
        f"{'LogLoss':>14}"
        f"{'Brier':>14}"
        f"{'Unseen':>12}"
    )

    print("-" * 72)

    for method in methods:

        result = evaluate_method(
            method,
            tables[method],
            global_counts[method],
            validation,
        )

        results[method] = result

        print(
            f"{method:<18}"
            f"{result['n']:>10,}"
            f"{result['accuracy']:>13.4%}"
            f"{result['log_loss']:>14.6f}"
            f"{result['brier']:>14.6f}"
            f"{result['unseen_rate']:>11.4%}"
        )

    # --------------------------------------------------------
    # Response-specific analysis
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESPONSE POLICY")
    print("=" * 72)

    response = response_summary(
        tables["respond"],
        global_counts,
        validation,
    )

    print(
        f"Observations: "
        f"{response['n']:,}"
    )

    print()
    print("EMPIRICAL FREQUENCY")

    for action, value in response[
        "empirical_frequency"
    ].items():

        print(
            f"  {action:<14}"
            f"{value:.4%}"
        )

    print()
    print("MEAN PREDICTED PROBABILITY")

    for action, value in response[
        "mean_predicted_probability"
    ].items():

        print(
            f"  {action:<14}"
            f"{value:.4%}"
        )

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    analysis = {
        "phase": "3D",

        "representation": "coarse",

        "train_seeds": train_seeds,

        "validation_seeds":
            validation_seeds,

        "train_observations":
            len(train),

        "validation_observations":
            len(validation),

        "results": results,

        "response_summary":
            response,
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
    # Save method-specific policies
    # --------------------------------------------------------

    compact = {}

    for method in methods:

        compact[method] = (
            build_compact_policy(
                tables[method]
            )
        )

    policy_output = {
        "phase": "3D",
        "representation": "coarse",
        "policies": compact,
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