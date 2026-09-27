from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def entropy(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    return -sum(
        (n / total) * math.log2(n / total)
        for n in counts.values()
        if n > 0
    )


def make_state_key(row):
    obs = row["obs"]
    method = str(row["method"])

    round_no = int(obs["round"])
    k_mine = int(obs["k_mine"])
    te_mine = int(obs["te_mine"])
    te_theirs = int(obs["te_theirs"])

    powers_mine = tuple(sorted(str(x) for x in obs["powers_mine"]))
    powers_theirs = tuple(sorted(str(x) for x in obs["powers_theirs"]))

    offered = tuple(sorted(str(x) for x in row.get("offered", [])))

    foresight_n = int(obs["foresight_n"])
    foresight_sum = int(obs["foresight_sum"])

    if method == "respond":
        quote = row["quote"]
        bid = int(quote[0])
        ask = int(quote[1])

        width_band = (ask - bid) // 2

        midpoint = (bid + ask) / 2.0
        midpoint_band = round(midpoint - k_mine) // 2

        turn = int(row["turn"])

    else:
        width_band = -1
        midpoint_band = -1
        turn = -1

    return (
        method,
        round_no,
        bool(obs["is_maker"]),
        k_mine // 4,
        te_mine // 4,
        te_theirs // 4,
        bool(foresight_n > 0),
        foresight_sum // 2,
        powers_mine,
        powers_theirs,
        offered,
        width_band,
        midpoint_band,
        turn,
    )


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


def load_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def split_by_seed(rows, validation_fraction=0.30):
    """
    Deterministic deal-level split.

    Every decision from a seed stays in the same partition.
    """

    seeds = sorted({
        int(row["seed"])
        for row in rows
    })

    n_validation = max(
        1,
        int(len(seeds) * validation_fraction)
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

    return train, validation


def fit_policy(rows):
    counts = defaultdict(Counter)

    for row in rows:
        key = make_state_key(row)
        action = action_label(row)
        counts[key][action] += 1

    return counts


def evaluate(policy, rows):
    total = 0
    correct = 0

    log_loss_sum = 0.0

    method_total = Counter()
    method_correct = Counter()

    confidence_buckets = Counter()

    unseen_states = 0

    # Laplace smoothing.
    alpha = 1.0

    global_actions = Counter()

    for counts in policy.values():
        global_actions.update(counts)

    all_actions = sorted(global_actions)

    for row in rows:

        key = make_state_key(row)
        actual = action_label(row)

        total += 1
        method = row["method"]
        method_total[method] += 1

        if key not in policy:
            unseen_states += 1

            # Global fallback.
            fallback_total = sum(global_actions.values())

            probabilities = {
                a: (
                    global_actions[a] + alpha
                )
                / (
                    fallback_total
                    + alpha * len(all_actions)
                )
                for a in all_actions
            }

        else:
            counts = policy[key]

            state_total = sum(counts.values())

            probabilities = {
                a: (
                    counts.get(a, 0) + alpha
                )
                / (
                    state_total
                    + alpha * len(all_actions)
                )
                for a in all_actions
            }

        prediction = max(
            probabilities.items(),
            key=lambda x: x[1],
        )[0]

        p_actual = probabilities.get(
            actual,
            1.0 / (
                sum(probabilities.values())
                or 1
            ),
        )

        if prediction == actual:
            correct += 1
            method_correct[method] += 1

        log_loss_sum -= math.log(
            max(p_actual, 1e-12)
        )

        confidence = probabilities[prediction]

        if confidence >= 0.90:
            confidence_buckets[">=0.90"] += 1
        elif confidence >= 0.75:
            confidence_buckets["0.75-0.90"] += 1
        elif confidence >= 0.60:
            confidence_buckets["0.60-0.75"] += 1
        else:
            confidence_buckets["<0.60"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "log_loss": log_loss_sum / total if total else 0.0,
        "unseen_states": unseen_states,
        "unseen_rate": unseen_states / total if total else 0.0,
        "method_total": dict(method_total),
        "method_correct": dict(method_correct),
        "method_accuracy": {
            method: (
                method_correct[method]
                / count
            )
            for method, count in method_total.items()
        },
        "confidence": dict(confidence_buckets),
    }


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
            "phase3_validation.json"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    print("=" * 68)
    print("PHASE 3A — HELD-OUT OPPONENT POLICY VALIDATION")
    print("=" * 68)

    rows = load_rows(args.input)

    print(f"Total observations: {len(rows):,}")

    train, validation = split_by_seed(rows)

    print(f"Training observations:   {len(train):,}")
    print(f"Validation observations: {len(validation):,}")

    train_seeds = sorted({
        int(x["seed"])
        for x in train
    })

    validation_seeds = sorted({
        int(x["seed"])
        for x in validation
    })

    print(
        f"Training seeds:     "
        f"{train_seeds[0]} → {train_seeds[-1]}"
    )

    print(
        f"Validation seeds:   "
        f"{validation_seeds[0]} → {validation_seeds[-1]}"
    )

    print()
    print("FITTING EMPIRICAL POLICY...")

    policy = fit_policy(train)

    print(
        f"Training states: {len(policy):,}"
    )

    print()
    print("VALIDATING...")

    result = evaluate(
        policy,
        validation,
    )

    print()
    print("=" * 68)
    print("VALIDATION RESULT")
    print("=" * 68)

    print(
        f"Overall accuracy: "
        f"{result['accuracy']:.4%}"
    )

    print(
        f"Mean log loss:    "
        f"{result['log_loss']:.6f}"
    )

    print(
        f"Unseen states:    "
        f"{result['unseen_states']:,}"
    )

    print(
        f"Unseen rate:      "
        f"{result['unseen_rate']:.4%}"
    )

    print()
    print("BY DECISION TYPE")

    for method, count in result["method_total"].items():

        accuracy = result["method_accuracy"][method]

        print(
            f"  {method:16s}"
            f" n={count:7,}"
            f" accuracy={accuracy:.4%}"
        )

    print()
    print("PREDICTION CONFIDENCE")

    for bucket, count in result["confidence"].items():

        print(
            f"  {bucket:12s}"
            f" {count:10,}"
        )

    output = {
        "phase": "3A",
        "input": str(args.input),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "train_observations": len(train),
        "validation_observations": len(validation),
        "train_states": len(policy),
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
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved: {args.output}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()