from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ALPHA = 1.0


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
            f"{p}={v}"
            for p, v in sorted(action.items())
        )

    return "UNKNOWN"


def representation_coarse(row):
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
        midpoint_band = round(
            ((bid + ask) / 2) - k
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
        bool(int(obs["foresight_n"]) > 0),
        int(obs["foresight_sum"]) // 2,
        tuple(sorted(obs["powers_mine"])),
        tuple(sorted(obs["powers_theirs"])),
        tuple(sorted(row.get("offered", []))),
        width_band,
        midpoint_band,
        turn,
    )


def representation_exact(row):
    obs = row["obs"]
    method = row["method"]

    if method == "respond":
        quote = row["quote"]
        bid = int(quote[0])
        ask = int(quote[1])
        turn = int(row["turn"])
    else:
        bid = -999
        ask = -999
        turn = -1

    return (
        method,
        int(obs["round"]),
        int(obs["k_mine"]),
        int(obs["te_mine"]),
        int(obs["te_theirs"]),
        bool(obs["is_maker"]),
        tuple(sorted(obs["powers_mine"])),
        tuple(sorted(obs["powers_theirs"])),
        tuple(sorted(row.get("offered", []))),
        int(obs["foresight_n"]),
        int(obs["foresight_sum"]),
        bid,
        ask,
        turn,
    )


def representation_engineered(row):
    obs = row["obs"]
    method = row["method"]

    k = int(obs["k_mine"])
    te = int(obs["te_mine"])
    opp_te = int(obs["te_theirs"])

    if method == "respond":
        quote = row["quote"]
        bid = int(quote[0])
        ask = int(quote[1])

        width = ask - bid
        midpoint = (bid + ask) / 2.0

        midpoint_offset = round(midpoint - k)

        turn = int(row["turn"])
    else:
        width = -1
        midpoint_offset = -999
        turn = -1

    return (
        method,
        int(obs["round"]),

        # Exact private observable value.
        k,

        # Exact resource state.
        te,
        opp_te,
        te - opp_te,

        bool(obs["is_maker"]),

        # Power state.
        tuple(sorted(obs["powers_mine"])),
        tuple(sorted(obs["powers_theirs"])),
        tuple(sorted(row.get("offered", []))),

        # Foresight.
        int(obs["foresight_n"]),
        int(obs["foresight_sum"]),

        # Negotiation geometry.
        width,
        midpoint_offset,
        turn,
    )


def fit(rows, key_function):
    table = defaultdict(Counter)

    for row in rows:
        key = key_function(row)
        action = action_label(row)
        table[key][action] += 1

    return table


def evaluate(table, rows):
    total = 0
    correct = 0
    unseen = 0

    log_loss = 0.0

    by_method_total = Counter()
    by_method_correct = Counter()

    for row in rows:
        key = table_key = None

        # Caller attaches the representation key.
        key = row["_comparison_key"]

        actual = action_label(row)
        method = row["method"]

        total += 1
        by_method_total[method] += 1

        if key not in table:
            unseen += 1

            # Unknown state: use a uniform fallback.
            all_actions = set()

            for counts in table.values():
                all_actions.update(counts.keys())

            if not all_actions:
                continue

            p = 1.0 / len(all_actions)

            prediction = None

            # No information: this observation is not counted
            # as a correct state prediction.
            log_loss -= math.log(p)

        else:
            counts = table[key]

            state_total = sum(counts.values())
            actions = set(counts)

            prediction = max(
                counts.items(),
                key=lambda x: x[1],
            )[0]

            probabilities = {
                a: (
                    counts.get(a, 0) + ALPHA
                )
                / (
                    state_total
                    + ALPHA * len(actions)
                )
                for a in actions
            }

            p = probabilities.get(
                actual,
                1e-12,
            )

            log_loss -= math.log(
                max(p, 1e-12)
            )

        if prediction == actual:
            correct += 1
            by_method_correct[method] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "log_loss": log_loss / total if total else 0.0,
        "unseen": unseen,
        "unseen_rate": unseen / total if total else 0.0,
        "method_accuracy": {
            method: (
                by_method_correct[method]
                / count
            )
            for method, count in by_method_total.items()
        },
    }


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

    return train, validation


def run_representation(name, key_function, train, validation):

    train_table = fit(
        train,
        key_function,
    )

    validation_rows = []

    for row in validation:
        copy = dict(row)
        copy["_comparison_key"] = key_function(row)
        validation_rows.append(copy)

    result = evaluate(
        train_table,
        validation_rows,
    )

    result["representation"] = name
    result["train_states"] = len(train_table)

    return result


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
            "phase3_state_comparison.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print("PHASE 3B — STATE REPRESENTATION COMPARISON")
    print("=" * 72)

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:
        dataset = json.load(f)

    rows = dataset["rows"]

    train, validation = split_by_seed(rows)

    print(
        f"Total observations:    {len(rows):,}"
    )
    print(
        f"Training observations: {len(train):,}"
    )
    print(
        f"Validation observations: "
        f"{len(validation):,}"
    )

    representations = [
        (
            "Coarse",
            representation_coarse,
        ),
        (
            "Exact",
            representation_exact,
        ),
        (
            "Engineered",
            representation_engineered,
        ),
    ]

    results = []

    for name, fn in representations:

        print()
        print(
            f"Evaluating {name}..."
        )

        result = run_representation(
            name,
            fn,
            train,
            validation,
        )

        results.append(result)

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"{'Representation':<18}"
        f"{'States':>10}"
        f"{'Accuracy':>13}"
        f"{'LogLoss':>12}"
        f"{'Unseen':>12}"
    )

    print("-" * 72)

    for result in results:

        print(
            f"{result['representation']:<18}"
            f"{result['train_states']:>10,}"
            f"{result['accuracy']:>12.4%}"
            f"{result['log_loss']:>12.4f}"
            f"{result['unseen_rate']:>11.4%}"
        )

    print()
    print("BY DECISION TYPE")
    print()

    methods = [
        "quote",
        "respond",
        "bid",
        "use_transform",
    ]

    for result in results:

        print(
            f"{result['representation']}:"
        )

        for method in methods:

            accuracy = result[
                "method_accuracy"
            ].get(method)

            if accuracy is not None:
                print(
                    f"  {method:16s}"
                    f" {accuracy:.4%}"
                )

    output = {
        "phase": "3B",
        "input": str(args.input),
        "results": results,
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
    print("=" * 72)


if __name__ == "__main__":
    main()