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

    result = 0.0

    for count in counts.values():
        if count <= 0:
            continue

        p = count / total
        result -= p * math.log2(p)

    return result


def make_state_key(row):
    """
    Convert one recorded decision into a compact observable state key.

    IMPORTANT:
    This uses only information recorded in the dataset.
    """

    obs = row["obs"]

    method = row["method"]

    round_no = int(obs["round"])
    k_mine = int(obs["k_mine"])
    te_mine = int(obs["te_mine"])
    te_theirs = int(obs["te_theirs"])

    is_maker = bool(obs["is_maker"])

    powers_mine = tuple(
        sorted(str(x) for x in obs["powers_mine"])
    )

    powers_theirs = tuple(
        sorted(str(x) for x in obs["powers_theirs"])
    )

    foresight_n = int(obs["foresight_n"])
    foresight_sum = int(obs["foresight_sum"])

    offered = tuple(
        sorted(str(x) for x in row.get("offered", []))
    )

    # Coarse bins deliberately prevent the table from exploding.
    k_band = k_mine // 4
    te_band = te_mine // 4
    opp_te_band = te_theirs // 4
    foresight_band = foresight_sum // 2

    if method == "respond":

        quote = row["quote"]

        bid = int(quote[0])
        ask = int(quote[1])

        width = ask - bid
        midpoint = (bid + ask) / 2.0

        midpoint_offset = round(
            midpoint - k_mine
        )

        width_band = width // 2
        midpoint_band = midpoint_offset // 2

        turn = int(row["turn"])

    else:

        width_band = -1
        midpoint_band = -1
        turn = -1

    return (
        method,
        round_no,
        is_maker,
        k_band,
        te_band,
        opp_te_band,
        bool(foresight_n > 0),
        foresight_band,
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

        if isinstance(action, list):

            if len(action) > 0:
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

        # Keep individual power decisions separate.
        # Example:
        #   BID_TRANSFORM_4
        #   BID_FORESIGHT_2
        parts = []

        for power, amount in sorted(action.items()):

            parts.append(
                f"{power}={amount}"
            )

        return "BID|" + "|".join(parts)

    return "UNKNOWN"


def analyse(input_path, output_path):

    print("=" * 68)
    print("PHASE 3 POLICY ANALYSIS")
    print("=" * 68)
    print(f"Input: {input_path}")
    print()

    # Your actual dataset has:
    #
    # {
    #   phase,
    #   description,
    #   n_rows,
    #   rows
    # }
    #
    # So we load the container once.
    #
    # 90 MB is acceptable and this avoids the incorrect
    # "states" parser from the previous version.

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        dataset = json.load(f)

    rows = dataset["rows"]

    print(
        f"Dataset rows: {len(rows):,}"
    )

    # ----------------------------------------------------------
    # Global statistics
    # ----------------------------------------------------------

    method_counts = Counter()
    bot_counts = Counter()
    opponent_counts = Counter()

    action_counts = Counter()
    action_by_method = defaultdict(Counter)
    action_by_bot = defaultdict(Counter)

    # ----------------------------------------------------------
    # State -> action counts
    # ----------------------------------------------------------

    state_counts = defaultdict(Counter)

    # ----------------------------------------------------------
    # More useful raw statistics
    # ----------------------------------------------------------

    quote_widths = defaultdict(Counter)

    quote_offsets = defaultdict(Counter)

    transform_counts = defaultdict(Counter)

    bid_amounts = defaultdict(Counter)

    # ----------------------------------------------------------
    # Process rows
    # ----------------------------------------------------------

    for index, row in enumerate(rows):

        method = str(row["method"])

        bot = str(row["bot"])

        opponent = str(
            row["opponent"]
        )

        method_counts[method] += 1
        bot_counts[bot] += 1
        opponent_counts[opponent] += 1

        action = action_label(row)

        action_counts[action] += 1

        action_by_method[
            method
        ][action] += 1

        action_by_bot[
            bot
        ][action] += 1

        key = make_state_key(row)

        state_counts[key][action] += 1

        # ------------------------------------------------------
        # Quotes
        # ------------------------------------------------------

        if method == "quote":

            bid = int(row["action"][0])
            ask = int(row["action"][1])

            width = ask - bid

            k = int(
                row["obs"]["k_mine"]
            )

            midpoint = (
                bid + ask
            ) / 2.0

            offset = round(
                midpoint - k
            )

            quote_widths[
                key
            ][str(width)] += 1

            quote_offsets[
                key
            ][str(offset)] += 1

        # ------------------------------------------------------
        # Transform
        # ------------------------------------------------------

        elif method == "use_transform":

            transform_counts[
                key
            ][
                "USE"
                if bool(row["action"])
                else "NO"
            ] += 1

        # ------------------------------------------------------
        # Bids
        # ------------------------------------------------------

        elif method == "bid":

            action_dict = row["action"]

            for power, amount in action_dict.items():

                bid_amounts[
                    (key, str(power))
                ][
                    str(amount)
                ] += 1

    # ----------------------------------------------------------
    # Convert state distributions into policy table
    # ----------------------------------------------------------

    policy_states = []

    for key, counts in state_counts.items():

        total = sum(
            counts.values()
        )

        if total <= 0:
            continue

        ranked = counts.most_common()

        dominant_action = ranked[0][0]
        dominant_count = ranked[0][1]

        confidence = (
            dominant_count / total
        )

        probabilities = {
            action: round(
                count / total,
                6,
            )
            for action, count in counts.items()
        }

        policy_states.append(
            {
                "key": list(key),
                "n": total,
                "action": dominant_action,
                "confidence": round(
                    confidence,
                    6,
                ),
                "entropy": round(
                    entropy(counts),
                    6,
                ),
                "p": probabilities,
            }
        )

    # Most observed states first.
    policy_states.sort(
        key=lambda x: (
            x["n"],
            x["confidence"],
        ),
        reverse=True,
    )

    # ----------------------------------------------------------
    # Confidence statistics
    # ----------------------------------------------------------

    confidence_distribution = Counter()

    stable = 0
    ambiguous = 0

    for state in policy_states:

        c = state["confidence"]

        if c >= 0.90:

            confidence_distribution[
                ">=0.90"
            ] += 1

            stable += 1

        elif c >= 0.75:

            confidence_distribution[
                "0.75-0.90"
            ] += 1

            stable += 1

        elif c >= 0.60:

            confidence_distribution[
                "0.60-0.75"
            ] += 1

            ambiguous += 1

        else:

            confidence_distribution[
                "<0.60"
            ] += 1

            ambiguous += 1

    # ----------------------------------------------------------
    # Mean entropy
    # ----------------------------------------------------------

    if policy_states:

        mean_entropy = (
            sum(
                s["entropy"]
                for s in policy_states
            )
            / len(policy_states)
        )

    else:

        mean_entropy = 0.0

    # ----------------------------------------------------------
    # Save analysis
    # ----------------------------------------------------------

    analysis = {

        "phase": 3,

        "input": str(
            input_path
        ),

        "dataset_rows": len(rows),

        "methods": dict(
            method_counts
        ),

        "bots": dict(
            bot_counts
        ),

        "opponents": dict(
            opponent_counts
        ),

        "actions": dict(
            action_counts
        ),

        "actions_by_method": {
            k: dict(v)
            for k, v in action_by_method.items()
        },

        "actions_by_bot": {
            k: dict(v)
            for k, v in action_by_bot.items()
        },

        "state_count": len(
            policy_states
        ),

        "confidence_distribution": dict(
            confidence_distribution
        ),

        "stable_states": stable,

        "ambiguous_states": ambiguous,

        "mean_state_entropy": round(
            mean_entropy,
            6,
        ),

        "states": policy_states,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            analysis,
            f,
            separators=(",", ":"),
        )

    # ----------------------------------------------------------
    # Terminal report
    # ----------------------------------------------------------

    print()
    print("METHODS")

    for method, count in method_counts.most_common():

        print(
            f"  {method:16s}"
            f" {count:10,}"
        )

    print()
    print("BOTS")

    for bot, count in bot_counts.most_common():

        print(
            f"  {bot:20s}"
            f" {count:10,}"
        )

    print()
    print("ACTIONS")

    for action, count in action_counts.most_common():

        print(
            f"  {action:30s}"
            f" {count:10,}"
        )

    print()
    print("STATE ANALYSIS")

    print(
        f"  Unique states:       "
        f"{len(policy_states):,}"
    )

    print(
        f"  Stable states:       "
        f"{stable:,}"
    )

    print(
        f"  Ambiguous states:    "
        f"{ambiguous:,}"
    )

    print(
        f"  Mean entropy:        "
        f"{mean_entropy:.6f}"
    )

    print()
    print("CONFIDENCE")

    for bucket, count in (
        confidence_distribution.items()
    ):

        print(
            f"  {bucket:12s}"
            f" {count:10,}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    print("=" * 68)


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
            "phase3_analysis.json"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():

        raise FileNotFoundError(
            f"Dataset not found: {args.input}"
        )

    analyse(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()