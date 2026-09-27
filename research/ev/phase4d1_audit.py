from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_config import DEFAULT_CONFIG
from research.ev.quote_opponent import QuoteOpponentModel
from research.ev.continuation import evaluate_all_actions


ACTIONS = (
    "ACCEPT_BUY",
    "ACCEPT_SELL",
    "COUNTER",
)


def load_rows(path):

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["rows"]


def split_by_seed(rows, validation_fraction=0.30):

    seeds = sorted({
        int(row["seed"])
        for row in rows
    })

    n_validation = max(
        1,
        int(len(seeds) * validation_fraction),
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


class ResearchObs:

    def __init__(self, row):

        obs = row["obs"]

        self.round = int(obs["round"])
        self.is_maker = bool(obs["is_maker"])
        self.k_mine = int(obs["k_mine"])

        self.foresight_sum = int(
            obs.get("foresight_sum", 0)
        )

        self.foresight_n = int(
            obs.get("foresight_n", 0)
        )

        self.foresight = ()

        self.te_mine = int(obs["te_mine"])
        self.te_theirs = int(obs["te_theirs"])

        self.powers_mine = frozenset(
            str(x)
            for x in obs.get("powers_mine", [])
        )

        self.powers_theirs = frozenset(
            str(x)
            for x in obs.get("powers_theirs", [])
        )

        self.final_cap = int(obs["final_cap"])
        self.spread_cap = int(obs["spread_cap"])


def get_quote(row):

    q = row.get("quote")

    if q is None or len(q) != 2:
        return None

    return int(q[0]), int(q[1])


def normalize_action(action):

    # NegotiationAction
    if hasattr(action, "kind"):

        kind = str(action.kind)

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

    # Dataset list/tuple
    if isinstance(action, (list, tuple)):

        if not action:
            return ("", None, None)

        kind = str(action[0])

        if kind == "COUNTER" and len(action) >= 3:

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

    # String
    if isinstance(action, str):

        return (
            action,
            None,
            None,
        )

    return (
        str(action),
        None,
        None,
    )


def action_key(action):

    return normalize_action(action)


def format_action(action):

    kind, bid, ask = normalize_action(action)

    if kind == "COUNTER":

        return f"COUNTER [{bid},{ask}]"

    return kind


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
        default=20,
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "PHASE 4D.1 — SKIPPED ACTION AUDIT"
    )
    print("=" * 72)

    rows = load_rows(args.input)

    train, validation = split_by_seed(rows)

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

    print(
        f"Training respond:    {len(train_respond):,}"
    )

    print(
        f"Validation respond:  {len(validation_respond):,}"
    )

    print()
    print(
        "FITTING QUOTE MODEL..."
    )

    model = (
        QuoteOpponentModel()
        .fit(train_respond)
    )

    print(
        f"Quote states: {model.n_states:,}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0
    matched = 0
    skipped = 0

    skipped_by_action = Counter()

    skipped_by_turn = Counter()

    skipped_by_width = Counter()

    skipped_by_round = Counter()

    skipped_examples = []

    generated_action_counts = Counter()

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    for row in validation_respond:

        total += 1

        obs = ResearchObs(row)

        quote = get_quote(row)

        turn = int(row["turn"])

        historical = row["action"]

        ranked = evaluate_all_actions(
            obs=obs,
            current_quote=quote,
            opponent_model=model,
            turn=turn,
            final_cap=obs.final_cap,
            min_reduction=DEFAULT_CONFIG.MIN_REDUCTION,
            forcing_fee=DEFAULT_CONFIG.FORCED_FILL_FEE,
        )

        generated = [
            action_key(result.action)
            for result in ranked
        ]

        for action in generated:
            generated_action_counts[action] += 1

        historical_key = action_key(
            historical
        )

        if historical_key in generated:

            matched += 1
            continue

        # ----------------------------------------------------
        # SKIPPED
        # ----------------------------------------------------

        skipped += 1

        kind, bid, ask = normalize_action(
            historical
        )

        skipped_by_action[kind] += 1

        skipped_by_turn[turn] += 1

        skipped_by_round[
            int(row["obs"]["round"])
        ] += 1

        width = (
            quote[1] - quote[0]
            if quote is not None
            else None
        )

        if width is not None:
            skipped_by_width[width] += 1

        if len(skipped_examples) < args.examples:

            skipped_examples.append({
                "seed": int(row["seed"]),
                "round": int(row["obs"]["round"]),
                "turn": turn,
                "quote": list(quote)
                    if quote is not None
                    else None,
                "historical_action": repr(
                    historical
                ),
                "historical_normalized":
                    list(historical_key),
                "generated_actions": [
                    list(x)
                    for x in generated
                ],
            })

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 72)
    print(
        "AUDIT RESULT"
    )
    print("=" * 72)

    print(
        f"Total validation observations: {total:,}"
    )

    print(
        f"Matched:                       {matched:,}"
    )

    print(
        f"Skipped:                       {skipped:,}"
    )

    print(
        f"Coverage:                      "
        f"{matched / max(1, total) * 100:.4f}%"
    )

    print()

    # --------------------------------------------------------
    # By action
    # --------------------------------------------------------

    print(
        "SKIPPED BY HISTORICAL ACTION"
    )

    for action, n in skipped_by_action.most_common():

        print(
            f"  {action:<20}"
            f"{n:>8,}"
            f"  {n / max(1, skipped) * 100:>7.2f}%"
        )

    # --------------------------------------------------------
    # By turn
    # --------------------------------------------------------

    print()
    print(
        "SKIPPED BY TURN"
    )

    for turn in sorted(
        skipped_by_turn
    ):

        print(
            f"  Turn {turn:<4}"
            f"{skipped_by_turn[turn]:>8,}"
        )

    # --------------------------------------------------------
    # By width
    # --------------------------------------------------------

    print()
    print(
        "SKIPPED BY QUOTE WIDTH"
    )

    for width in sorted(
        skipped_by_width
    ):

        print(
            f"  Width {width:<3}"
            f"{skipped_by_width[width]:>8,}"
        )

    # --------------------------------------------------------
    # By round
    # --------------------------------------------------------

    print()
    print(
        "SKIPPED BY ROUND"
    )

    for round_no in sorted(
        skipped_by_round
    ):

        print(
            f"  Round {round_no:<3}"
            f"{skipped_by_round[round_no]:>8,}"
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print()
    print(
        "SKIPPED EXAMPLES"
    )

    for i, example in enumerate(
        skipped_examples,
        start=1,
    ):

        print()
        print(
            f"Example {i}"
        )

        print(
            f"  seed:  {example['seed']}"
        )

        print(
            f"  round: {example['round']}"
        )

        print(
            f"  turn:  {example['turn']}"
        )

        print(
            f"  quote: {example['quote']}"
        )

        print(
            f"  historical: "
            f"{example['historical_action']}"
        )

        print(
            f"  normalized: "
            f"{example['historical_normalized']}"
        )

        print(
            "  generated:"
        )

        for action in example[
            "generated_actions"
        ]:

            print(
                f"    {action}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    output = (
        ROOT
        / "research"
        / "results"
        / "phase4d1_action_audit.json"
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
                "phase": "4D.1",

                "total":
                    total,

                "matched":
                    matched,

                "skipped":
                    skipped,

                "coverage":
                    matched / max(1, total),

                "skipped_by_action":
                    dict(
                        skipped_by_action
                    ),

                "skipped_by_turn":
                    dict(
                        skipped_by_turn
                    ),

                "skipped_by_width":
                    dict(
                        skipped_by_width
                    ),

                "skipped_by_round":
                    dict(
                        skipped_by_round
                    ),

                "examples":
                    skipped_examples,
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