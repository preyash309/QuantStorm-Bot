"""Summarise the empirical behaviour of each reference bot."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


def analyse(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["rows"]

    by_bot = defaultdict(lambda: defaultdict(Counter))

    for row in rows:
        bot = row["bot"]
        method = row["method"]

        if method == "respond":
            action = row["action"]
            label = (
                action[0]
                if isinstance(action, list)
                else str(action)
            )
            by_bot[bot][method][label] += 1

        elif method == "use_transform":
            by_bot[bot][method][str(bool(row["action"]))] += 1

        elif method == "bid":
            bids = row["action"]
            for power, amount in bids.items():
                by_bot[bot]["bid:" + power][str(amount)] += 1

        elif method == "quote":
            bid, ask = row["action"]
            k = row["obs"]["k_mine"]
            by_bot[bot]["quote_width"][str(ask-bid)] += 1
            by_bot[bot]["quote_offset"][str((bid+ask)/2-k)] += 1

    for bot, methods in sorted(by_bot.items()):
        print("\n" + "=" * 68)
        print(f"BOT: {bot}")
        print("=" * 68)

        for method, counts in methods.items():
            total = sum(counts.values())
            print(f"\n{method}  n={total}")
            for action, count in counts.most_common(12):
                print(f"  {action:20s} {count:8d}  {count/total:7.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        default="research/results/phase3_policy_dataset.json",
        nargs="?",
    )
    args = parser.parse_args()
    analyse(args.dataset)
