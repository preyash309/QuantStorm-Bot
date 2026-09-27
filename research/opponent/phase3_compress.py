from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MIN_VISITS = 20
DEFAULT_MIN_CONFIDENCE = 0.60


def compress(
    analysis_path: Path,
    output_path: Path,
    min_visits: int,
    min_confidence: float,
):
    with analysis_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    source_states = analysis["states"]

    retained = []
    rejected_low_count = 0
    rejected_low_confidence = 0

    for state in source_states:
        total = int(state["total"])
        confidence = float(state["confidence"])

        if total < min_visits:
            rejected_low_count += 1
            continue

        if confidence < min_confidence:
            rejected_low_confidence += 1
            continue

        counts = state["counts"]

        probabilities = {
            action: round(
                count / total,
                6
            )
            for action, count in counts.items()
        }

        retained.append(
            {
                "key": state["key"],
                "n": total,
                "action": state["dominant_action"],
                "confidence": confidence,
                "p": probabilities,
            }
        )

    # Highest frequency states first.
    retained.sort(
        key=lambda x: (
            x["n"],
            x["confidence"],
        ),
        reverse=True,
    )

    # Aggregate summary.
    action_counts = {}

    for state in retained:
        action = state["action"]
        action_counts[action] = (
            action_counts.get(action, 0) + state["n"]
        )

    output = {
        "phase": 3,
        "policy_type": "empirical_opponent_policy",
        "source": str(analysis_path),
        "filters": {
            "min_visits": min_visits,
            "min_confidence": min_confidence,
        },
        "source_state_count": len(source_states),
        "retained_state_count": len(retained),
        "rejected_low_count": rejected_low_count,
        "rejected_low_confidence": rejected_low_confidence,
        "retained_observations": sum(
            x["n"] for x in retained
        ),
        "dominant_action_observations": action_counts,
        "states": retained,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            separators=(",", ":"),
        )

    print("=" * 60)
    print("PHASE 3 POLICY COMPRESSION COMPLETE")
    print("=" * 60)
    print(f"Source states:          {len(source_states):,}")
    print(f"Retained states:        {len(retained):,}")
    print(f"Rejected low-count:     {rejected_low_count:,}")
    print(f"Rejected low-confidence:{rejected_low_confidence:,}")
    print(
        f"Retained observations:  "
        f"{sum(x['n'] for x in retained):,}"
    )

    print()
    print("FILTERS")
    print(f"  min visits:            {min_visits}")
    print(f"  min confidence:        {min_confidence:.2f}")

    print()
    print("RETAINED DOMINANT ACTIONS")
    for action, count in sorted(
        action_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {action:12s}: {count:,}")

    print()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/results/phase3_analysis.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/phase3_policy.json"),
    )

    parser.add_argument(
        "--min-visits",
        type=int,
        default=DEFAULT_MIN_VISITS,
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {args.input}"
        )

    if args.min_visits < 1:
        raise ValueError("--min-visits must be >= 1")

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError(
            "--min-confidence must be between 0 and 1"
        )

    compress(
        args.input,
        args.output,
        args.min_visits,
        args.min_confidence,
    )


if __name__ == "__main__":
    main()