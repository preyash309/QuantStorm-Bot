"""Empirical opponent policy.

This is not ML. It is a smoothed conditional-frequency model:

    P(action | information-state)

It is intentionally transparent and fast enough to inspect and later embed
as constants if a useful policy emerges.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .features import state_key, action_label


class EmpiricalPolicy:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.actions = set()

    def fit_rows(self, rows):
        for row in rows:
            if row["method"] == "respond":
                quote = row["quote"]
                turn = row["turn"]
            else:
                quote = None
                turn = None

            obs = row["obs"]
            key = state_key(
                obs,
                method=row["method"],
                quote=quote,
                turn=turn,
                offered=row.get("offered", ()),
            )

            label = action_label(row["action"])
            self.counts[key][label] += 1
            self.actions.add(label)

        return self

    def probabilities(self, key):
        labels = sorted(self.actions)
        if not labels:
            return {}

        total = sum(self.counts[key].get(a, 0) for a in labels)
        denom = total + self.alpha * len(labels)

        return {
            a: (self.counts[key].get(a, 0) + self.alpha) / denom
            for a in labels
        }

    def top_action(self, key):
        probs = self.probabilities(key)
        if not probs:
            return None
        return max(probs.items(), key=lambda x: x[1])[0]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serial = []
        for key, counts in self.counts.items():
            serial.append({
                "key": list(key),
                "counts": dict(counts),
            })

        payload = {
            "alpha": self.alpha,
            "actions": sorted(self.actions),
            "states": serial,
        }

        path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(payload["alpha"])
        obj.actions = set(payload["actions"])

        for item in payload["states"]:
            obj.counts[tuple(item["key"])].update(item["counts"])

        return obj
