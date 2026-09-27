"""
PHASE 4C — Quote-conditioned opponent response model.

Research only.

Learns:

    P(response | public_state, quote, turn)

from the Phase 3 policy dataset.

This is deliberately empirical and lightweight.
No Monte Carlo is performed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict


ACTIONS = (
    "ACCEPT_BUY",
    "ACCEPT_SELL",
    "COUNTER",
)


class QuoteOpponentModel:

    def __init__(self):

        self.tables = defaultdict(Counter)

        self.global_counts = Counter()

        self.n_rows = 0

    # ========================================================
    # STATE KEY
    # ========================================================

    @staticmethod
    def state_key(row):

        obs = row["obs"]

        quote = row.get(
            "quote"
        )

        if quote is None:
            quote = (
                None,
                None,
            )
        else:
            quote = (
                int(quote[0]),
                int(quote[1]),
            )

        powers_mine = tuple(
            sorted(
                str(x)
                for x in obs.get(
                    "powers_mine",
                    [],
                )
            )
        )

        powers_theirs = tuple(
            sorted(
                str(x)
                for x in obs.get(
                    "powers_theirs",
                    [],
                )
            )
        )

        return (
            int(obs["round"]),
            int(obs["te_mine"]),
            int(obs["te_theirs"]),
            bool(obs["is_maker"]),
            powers_mine,
            powers_theirs,
            int(row.get("turn", 0)),
            quote,
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        rows,
    ):

        self.tables.clear()
        self.global_counts.clear()

        self.n_rows = 0

        for row in rows:

            if row.get("method") != "respond":
                continue

            action = row.get(
                "action"
            )

            # Serialized actions may be lists.
            if isinstance(
                action,
                list,
            ):
                action_name = str(
                    action[0]
                )
            else:
                action_name = str(
                    action
                )

            if action_name not in ACTIONS:
                continue

            key = self.state_key(
                row
            )

            self.tables[key][
                action_name
            ] += 1

            self.global_counts[
                action_name
            ] += 1

            self.n_rows += 1

        return self

    # ========================================================
    # PROBABILITY
    # ========================================================

    def predict_from_row(
        self,
        row,
        smoothing=1.0,
    ):

        key = self.state_key(
            row
        )

        return self._probabilities(
            self.tables.get(
                key,
                Counter(),
            ),
            smoothing,
        )

    def predict(
        self,
        *,
        round_no,
        te_mine,
        te_theirs,
        is_maker,
        powers_mine,
        powers_theirs,
        turn,
        quote,
        smoothing=1.0,
    ):

        fake_row = {
            "method": "respond",
            "turn": turn,
            "quote": list(
                quote
            ),
            "obs": {
                "round": round_no,
                "te_mine": te_mine,
                "te_theirs": te_theirs,
                "is_maker": is_maker,
                "powers_mine": list(
                    powers_mine
                ),
                "powers_theirs": list(
                    powers_theirs
                ),
            },
        }

        return self.predict_from_row(
            fake_row,
            smoothing=smoothing,
        )

    # ========================================================
    # INTERNAL PROBABILITIES
    # ========================================================

    def _probabilities(
        self,
        counts,
        smoothing,
    ):

        if not counts:

            total = sum(
                self.global_counts.values()
            )

            if total == 0:

                return {
                    action: 1.0 / len(ACTIONS)
                    for action in ACTIONS
                }

            return {
                action:
                    (
                        self.global_counts[action]
                        + smoothing
                    )
                    /
                    (
                        total
                        + smoothing * len(ACTIONS)
                    )
                for action in ACTIONS
            }

        total = sum(
            counts.values()
        )

        denominator = (
            total
            + smoothing * len(ACTIONS)
        )

        return {
            action:
                (
                    counts[action]
                    + smoothing
                )
                / denominator
            for action in ACTIONS
        }

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    @property
    def n_states(self):

        return len(
            self.tables
        )