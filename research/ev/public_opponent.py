"""
PHASE 4B — Public-information opponent response model.

IMPORTANT:
    Only information available to our bot is used.

We deliberately do NOT use:
    opponent k_mine
    opponent foresight
    opponent private information

The model is trained from the Phase 3 dataset but projects each
observation onto information that would actually be visible to us.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ACTIONS = (
    "ACCEPT_BUY",
    "ACCEPT_SELL",
    "COUNTER",
)


class PublicOpponentModel:
    """
    Empirical P(opponent response | public state).
    """

    def __init__(
        self,
        alpha: float = 1.0,
    ):
        self.alpha = float(alpha)

        self.tables = defaultdict(Counter)
        self.global_counts = Counter()

    # ========================================================
    # PUBLIC STATE
    # ========================================================

    @staticmethod
    def public_state(
        row,
        *,
        quote_override=None,
        turn_override=None,
    ):
        """
        Construct ONLY information that the opponent's
        counterparty can know.
        """

        obs = row["obs"]

        method = str(
            row["method"]
        )

        if method != "respond":
            raise ValueError(
                "PublicOpponentModel only models respond()."
            )

        quote = (
            quote_override
            if quote_override is not None
            else row["quote"]
        )

        bid = int(quote[0])
        ask = int(quote[1])

        turn = int(
            row["turn"]
            if turn_override is None
            else turn_override
        )

        # ----------------------------------------------------
        # Everything below is observable to the opponent.
        # ----------------------------------------------------

        round_no = int(
            obs["round"]
        )

        is_maker = bool(
            obs["is_maker"]
        )

        te_mine = int(
            obs["te_mine"]
        )

        te_theirs = int(
            obs["te_theirs"]
        )

        # Our opponent's powers are observable because powers
        # won in earlier auctions are public.
        #
        # From this observation's perspective:
        #
        # powers_mine = opponent's own powers
        # powers_theirs = our powers
        #
        # Therefore both sets are public.
        powers_mine = tuple(
            sorted(
                str(x)
                for x in obs[
                    "powers_mine"
                ]
            )
        )

        powers_theirs = tuple(
            sorted(
                str(x)
                for x in obs[
                    "powers_theirs"
                ]
            )
        )

        width = ask - bid

        midpoint = (
            bid + ask
        ) / 2.0

        return (
            round_no,
            is_maker,

            te_mine,
            te_theirs,

            powers_mine,
            powers_theirs,

            width,

            round(
                midpoint
            ),

            turn,
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        rows,
    ):

        for row in rows:

            if row["method"] != "respond":
                continue

            state = self.public_state(
                row
            )

            action = self.action_label(
                row
            )

            if action not in ACTIONS:
                continue

            self.tables[state][
                action
            ] += 1

            self.global_counts[
                action
            ] += 1

        return self

    # ========================================================
    # ACTION LABEL
    # ========================================================

    @staticmethod
    def action_label(row):

        action = row["action"]

        if isinstance(
            action,
            list,
        ):
            if action:
                return str(
                    action[0]
                )

        return str(action)

    # ========================================================
    # PROBABILITIES
    # ========================================================

    def probabilities_from_counts(
        self,
        counts,
    ):

        total = sum(
            counts.values()
        )

        denominator = (
            total
            + self.alpha
            * len(ACTIONS)
        )

        return {
            action:
                (
                    counts.get(
                        action,
                        0,
                    )
                    + self.alpha
                )
                / denominator
            for action in ACTIONS
        }

    # ========================================================
    # P(RESPONSE | PUBLIC STATE)
    # ========================================================

    def predict(
        self,
        *,
        round_no: int,
        is_maker: bool,
        te_mine: int,
        te_theirs: int,
        powers_mine,
        powers_theirs,
        quote,
        turn: int,
    ):

        bid = int(
            quote[0]
        )

        ask = int(
            quote[1]
        )

        state = (
            int(round_no),
            bool(is_maker),

            int(te_mine),
            int(te_theirs),

            tuple(
                sorted(
                    str(x)
                    for x in powers_mine
                )
            ),

            tuple(
                sorted(
                    str(x)
                    for x in powers_theirs
                )
            ),

            ask - bid,

            round(
                (bid + ask) / 2.0
            ),

            int(turn),
        )

        counts = self.tables.get(
            state
        )

        if counts is None:
            counts = self.global_counts

        return self.probabilities_from_counts(
            counts
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path: Path,
    ):

        payload = {
            "phase": "4B",

            "representation":
                "public_response_state",

            "alpha":
                self.alpha,

            "actions":
                list(ACTIONS),

            "global_counts":
                dict(
                    self.global_counts
                ),

            "states": {
                json.dumps(
                    state,
                    separators=(",", ":"),
                ): dict(counts)

                for state, counts
                in self.tables.items()
            },
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                payload,
                f,
                separators=(",", ":"),
            )