"""Phase 3 empirical opponent-policy collector.

Runs the official engine as the execution oracle and records ONLY the
information available to the observed bot plus the action it took.
No hidden score, private hand, or engine internals are recorded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from engine import play_deal, make_coins
from game_config import DEFAULT_CONFIG


class RecordingBot:
    def __init__(self, factory, label):
        self._factory = factory
        self._label = label
        self._inner = None
        self.events = []

    @property
    def name(self):
        return getattr(self._inner, "name", self._label)

    def reset(self, seat, config, seed):
        self._inner = self._factory()
        self._inner.reset(seat, config, seed)
        self.events.clear()

    @staticmethod
    def _obs(obs):
        return {
            "seat": int(obs.seat),
            "round": int(obs.round),
            "k_mine": int(obs.k_mine),
            "te_mine": int(obs.te_mine),
            "te_theirs": int(obs.te_theirs),
            "spread_cap": int(obs.spread_cap),
            "final_cap": int(obs.final_cap),
            "is_maker": bool(obs.is_maker),
            "powers_mine": sorted(str(x) for x in obs.powers_mine),
            "powers_theirs": sorted(str(x) for x in obs.powers_theirs),
            "foresight_sum": int(sum(obs.foresight)),
            "foresight_n": int(len(obs.foresight)),
            "n_unknown_both": int(obs.n_unknown_both),
            "n_turns": int(obs.n_turns),
            "auction_log": [
                {
                    "round": int(e["round"]),
                    "seat": int(e["seat"]),
                    "power": str(e["power"]),
                    "cost": int(e["cost"]),
                }
                for e in obs.auction_log
            ],
        }

    def _record(self, method, obs, action, **extra):
        row = {
            "method": method,
            "obs": self._obs(obs),
            "action": action,
        }
        row.update(extra)
        self.events.append(row)

    def bid(self, obs, offered):
        action = self._inner.bid(obs, offered)
        self._record(
            "bid",
            obs,
            {str(k): int(v) for k, v in (action or {}).items()},
            offered=sorted(str(x) for x in offered),
        )
        return action

    def quote(self, obs):
        action = self._inner.quote(obs)
        self._record(
            "quote",
            obs,
            [int(action[0]), int(action[1])],
        )
        return action

    def respond(self, obs, quote, turn):
        action = self._inner.respond(obs, quote, turn)
        if isinstance(action, tuple):
            action_json = [str(action[0]), int(action[1]), int(action[2])]
        else:
            action_json = str(action)

        self._record(
            "respond",
            obs,
            action_json,
            quote=[int(quote[0]), int(quote[1])],
            turn=int(turn),
        )
        return action

    def use_transform(self, obs):
        action = bool(self._inner.use_transform(obs))
        self._record("use_transform", obs, action)
        return action


def collect(
    bot_factories,
    *,
    opponent_pairs,
    n_deals=2000,
    seed=900000,
    mirror=True,
    output=None,
):
    """
    opponent_pairs:
        iterable of (label_a, factory_a, label_b, factory_b).

    Every deal uses the official engine. The resulting JSON is the empirical
    policy dataset for Phase 3.
    """
    config = DEFAULT_CONFIG
    rows = []
    match_summaries = []

    for pair_index, (label_a, factory_a, label_b, factory_b) in enumerate(
        opponent_pairs
    ):
        match_seed = seed + pair_index * 100000

        rng = random.Random(match_seed)

        for d in range(n_deals):
            coins = make_coins(rng, config)
            deal_seed = match_seed + d

            for swap, invert in (
                [(False, False), (True, True)] if mirror
                else [(False, False)]
            ):
                a = RecordingBot(factory_a, label_a)
                b = RecordingBot(factory_b, label_b)

                result, _, _ = play_deal(
                    a,
                    b,
                    coins,
                    config,
                    seed=deal_seed,
                    swap=swap,
                    invert_roles=invert,
                    verbose=False,
                    bot_a_name=label_a,
                    bot_b_name=label_b,
                )

                for seat, bot in enumerate((a, b)):
                    for event in bot.events:
                        row = dict(event)
                        row["bot"] = bot.name
                        row["seat"] = seat
                        row["opponent"] = label_b if seat == 0 else label_a
                        row["seed"] = int(deal_seed)
                        row["swap"] = bool(swap)
                        row["invert_roles"] = bool(invert)
                        row["deal_pnl"] = float(result.pnl[seat])
                        rows.append(row)

    payload = {
        "phase": 3,
        "description": "Empirical opponent-policy dataset from official engine",
        "n_rows": len(rows),
        "rows": rows,
    }

    if output is None:
        output = Path("research/results/phase3_policy_dataset.json")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )

    return output, len(rows)
