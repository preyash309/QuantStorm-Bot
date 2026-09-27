"""Deterministic state abstraction for empirical opponent policies."""

from __future__ import annotations


def sign_bucket(x):
    x = int(x)
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def band(x, width):
    x = int(x)
    width = max(1, int(width))
    return x // width


def state_key(obs, *, method, quote=None, turn=None, offered=()):
    """Small discrete information-state key.

    We deliberately avoid hidden variables. Every component comes from Obs.
    The bins prevent the empirical table from becoming sparse.
    """
    k = int(obs["k_mine"])
    te = int(obs["te_mine"])
    opp_te = int(obs["te_theirs"])

    quote_width = -1
    quote_mid_offset = -1
    if quote is not None:
        bid, ask = map(int, quote)
        quote_width = ask - bid
        quote_mid_offset = (bid + ask) / 2.0 - k

    powers = tuple(sorted(obs["powers_mine"]))
    opp_powers = tuple(sorted(obs["powers_theirs"]))

    return (
        str(method),
        int(obs["round"]),
        bool(obs["is_maker"]),
        band(k, 4),
        band(te, 4),
        band(opp_te, 4),
        sign_bucket(k),
        band(int(obs["foresight_sum"]), 2),
        int(obs["foresight_n"] > 0),
        tuple(powers),
        tuple(opp_powers),
        band(quote_width, 2) if quote_width >= 0 else -1,
        band(round(quote_mid_offset), 2) if quote is not None else -1,
        int(turn) if turn is not None else -1,
        tuple(sorted(str(x) for x in offered)),
    )


def action_label(action):
    if isinstance(action, str):
        return action

    if isinstance(action, list) and action:
        if action[0] == "COUNTER":
            return "COUNTER"
        return str(action[0])

    return "UNKNOWN"
