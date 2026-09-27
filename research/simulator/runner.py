"""Phase-1 research runner.

The official engine remains the source of truth. This module adds tracing and
serialization around it; it does not reimplement game rules yet.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from engine import play_deal, play_match
from game_config import DEFAULT_CONFIG

from research.simulator.trace_bot import TraceBot


def _make_trace_instance(bot_factory, label):
    return TraceBot(bot_factory, label)


def run_traced_deal(
    bot_a_factory,
    bot_b_factory,
    *,
    seed: int,
    coins: list[int],
    swap: bool = False,
    invert_roles: bool = False,
    verbose: bool = False,
    config=DEFAULT_CONFIG,
):
    """Run exactly one official deal and return result + both traces."""

    trace_a = _make_trace_instance(bot_a_factory, "TraceA")
    trace_b = _make_trace_instance(bot_b_factory, "TraceB")

    result, wrap_a, wrap_b = play_deal(
        trace_a,
        trace_b,
        list(coins),
        config,
        seed=seed,
        swap=swap,
        invert_roles=invert_roles,
        verbose=verbose,
        bot_a_name="TraceA",
        bot_b_name="TraceB",
    )

    return result, trace_a, trace_b, wrap_a, wrap_b


def _contract_dict(c):
    return {
        "round": c.round,
        "price": c.price,
        "long_seat": c.long_seat,
        "forced": c.forced,
        "forcer": c.forcer,
        "shift": c.shift,
        "maker_seat": c.maker_seat,
        "open_bid": c.open_bid,
        "open_ask": c.open_ask,
    }


def result_to_dict(result):
    return {
        "pnl": list(result.pnl),
        "score": result.score,
        "contracts": [_contract_dict(c) for c in result.contracts],
        "te_left": list(result.te_left),
        "logs": list(result.logs),
    }


def event_to_dict(event):
    state = asdict(event.state)
    state["powers_mine"] = sorted(state["powers_mine"])
    state["powers_theirs"] = sorted(state["powers_theirs"])
    state["auction_log"] = [dict(x) for x in state["auction_log"]]
    state["contracts"] = [dict(x) for x in state["contracts"]]

    action = event.action
    if isinstance(action, tuple):
        action = list(action)

    return {
        "method": event.method,
        "round": event.round,
        "turn": event.turn,
        "state": state,
        "offered": list(event.offered),
        "quote_before": list(event.quote_before)
        if event.quote_before is not None else None,
        "action": action,
    }


def save_trace(path, result, trace_a, trace_b):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "result": result_to_dict(result),
        "trace_a": [event_to_dict(e) for e in trace_a.events],
        "trace_b": [event_to_dict(e) for e in trace_b.events],
    }

    path.write_text(
        json.dumps(payload, indent=2, default=list),
        encoding="utf-8",
    )
    return path


def run_baseline_match(
    bot_a_factory,
    bot_b_factory,
    *,
    seed: int,
    deals: int = 100,
    mirror: bool = True,
    config=DEFAULT_CONFIG,
):
    """Run a normal official-engine research baseline."""

    return play_match(
        bot_a_factory,
        bot_b_factory,
        config=config,
        seed=seed,
        mirror=mirror,
        n_deals=deals,
        verbose=False,
        bot_a_name="ResearchA",
        bot_b_name="ResearchB",
    )
