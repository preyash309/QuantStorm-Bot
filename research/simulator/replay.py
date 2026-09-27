from __future__ import annotations
from .actions import AcceptBuy, AcceptSell, Counter

def _action_from_trace(action):
    if action == "ACCEPT_BUY":
        return AcceptBuy()
    if action == "ACCEPT_SELL":
        return AcceptSell()
    if isinstance(action, (tuple, list)) and len(action) == 3:
        if action[0] != "COUNTER":
            raise ValueError(action)
        return Counter(int(action[1]), int(action[2]))
    raise ValueError(f"Unknown trace action: {action!r}")

def trace_to_round_actions(trace_a, trace_b, n_rounds=5):
    events = {0: list(trace_a.events), 1: list(trace_b.events)}
    result = []

    for r in range(1, n_rounds+1):
        bids = {0: {}, 1: {}}
        transform = {0: False, 1: False}
        opening = None
        responses = {}

        for seat in (0, 1):
            for e in events[seat]:
                if e.round != r:
                    continue
                if e.method == "bid":
                    if hasattr(e.action, "items"):
                        bids[seat].update(
                            {str(k): int(v) for k, v in e.action.items()}
                        )
                elif e.method == "use_transform":
                    transform[seat] = bool(e.action)
                elif e.method == "quote":
                    if e.state.is_maker:
                        opening = tuple(map(int, e.action))
                elif e.method == "respond":
                    responses[int(e.turn)] = _action_from_trace(e.action)

        if opening is None:
            raise AssertionError(f"Missing Maker quote in round {r}")

        result.append({
            "bids": bids,
            "transform": transform,
            "opening_quote": opening,
            "responses": [responses[t] for t in sorted(responses)],
        })

    return result
