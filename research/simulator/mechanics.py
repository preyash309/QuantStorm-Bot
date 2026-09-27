from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class SimContract:
    round: int
    price: int
    long_seat: int
    forced: bool
    forcer: int = -1
    shift: int = 0
    maker_seat: int = -1
    open_bid: int = 0
    open_ask: int = 0

def split_coins(coins, config, swap=False):
    a = list(coins[:config.N_PRIVATE])
    b = list(coins[config.N_PRIVATE:])
    return (b, a) if swap else (a, b)

def apply_transform(hands, revealed):
    hands[0][:], hands[1][:] = list(hands[1]), list(hands[0])
    n0, n1 = len(revealed[0]), len(revealed[1])
    revealed[0][:] = hands[0][:n0]
    revealed[1][:] = hands[1][:n1]

def fill_shift(short_seat, powers, config):
    shift = 0
    for seat in (0, 1):
        sign = 1 if seat == short_seat else -1
        for name in ("TRICK_ROOM", "STEALTH_ROCK"):
            if name in powers[seat] and name in config.POWERS:
                shift += sign * int(config.POWERS[name]["magnitude"])
    return shift

def sanitize_counter(bid, ask, new_bid, new_ask, final_cap, min_reduction):
    nb, na = int(new_bid), int(new_ask)

    if nb > na:
        nb, na = na, nb

    nb = max(bid, min(nb, ask))
    na = max(nb, min(na, ask))

    max_width = min(
        ask - bid,
        max(int(final_cap), (ask - bid) - int(min_reduction)),
    )

    if na - nb > max_width:
        mid = (nb + na) // 2
        nb = max(bid, mid - max_width // 2)
        na = min(ask, nb + max_width)
        nb = max(bid, na - max_width)

    return nb, na

def forced_fill(*, bid, ask, last_quoter, maker, powers, config, rng):
    if config.MIDPOINT_SIDE_RULE == "last_quoter_sells":
        short = last_quoter
    elif config.MIDPOINT_SIDE_RULE == "maker_sells":
        short = maker
    else:
        short = rng.randrange(2)

    shift = fill_shift(short, powers, config)
    price = (bid + ask) // 2 + shift

    return SimContract(
        round=-1,
        price=price,
        long_seat=1-short,
        forced=True,
        forcer=last_quoter,
        shift=shift,
        maker_seat=maker,
        open_bid=bid,
        open_ask=ask,
    )

def maker_obligation_amount(contract, score, config):
    width = contract.open_ask - contract.open_bid
    amount = 0.0

    if config.MAKER_OBLIGATION:
        p_w = config.straddle_prob(contract.round, width)
        straddle = contract.open_bid <= score <= contract.open_ask
        amount += (
            config.MAKER_OBLIGATION * (1.0 - p_w)
            if straddle
            else -config.MAKER_OBLIGATION * p_w
        )

    if config.WIDTH_PREMIUM:
        amount -= config.WIDTH_PREMIUM * max(
            0,
            width - config.open_width_floor(contract.round),
        )

    return amount

def apply_settlement(contracts, score, power_state, te_left, config):
    pnl = [0.0, 0.0]

    for c in contracts:
        raw = score - c.price
        pnl[c.long_seat] += raw
        pnl[1-c.long_seat] -= raw

    for c in contracts:
        if c.maker_seat >= 0:
            amount = maker_obligation_amount(c, score, config)
            pnl[c.maker_seat] += amount
            pnl[1-c.maker_seat] -= amount

        if c.forced and c.forcer >= 0:
            fee = config.FORCED_FILL_FEE
            pnl[c.forcer] -= fee
            pnl[1-c.forcer] += fee

    if "SUBSTITUTE" in config.POWERS:
        cap = float(config.POWERS["SUBSTITUTE"]["magnitude"])
        for c in contracts:
            for seat in (0, 1):
                if c.round in power_state.substitute_rounds[seat]:
                    raw = score - c.price if c.long_seat == seat else c.price - score
                    if raw < -cap:
                        refund = (-cap) - raw
                        pnl[seat] += refund
                        pnl[1-seat] -= refund

    te_diff = (te_left[0] - te_left[1]) * config.TE_SALVAGE
    pnl[0] += te_diff
    pnl[1] -= te_diff

    if abs(pnl[0] + pnl[1]) > config.ZERO_SUM_TOL:
        raise AssertionError("Research simulator lost zero-sum invariant")

    return tuple(pnl)
