from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass

from game_config import DEFAULT_CONFIG
from .actions import AcceptBuy, AcceptSell, Counter
from .mechanics import (
    SimContract, apply_settlement, apply_transform,
    forced_fill, sanitize_counter, split_coins
)
from .power_state import ResearchPowerState
from .state import SimState

def derive_seed(*parts):
    key = "|".join(str(p) for p in parts).encode()
    h = hashlib.blake2b(key, digest_size=8)
    return int.from_bytes(h.digest(), "little")

@dataclass(frozen=True)
class RoundResult:
    round: int
    offered: tuple[str, ...]
    won: tuple[tuple[int, tuple[str, ...]], ...]
    contract: SimContract

@dataclass(frozen=True)
class DealResult:
    pnl: tuple[float, float]
    score: int
    contracts: tuple[SimContract, ...]
    te_left: tuple[int, int]
    rounds: tuple[RoundResult, ...]

class ExactSimulator:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def initial_state(self, *, coins, seed=0, swap=False, invert_roles=False):
        if len(coins) != self.config.N_COINS:
            raise ValueError(f"Expected {self.config.N_COINS} coins, got {len(coins)}")
        if any(c not in (-1, 1) for c in coins):
            raise ValueError("coins must contain only +/-1")

        hands = split_coins(coins, self.config, swap)
        slate = self.config.draw_slate(
            random.Random(derive_seed("deal-slate", seed))
        )

        return SimState(
            seed=seed,
            coins=tuple(coins),
            hands=[list(hands[0]), list(hands[1])],
            revealed=[[], []],
            score=sum(coins),
            round=0,
            turn=0,
            te=[self.config.TE_BUDGET, self.config.TE_BUDGET],
            maker_seat=0,
            powers=ResearchPowerState(),
            auction_log=[],
            contracts=[],
            slate=slate,
        )

    def reveal_round(self, state, r, invert_roles=False):
        if r != state.round + 1:
            raise ValueError(f"Expected round {state.round + 1}, got {r}")

        state.round = r
        state.turn = 0
        default_maker = 0 if r % 2 == 1 else 1
        state.maker_seat = 1-default_maker if invert_roles else default_maker
        state.powers.new_round(r)

        lo = (r-1) * self.config.REVEAL_PER_ROUND
        hi = lo + self.config.REVEAL_PER_ROUND
        for seat in (0, 1):
            state.revealed[seat].extend(state.hands[seat][lo:hi])

        state.foresight = [(), ()]

        return tuple(
            n for n in state.slate[r]
            if not state.powers.consumed(n, self.config)
        )

    def resolve_auction(self, state, offered, bids, *, tie_flip=0, rng=None):
        rng = rng or random.Random(derive_seed("deal-sym", state.seed))
        won = {0: set(), 1: set()}

        for name in offered:
            b0 = int(bids.get(0, {}).get(name, 0))
            b1 = int(bids.get(1, {}).get(name, 0))

            if b0 == b1:
                if b0 == 0 or self.config.TIE_RULE == "refund":
                    continue
                winner = rng.randrange(2) ^ int(tie_flip)
            else:
                winner = 0 if b0 > b1 else 1

            cost = int(bids[winner].get(name, 0))
            if cost > state.te[winner]:
                continue

            state.te[winner] -= cost
            won[winner].add(name)
            state.auction_log.append({
                "round": state.round,
                "seat": winner,
                "power": name,
                "cost": cost,
            })
            state.powers.acquire(winner, name, state.round, self.config)

        return won

    def apply_transform_if_won(self, state, won, fire):
        for seat in (0, 1):
            if "TRANSFORM" not in won[seat]:
                continue
            if bool(fire.get(seat, False)):
                apply_transform(state.hands, state.revealed)
            break

    def apply_foresight(self, state, won, *, rng=None):
        rng = rng or random.Random(derive_seed("deal-sym", state.seed))
        leaks = [(), ()]

        for seat in (0, 1):
            if "FORESIGHT" not in won[seat]:
                continue
            k = int(self.config.POWERS["FORESIGHT"]["magnitude"])
            opp = state.revealed[1-seat]
            if opp and k > 0:
                n = min(k, len(opp))
                idx = rng.sample(range(len(opp)), n)
                leaks[seat] = tuple(opp[i] for i in idx)

        state.foresight = leaks
        return leaks

    def negotiate(self, state, *, opening_quote, responses, rng=None):
        rng = rng or random.Random(
            derive_seed("deal", state.seed, False, False)
        )

        maker = state.maker_seat
        speaker = 1-maker
        last_quoter = maker

        bid, ask = map(int, opening_quote)
        state.turn = 1
        state.current_quote = (bid, ask)

        opening = {
            "maker_seat": maker,
            "open_bid": bid,
            "open_ask": ask,
        }

        if len(responses) > self.config.N_TURNS-1:
            raise ValueError("Too many responses")

        for i, action in enumerate(responses):
            state.turn = i+2

            if isinstance(action, AcceptBuy):
                c = SimContract(state.round, ask, speaker, False, maker_seat=maker,
                                open_bid=bid if i == 0 else opening["open_bid"],
                                open_ask=ask if i == 0 else opening["open_ask"])
                state.contracts.append(c)
                return c

            if isinstance(action, AcceptSell):
                c = SimContract(state.round, bid, 1-speaker, False, maker_seat=maker,
                                open_bid=opening["open_bid"], open_ask=opening["open_ask"])
                state.contracts.append(c)
                return c

            if not isinstance(action, Counter):
                raise TypeError(f"Unknown action: {action!r}")

            bid, ask = sanitize_counter(
                bid, ask, action.bid, action.ask,
                self.config.final_cap(state.round),
                self.config.MIN_REDUCTION,
            )
            state.current_quote = (bid, ask)
            last_quoter = speaker
            speaker = 1-speaker

        c0 = forced_fill(
            bid=bid, ask=ask, last_quoter=last_quoter, maker=maker,
            powers=(
                set(state.powers.active(0, state.round)),
                set(state.powers.active(1, state.round)),
            ),
            config=self.config, rng=rng,
        )
        c = SimContract(
            state.round, c0.price, c0.long_seat, True,
            c0.forcer, c0.shift, maker,
            opening["open_bid"], opening["open_ask"]
        )
        state.contracts.append(c)
        return c

    def settle(self, state):
        state.terminal = True
        return DealResult(
            pnl=apply_settlement(
                state.contracts, state.score, state.powers,
                state.te, self.config
            ),
            score=state.score,
            contracts=tuple(state.contracts),
            te_left=tuple(state.te),
            rounds=(),
        )

    def run_explicit_deal(
        self, *, coins, round_actions, seed=0,
        swap=False, invert_roles=False
    ):
        state = self.initial_state(
            coins=coins, seed=seed, swap=swap,
            invert_roles=invert_roles
        )

        sym = random.Random(derive_seed("deal-sym", seed))
        rng = random.Random(derive_seed("deal", seed, swap, invert_roles))
        rounds = []

        for r, spec in enumerate(round_actions, 1):
            offered = self.reveal_round(state, r, invert_roles)

            won = self.resolve_auction(
                state, offered, spec.get("bids", {}),
                tie_flip=int(swap), rng=sym
            )

            self.apply_transform_if_won(
                state, won, spec.get("transform", {})
            )
            self.apply_foresight(state, won, rng=sym)

            contract = self.negotiate(
                state,
                opening_quote=spec["opening_quote"],
                responses=spec.get("responses", []),
                rng=rng,
            )

            rounds.append(RoundResult(
                r, tuple(offered),
                tuple((s, tuple(sorted(v))) for s, v in won.items()),
                contract
            ))

        result = self.settle(state)
        return DealResult(
            result.pnl, result.score, result.contracts,
            result.te_left, tuple(rounds)
        )
