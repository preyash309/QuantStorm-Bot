from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AcceptBuy:
    pass

@dataclass(frozen=True)
class AcceptSell:
    pass

@dataclass(frozen=True)
class Counter:
    bid: int
    ask: int

@dataclass(frozen=True)
class BidAction:
    bids: tuple[tuple[str, int], ...]

    @staticmethod
    def from_dict(bids: dict[str, int]) -> "BidAction":
        return BidAction(tuple(sorted((str(k), int(v)) for k, v in bids.items())))

    def as_dict(self) -> dict[str, int]:
        return dict(self.bids)

@dataclass(frozen=True)
class TransformAction:
    fire: bool
