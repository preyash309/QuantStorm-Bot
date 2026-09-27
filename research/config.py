"""Research-only configuration. Official game rules live elsewhere."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    baseline_seed: int = 900001
    baseline_deals: int = 100
    mirror: bool = True
