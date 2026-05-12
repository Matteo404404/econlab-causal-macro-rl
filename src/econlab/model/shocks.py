from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class ShockType(Enum):
    NONE = auto()
    PRODUCTIVITY = auto()       # positive TFP shock
    DEMAND = auto()             # positive demand shock
    FINANCIAL_STRESS = auto()   # sudden spread/leverage spike
    CREDIT_CRUNCH = auto()      # sharp credit contraction
    STAGFLATION = auto()        # inflation up, output down


@dataclass(slots=True)
class ShockRealization:
    """
    A single-period structured shock that is added on top of the
    baseline white-noise innovations in MacroModel.step().
    """

    shock_type: ShockType
    inflation_add: float = 0.0
    output_add: float = 0.0
    credit_add: float = 0.0
    spread_add: float = 0.0
    leverage_add: float = 0.0
    default_add: float = 0.0
    asset_add: float = 0.0


NO_SHOCK = ShockRealization(ShockType.NONE)


class ShockSchedule:
    """
    Generates a sequence of shocks over an episode.

    Two modes:
      - 'none':       no structured shocks (baseline white noise only)
      - 'stochastic': random financial/macro shocks with given probabilities
      - 'scenario':   deterministic shock at a given period (for counterfactuals)
    """

    def __init__(
        self,
        mode: str = "stochastic",
        financial_crisis_prob: float = 0.015,
        credit_crunch_prob: float = 0.010,
        stagflation_prob: float = 0.008,
        demand_shock_prob: float = 0.020,
        productivity_shock_prob: float = 0.020,
        scenario_shocks: dict[int, ShockRealization] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        assert mode in ("none", "stochastic", "scenario")
        self.mode = mode
        self.financial_crisis_prob = financial_crisis_prob
        self.credit_crunch_prob = credit_crunch_prob
        self.stagflation_prob = stagflation_prob
        self.demand_shock_prob = demand_shock_prob
        self.productivity_shock_prob = productivity_shock_prob
        self.scenario_shocks = scenario_shocks or {}
        self.rng = rng or np.random.default_rng()

    def get(self, period: int) -> ShockRealization:
        if self.mode == "none":
            return NO_SHOCK

        if self.mode == "scenario":
            return self.scenario_shocks.get(period, NO_SHOCK)

        # Stochastic mode
        draw = self.rng.random()
        cumulative = 0.0

        cumulative += self.financial_crisis_prob
        if draw < cumulative:
            # Sudden spike in spreads, defaults, leverage
            return ShockRealization(
                ShockType.FINANCIAL_STRESS,
                spread_add=float(self.rng.uniform(0.04, 0.10)),
                default_add=float(self.rng.uniform(0.04, 0.10)),
                leverage_add=float(self.rng.uniform(2.0, 6.0)),
                output_add=float(self.rng.uniform(-0.03, -0.01)),
            )

        cumulative += self.credit_crunch_prob
        if draw < cumulative:
            return ShockRealization(
                ShockType.CREDIT_CRUNCH,
                credit_add=float(self.rng.uniform(-0.05, -0.02)),
                spread_add=float(self.rng.uniform(0.02, 0.05)),
                output_add=float(self.rng.uniform(-0.02, -0.005)),
            )

        cumulative += self.stagflation_prob
        if draw < cumulative:
            return ShockRealization(
                ShockType.STAGFLATION,
                inflation_add=float(self.rng.uniform(0.005, 0.015)),
                output_add=float(self.rng.uniform(-0.02, -0.005)),
            )

        cumulative += self.demand_shock_prob
        if draw < cumulative:
            sign = 1.0 if self.rng.random() > 0.5 else -1.0
            return ShockRealization(
                ShockType.DEMAND,
                output_add=float(sign * self.rng.uniform(0.005, 0.02)),
                inflation_add=float(sign * self.rng.uniform(0.002, 0.008)),
            )

        cumulative += self.productivity_shock_prob
        if draw < cumulative:
            sign = 1.0 if self.rng.random() > 0.5 else -1.0
            return ShockRealization(
                ShockType.PRODUCTIVITY,
                output_add=float(sign * self.rng.uniform(0.005, 0.025)),
                inflation_add=float(-sign * self.rng.uniform(0.001, 0.005)),
            )

        return NO_SHOCK