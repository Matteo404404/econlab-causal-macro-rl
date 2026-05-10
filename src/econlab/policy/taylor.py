from __future__ import annotations

from dataclasses import dataclass

from econlab.core.parameters import ModelParameters
from econlab.core.state import MacroState


@dataclass(slots=True)
class TaylorRule:
    """
    Simple Taylor-style monetary policy rule.

    This is our baseline policy against which RL will be compared.
    """

    inflation_weight: float = 1.5
    output_weight: float = 0.5

    def __call__(self, state: MacroState, params: ModelParameters) -> float:
        inflation_gap = state.inflation - params.inflation_target
        output_gap = state.output_gap

        rate = (
            params.neutral_rate
            + self.inflation_weight * inflation_gap
            + self.output_weight * output_gap
        )

        return float(
            min(max(rate, params.min_policy_rate), params.max_policy_rate)
        )