from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MacroState:
    """
    Compact macro-financial state for the first simulator version.

    All variables are expressed in simple annualized or normalized units.
    This is intentionally stylized rather than fully structural.
    """

    inflation: float
    output_gap: float
    unemployment: float
    credit_growth: float
    credit_spread: float
    bank_leverage: float
    default_rate: float
    asset_price_gap: float
    policy_rate: float

    def as_array(self) -> np.ndarray:
        """Return the state as a float32 observation vector."""
        return np.array(
            [
                self.inflation,
                self.output_gap,
                self.unemployment,
                self.credit_growth,
                self.credit_spread,
                self.bank_leverage,
                self.default_rate,
                self.asset_price_gap,
                self.policy_rate,
            ],
            dtype=np.float32,
        )