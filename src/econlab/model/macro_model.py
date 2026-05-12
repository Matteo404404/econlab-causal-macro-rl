from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from econlab.core.parameters import ModelParameters
from econlab.core.state import MacroState
from econlab.model.shocks import ShockRealization


@dataclass(slots=True)
class MacroModel:
    """
    Stylized macro-finance transition model with structured shock support.
    """

    params: ModelParameters

    def initial_state(self, rng: np.random.Generator) -> MacroState:
        p = self.params
        return MacroState(
            inflation=float(rng.normal(p.inflation_target, 0.002)),
            output_gap=float(rng.normal(0.0, 0.01)),
            unemployment=float(rng.normal(p.natural_unemployment, 0.003)),
            credit_growth=float(rng.normal(0.02, 0.01)),
            credit_spread=float(rng.normal(0.015, 0.002)),
            bank_leverage=float(rng.normal(10.0, 0.5)),
            default_rate=float(rng.normal(0.02, 0.003)),
            asset_price_gap=float(rng.normal(0.0, 0.02)),
            policy_rate=float(rng.normal(p.neutral_rate, 0.002)),
        )

    def step(
        self,
        state: MacroState,
        policy_rate_next: float,
        rng: np.random.Generator,
        shock: ShockRealization | None = None,
    ) -> MacroState:
        """
        Advance the model by one period.
        Optional structured shock is added on top of white-noise innovations.
        """
        from econlab.model.shocks import NO_SHOCK
        shock = shock or NO_SHOCK

        p = self.params
        policy_rate_next = float(
            np.clip(policy_rate_next, p.min_policy_rate, p.max_policy_rate)
        )
        policy_gap = policy_rate_next - p.neutral_rate

        # White noise innovations
        inflation_shock = rng.normal(0.0, p.sigma_inflation) + shock.inflation_add
        output_shock = rng.normal(0.0, p.sigma_output) + shock.output_add
        unemployment_shock = rng.normal(0.0, p.sigma_unemployment)
        credit_shock = rng.normal(0.0, p.sigma_credit) + shock.credit_add
        spread_shock = rng.normal(0.0, p.sigma_spread) + shock.spread_add
        leverage_shock = rng.normal(0.0, p.sigma_leverage) + shock.leverage_add
        default_shock = rng.normal(0.0, p.sigma_default) + shock.default_add
        asset_shock = rng.normal(0.0, p.sigma_asset) + shock.asset_add

        output_gap_next = (
            p.output_gap_persistence * state.output_gap
            - p.policy_to_output * policy_gap
            + p.credit_to_output * (state.credit_growth - 0.02)
            + p.spread_to_output * (state.credit_spread - 0.015)
            + output_shock
        )

        inflation_next = (
            p.inflation_persistence * state.inflation
            + (1.0 - p.inflation_persistence) * p.inflation_target
            + p.output_to_inflation * output_gap_next
            - p.policy_to_inflation * policy_gap
            + inflation_shock
        )

        unemployment_next = (
            p.unemployment_persistence * state.unemployment
            + (1.0 - p.unemployment_persistence) * p.natural_unemployment
            - p.output_to_unemployment * output_gap_next
            + unemployment_shock
        )

        asset_price_gap_next = (
            p.asset_gap_persistence * state.asset_price_gap
            - p.policy_to_asset_prices * policy_gap
            + 0.06 * state.output_gap
            + asset_shock
        )

        credit_growth_next = (
            p.credit_growth_persistence * state.credit_growth
            + 0.4 * 0.02
            - p.policy_to_credit * policy_gap
            - 0.08 * state.credit_spread
            + p.asset_to_credit * asset_price_gap_next
            + credit_shock
        )

        default_rate_next = (
            p.default_persistence * state.default_rate
            + 0.4 * 0.02
            + p.leverage_to_default * max(state.bank_leverage - 10.0, 0.0) / 10.0
            - 0.03 * output_gap_next
            + default_shock
        )

        credit_spread_next = (
            p.spread_persistence * state.credit_spread
            + 0.3 * 0.015
            + p.default_to_spread * default_rate_next
            + p.policy_to_spread * policy_gap
            + spread_shock
        )

        bank_leverage_next = (
            p.leverage_persistence * state.bank_leverage
            + 0.25 * 10.0
            + 8.0 * max(credit_growth_next - 0.02, 0.0)
            - 10.0 * default_rate_next
            + leverage_shock
        )

        return MacroState(
            inflation=float(np.clip(inflation_next, -0.10, 0.30)),
            output_gap=float(np.clip(output_gap_next, -1.0, 1.0)),
            unemployment=float(
                np.clip(unemployment_next, p.min_unemployment, p.max_unemployment)
            ),
            credit_growth=float(np.clip(credit_growth_next, -0.20, 0.30)),
            credit_spread=float(
                np.clip(credit_spread_next, p.min_credit_spread, p.max_credit_spread)
            ),
            bank_leverage=float(
                np.clip(bank_leverage_next, p.min_bank_leverage, p.max_bank_leverage)
            ),
            default_rate=float(
                np.clip(default_rate_next, p.min_default_rate, p.max_default_rate)
            ),
            asset_price_gap=float(
                np.clip(asset_price_gap_next, p.min_asset_gap, p.max_asset_gap)
            ),
            policy_rate=policy_rate_next,
        )