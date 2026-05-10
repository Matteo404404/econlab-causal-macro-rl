from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelParameters:
    """
    Parameters for the first stylized macro-finance transition model.
    """

    inflation_target: float = 0.02
    natural_unemployment: float = 0.05
    neutral_rate: float = 0.02

    # Phillips / demand style dynamics
    inflation_persistence: float = 0.65
    output_gap_persistence: float = 0.70
    unemployment_persistence: float = 0.80

    # Financial block persistence
    credit_growth_persistence: float = 0.60
    spread_persistence: float = 0.70
    leverage_persistence: float = 0.75
    default_persistence: float = 0.60
    asset_gap_persistence: float = 0.72

    # Policy transmission
    policy_to_output: float = 0.12
    policy_to_inflation: float = 0.05
    policy_to_credit: float = 0.10
    policy_to_asset_prices: float = 0.09
    policy_to_spread: float = 0.06

    # Macro-financial spillovers
    output_to_inflation: float = 0.18
    output_to_unemployment: float = -0.10
    credit_to_output: float = 0.10
    spread_to_output: float = -0.12
    leverage_to_default: float = 0.04
    default_to_spread: float = 0.30
    asset_to_credit: float = 0.10

    # Shock scales
    sigma_inflation: float = 0.002
    sigma_output: float = 0.004
    sigma_unemployment: float = 0.002
    sigma_credit: float = 0.005
    sigma_spread: float = 0.003
    sigma_leverage: float = 0.010
    sigma_default: float = 0.002
    sigma_asset: float = 0.006

    # Bounds to keep the first environment well-behaved
    min_policy_rate: float = -0.01
    max_policy_rate: float = 0.10
    min_unemployment: float = 0.01
    max_unemployment: float = 0.25
    min_default_rate: float = 0.0
    max_default_rate: float = 0.20
    min_bank_leverage: float = 1.0
    max_bank_leverage: float = 25.0
    min_credit_spread: float = 0.0
    max_credit_spread: float = 0.20
    min_asset_gap: float = -0.50
    max_asset_gap: float = 0.50