from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econlab.core.parameters import ModelParameters
from econlab.core.state import MacroState
from econlab.model.macro_model import MacroModel


@dataclass(slots=True)
class RewardWeights:
    inflation_gap: float = 1.0
    output_gap: float = 0.7
    financial_stress: float = 0.8
    rate_smoothing: float = 0.1


class CentralBankEnv(gym.Env):
    """
    Gymnasium environment for a central bank policy agent.

    Observation:
        [inflation, output_gap, unemployment, credit_growth, credit_spread,
         bank_leverage, default_rate, asset_price_gap, policy_rate]

    Action:
        Discrete choice of next policy rate on a fixed grid.

    Reward:
        Negative stabilization loss.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: ModelParameters | None = None,
        horizon: int = 120,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()

        self.params = params or ModelParameters()
        self.model = MacroModel(self.params)
        self.horizon = horizon
        self.reward_weights = reward_weights or RewardWeights()

        self.rate_grid = np.array(
            [-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(len(self.rate_grid))

        obs_low = np.array(
            [-0.20, -1.00, self.params.min_unemployment, -1.00, self.params.min_credit_spread,
             self.params.min_bank_leverage, self.params.min_default_rate,
             self.params.min_asset_gap, self.params.min_policy_rate],
            dtype=np.float32,
        )
        obs_high = np.array(
            [0.20, 1.00, self.params.max_unemployment, 1.00, self.params.max_credit_spread,
             self.params.max_bank_leverage, self.params.max_default_rate,
             self.params.max_asset_gap, self.params.max_policy_rate],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.rng = np.random.default_rng()
        self.state: MacroState | None = None
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        return self.state.as_array()

    def _financial_stress(self, state: MacroState) -> float:
        leverage_excess = max(state.bank_leverage - 10.0, 0.0) / 10.0
        return (
            state.credit_spread
            + state.default_rate
            + 0.5 * leverage_excess
        )

    def _reward(self, prev_state: MacroState, new_state: MacroState) -> float:
        w = self.reward_weights
        inflation_gap = new_state.inflation - self.params.inflation_target
        output_gap = new_state.output_gap
        financial_stress = self._financial_stress(new_state)
        rate_change = new_state.policy_rate - prev_state.policy_rate

        loss = (
            w.inflation_gap * inflation_gap**2
            + w.output_gap * output_gap**2
            + w.financial_stress * financial_stress**2
            + w.rate_smoothing * rate_change**2
        )
        return float(-loss)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.state = self.model.initial_state(self.rng)
        self.step_count = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_state = self.state
        next_rate = float(self.rate_grid[action])
        self.state = self.model.step(prev_state, next_rate, self.rng)
        self.step_count += 1

        reward = self._reward(prev_state, self.state)
        terminated = False
        truncated = self.step_count >= self.horizon

        info = {
            "policy_rate": self.state.policy_rate,
            "inflation": self.state.inflation,
            "output_gap": self.state.output_gap,
            "credit_spread": self.state.credit_spread,
            "default_rate": self.state.default_rate,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None