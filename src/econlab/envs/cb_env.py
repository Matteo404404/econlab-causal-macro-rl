from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econlab.core.parameters import ModelParameters
from econlab.core.state import MacroState
from econlab.model.macro_model import MacroModel
from econlab.model.shocks import ShockSchedule


@dataclass(slots=True)
class RewardWeights:
    inflation_gap: float = 2.0
    output_gap: float = 1.0
    financial_stress: float = 0.3
    rate_smoothing: float = 0.05


class CentralBankEnv(gym.Env):
    """
    Central bank environment with CONTINUOUS action space.
    Action: target policy rate in [min_rate, max_rate].
    Compatible with SAC, TD3 (off-policy continuous control).
    Also supports discrete mode for PPO/DQN via use_discrete=True.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: ModelParameters | None = None,
        horizon: int = 120,
        reward_weights: RewardWeights | None = None,
        shock_mode: str = "stochastic",
        use_discrete: bool = False,
    ) -> None:
        super().__init__()

        self.params = params or ModelParameters()
        self.model = MacroModel(self.params)
        self.horizon = horizon
        self.reward_weights = reward_weights or RewardWeights()
        self.shock_mode = shock_mode
        self.use_discrete = use_discrete

        # Discrete grid kept for PPO/DQN compatibility
        self.rate_grid = np.array(
            [-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
            dtype=np.float32,
        )

        if use_discrete:
            self.action_space = spaces.Discrete(len(self.rate_grid))
        else:
            # Continuous: action is the target policy rate directly
            self.action_space = spaces.Box(
                low=np.array([self.params.min_policy_rate], dtype=np.float32),
                high=np.array([self.params.max_policy_rate], dtype=np.float32),
                dtype=np.float32,
            )

        obs_low = np.array(
            [
                -0.20, -1.00,
                self.params.min_unemployment,
                -1.00,
                self.params.min_credit_spread,
                self.params.min_bank_leverage,
                self.params.min_default_rate,
                self.params.min_asset_gap,
                self.params.min_policy_rate,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                0.30, 1.00,
                self.params.max_unemployment,
                1.00,
                self.params.max_credit_spread,
                self.params.max_bank_leverage,
                self.params.max_default_rate,
                self.params.max_asset_gap,
                self.params.max_policy_rate,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.rng = np.random.default_rng()
        self.shock_schedule: ShockSchedule | None = None
        self.state: MacroState | None = None
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment not initialized.")
        obs = self.state.as_array()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _financial_stress(self, state: MacroState) -> float:
        leverage_excess = max(state.bank_leverage - 10.0, 0.0) / 10.0
        return state.credit_spread + state.default_rate + 0.5 * leverage_excess

    def _reward(self, prev_state: MacroState, new_state: MacroState) -> float:
        w = self.reward_weights
        inflation_gap = new_state.inflation - self.params.inflation_target
        output_gap = new_state.output_gap
        financial_stress = self._financial_stress(new_state)
        rate_change = new_state.policy_rate - prev_state.policy_rate

        raw_loss = (
            w.inflation_gap * inflation_gap**2
            + w.output_gap * output_gap**2
            + w.financial_stress * financial_stress**2
            + w.rate_smoothing * rate_change**2
        )
        return float(np.clip(-raw_loss / 0.01, -10.0, 0.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.shock_schedule = ShockSchedule(
            mode=self.shock_mode,
            rng=np.random.default_rng(seed),
        )
        self.state = self.model.initial_state(self.rng)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if self.use_discrete:
            next_rate = float(self.rate_grid[int(action)])
        else:
            next_rate = float(np.clip(action[0],
                                      self.params.min_policy_rate,
                                      self.params.max_policy_rate))

        prev_state = self.state
        shock = self.shock_schedule.get(self.step_count) if self.shock_schedule else None
        self.state = self.model.step(prev_state, next_rate, self.rng, shock=shock)
        self.step_count += 1

        reward = self._reward(prev_state, self.state)
        truncated = self.step_count >= self.horizon

        info = {
            "policy_rate": self.state.policy_rate,
            "inflation": self.state.inflation,
            "output_gap": self.state.output_gap,
            "credit_spread": self.state.credit_spread,
            "default_rate": self.state.default_rate,
            "bank_leverage": self.state.bank_leverage,
        }

        return self._get_obs(), reward, False, truncated, info

    def render(self):
        return None
