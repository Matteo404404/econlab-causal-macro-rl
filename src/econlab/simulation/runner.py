from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from econlab.core.parameters import ModelParameters
from econlab.core.state import MacroState
from econlab.envs.cb_env import CentralBankEnv
from econlab.policy.taylor import TaylorRule


@dataclass
class Trajectory:
    """
    Stores the full time series produced by one episode.
    Each list has length equal to the number of steps taken.
    """

    inflation: list[float] = field(default_factory=list)
    output_gap: list[float] = field(default_factory=list)
    unemployment: list[float] = field(default_factory=list)
    credit_growth: list[float] = field(default_factory=list)
    credit_spread: list[float] = field(default_factory=list)
    bank_leverage: list[float] = field(default_factory=list)
    default_rate: list[float] = field(default_factory=list)
    asset_price_gap: list[float] = field(default_factory=list)
    policy_rate: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)

    def append_state(self, state: MacroState, reward: float) -> None:
        self.inflation.append(state.inflation)
        self.output_gap.append(state.output_gap)
        self.unemployment.append(state.unemployment)
        self.credit_growth.append(state.credit_growth)
        self.credit_spread.append(state.credit_spread)
        self.bank_leverage.append(state.bank_leverage)
        self.default_rate.append(state.default_rate)
        self.asset_price_gap.append(state.asset_price_gap)
        self.policy_rate.append(state.policy_rate)
        self.reward.append(reward)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "inflation": self.inflation,
                "output_gap": self.output_gap,
                "unemployment": self.unemployment,
                "credit_growth": self.credit_growth,
                "credit_spread": self.credit_spread,
                "bank_leverage": self.bank_leverage,
                "default_rate": self.default_rate,
                "asset_price_gap": self.asset_price_gap,
                "policy_rate": self.policy_rate,
                "reward": self.reward,
            }
        )

    @property
    def total_reward(self) -> float:
        return float(np.sum(self.reward))

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.reward))

    @property
    def inflation_rmse(self) -> float:
        """Root mean squared deviation of inflation from the 2% target."""
        arr = np.array(self.inflation)
        return float(np.sqrt(np.mean((arr - 0.02) ** 2)))

    @property
    def output_gap_rmse(self) -> float:
        arr = np.array(self.output_gap)
        return float(np.sqrt(np.mean(arr**2)))

    @property
    def crisis_count(self) -> int:
        """
        Number of periods where the default rate exceeds 8%.
        This is a rough proxy for financial stress episodes.
        """
        return int(np.sum(np.array(self.default_rate) > 0.08))

    @property
    def peak_leverage(self) -> float:
        return float(np.max(self.bank_leverage))

    @property
    def peak_spread(self) -> float:
        return float(np.max(self.credit_spread))


def run_taylor_episode(
    params: ModelParameters | None = None,
    horizon: int = 120,
    seed: int = 0,
    rule: TaylorRule | None = None,
) -> Trajectory:
    """
    Run a single episode under the Taylor rule.
    Returns the full trajectory.
    """
    params = params or ModelParameters()
    rule = rule or TaylorRule()
    env = CentralBankEnv(params=params, horizon=horizon)

    _, _ = env.reset(seed=seed)
    traj = Trajectory()
    done = False

    while not done:
        assert env.state is not None
        next_rate = rule(env.state, params)

        action = min(
            range(len(env.rate_grid)),
            key=lambda i: abs(float(env.rate_grid[i]) - next_rate),
        )

        obs, reward, terminated, truncated, _ = env.step(action)
        traj.append_state(env.state, reward)
        done = terminated or truncated

    return traj


def run_policy_episode(
    policy_fn,
    params: ModelParameters | None = None,
    horizon: int = 120,
    seed: int = 0,
) -> Trajectory:
    """
    Run a single episode under an arbitrary policy function.

    policy_fn: callable(observation: np.ndarray) -> int
        Takes the raw obs vector, returns an action index.
    """
    params = params or ModelParameters()
    env = CentralBankEnv(params=params, horizon=horizon)

    obs, _ = env.reset(seed=seed)
    traj = Trajectory()
    done = False

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.state is not None
        traj.append_state(env.state, reward)
        done = terminated or truncated

    return traj