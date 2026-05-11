from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from econlab.core.parameters import ModelParameters
from econlab.policy.taylor import TaylorRule
from econlab.simulation.runner import (
    Trajectory,
    run_policy_episode,
    run_taylor_episode,
)


@dataclass
class MonteCarloResult:
    """
    Summary statistics across N Monte Carlo episodes.
    """

    policy_name: str
    n_episodes: int
    mean_total_reward: float
    std_total_reward: float
    mean_inflation_rmse: float
    std_inflation_rmse: float
    mean_output_gap_rmse: float
    std_output_gap_rmse: float
    mean_crisis_count: float
    std_crisis_count: float
    mean_peak_leverage: float
    mean_peak_spread: float
    p5_total_reward: float
    p95_total_reward: float
    crisis_episode_fraction: float

    def summary_dict(self) -> dict:
        return {
            "policy": self.policy_name,
            "n_episodes": self.n_episodes,
            "mean_reward": round(self.mean_total_reward, 4),
            "std_reward": round(self.std_total_reward, 4),
            "p5_reward": round(self.p5_total_reward, 4),
            "p95_reward": round(self.p95_total_reward, 4),
            "inflation_rmse": round(self.mean_inflation_rmse, 5),
            "output_gap_rmse": round(self.mean_output_gap_rmse, 5),
            "mean_crisis_periods": round(self.mean_crisis_count, 2),
            "crisis_episode_frac": round(self.crisis_episode_fraction, 3),
            "peak_leverage": round(self.mean_peak_leverage, 2),
            "peak_spread": round(self.mean_peak_spread, 4),
        }


def _aggregate(
    trajectories: list[Trajectory],
    policy_name: str,
) -> MonteCarloResult:
    total_rewards = np.array([t.total_reward for t in trajectories])
    inf_rmses = np.array([t.inflation_rmse for t in trajectories])
    og_rmses = np.array([t.output_gap_rmse for t in trajectories])
    crisis_counts = np.array([t.crisis_count for t in trajectories])
    peak_leverages = np.array([t.peak_leverage for t in trajectories])
    peak_spreads = np.array([t.peak_spread for t in trajectories])

    return MonteCarloResult(
        policy_name=policy_name,
        n_episodes=len(trajectories),
        mean_total_reward=float(np.mean(total_rewards)),
        std_total_reward=float(np.std(total_rewards)),
        mean_inflation_rmse=float(np.mean(inf_rmses)),
        std_inflation_rmse=float(np.std(inf_rmses)),
        mean_output_gap_rmse=float(np.mean(og_rmses)),
        std_output_gap_rmse=float(np.std(og_rmses)),
        mean_crisis_count=float(np.mean(crisis_counts)),
        std_crisis_count=float(np.std(crisis_counts)),
        mean_peak_leverage=float(np.mean(peak_leverages)),
        mean_peak_spread=float(np.mean(peak_spreads)),
        p5_total_reward=float(np.percentile(total_rewards, 5)),
        p95_total_reward=float(np.percentile(total_rewards, 95)),
        crisis_episode_fraction=float(np.mean(crisis_counts > 0)),
    )


def run_taylor_monte_carlo(
    n_episodes: int = 100,
    horizon: int = 120,
    params: ModelParameters | None = None,
    rule: TaylorRule | None = None,
    base_seed: int = 0,
) -> MonteCarloResult:
    """Run N Taylor-rule episodes and return aggregated statistics."""
    params = params or ModelParameters()
    rule = rule or TaylorRule()

    trajectories = [
        run_taylor_episode(
            params=params,
            horizon=horizon,
            seed=base_seed + i,
            rule=rule,
        )
        for i in range(n_episodes)
    ]

    return _aggregate(trajectories, policy_name="Taylor Rule")


def run_policy_monte_carlo(
    policy_fn,
    policy_name: str,
    n_episodes: int = 100,
    horizon: int = 120,
    params: ModelParameters | None = None,
    base_seed: int = 0,
) -> MonteCarloResult:
    """Run N episodes under an arbitrary policy and return aggregated statistics."""
    params = params or ModelParameters()

    trajectories = [
        run_policy_episode(
            policy_fn=policy_fn,
            params=params,
            horizon=horizon,
            seed=base_seed + i,
        )
        for i in range(n_episodes)
    ]

    return _aggregate(trajectories, policy_name=policy_name)


def compare_results_table(results: list[MonteCarloResult]) -> pd.DataFrame:
    """Return a DataFrame comparing multiple Monte Carlo results side by side."""
    rows = [r.summary_dict() for r in results]
    df = pd.DataFrame(rows).set_index("policy")
    return df