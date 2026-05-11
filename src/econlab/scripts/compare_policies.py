from __future__ import annotations

import argparse
from pathlib import Path

from econlab.core.parameters import ModelParameters
from econlab.plots.macro import (
    plot_monte_carlo_bands,
    plot_reward_distribution,
    plot_trajectory,
)
from econlab.rl.sb3_train import TrainingConfig, load_model, model_policy_fn
from econlab.simulation.monte_carlo import (
    MonteCarloResult,
    compare_results_table,
    run_policy_monte_carlo,
    run_taylor_monte_carlo,
)
from econlab.simulation.runner import run_policy_episode, run_taylor_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Taylor rule vs trained RL policies."
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name of a trained RL run to load and compare. "
                             "If not given, runs Taylor only + random baseline.")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--plots", action="store_true",
                        help="Show and save comparison plots.")
    parser.add_argument("--plot-dir", type=str, default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParameters()
    plot_dir = Path(args.plot_dir)

    print(f"\nRunning Monte Carlo: {args.n_episodes} episodes, horizon={args.horizon}")
    print("-" * 55)

    results: list[MonteCarloResult] = []
    all_trajectories: dict[str, list] = {}

    # --- Taylor rule ---
    print("Policy: Taylor Rule")
    taylor_trajs = [
        run_taylor_episode(params=params, horizon=args.horizon, seed=i)
        for i in range(args.n_episodes)
    ]
    taylor_mc = run_taylor_monte_carlo(
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        params=params,
    )
    results.append(taylor_mc)
    all_trajectories["Taylor Rule"] = taylor_trajs

    # --- Random baseline ---
    print("Policy: Random")
    from econlab.envs.cb_env import CentralBankEnv
    import numpy as np

    def random_policy(obs):
        env_tmp = CentralBankEnv(params=params, horizon=args.horizon)
        return env_tmp.action_space.sample()

    random_mc = run_policy_monte_carlo(
        policy_fn=random_policy,
        policy_name="Random",
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        params=params,
    )
    random_trajs = [
        run_policy_episode(policy_fn=random_policy, params=params,
                           horizon=args.horizon, seed=i)
        for i in range(args.n_episodes)
    ]
    results.append(random_mc)
    all_trajectories["Random"] = random_trajs

    # --- Trained RL policy (if provided) ---
    if args.run_name:
        print(f"Policy: {args.algo} ({args.run_name})")
        model = load_model(args.run_name, algorithm=args.algo)
        rl_fn = model_policy_fn(model)

        rl_mc = run_policy_monte_carlo(
            policy_fn=rl_fn,
            policy_name=args.algo,
            n_episodes=args.n_episodes,
            horizon=args.horizon,
            params=params,
        )
        rl_trajs = [
            run_policy_episode(
                policy_fn=rl_fn, params=params,
                horizon=args.horizon, seed=i
            )
            for i in range(args.n_episodes)
        ]
        results.append(rl_mc)
        all_trajectories[args.algo] = rl_trajs

    # --- Print comparison table ---
    print("\n" + "=" * 55)
    print("POLICY COMPARISON TABLE")
    print("=" * 55)
    df = compare_results_table(results)
    print(df.to_string())
    print()

    # --- Save table to CSV ---
    table_path = plot_dir / "policy_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path)
    print(f"Table saved: {table_path}")

    # --- Plots ---
    if args.plots:
        # Single Taylor episode trajectory
        plot_trajectory(
            taylor_trajs[0],
            title="Taylor Rule — Single Episode",
            save_path=plot_dir / "taylor_single_episode.png",
        )

        # Monte Carlo bands for inflation
        plot_monte_carlo_bands(
            all_trajectories,
            variable="inflation",
            title="Inflation: Policy Comparison (Monte Carlo Bands)",
            save_path=plot_dir / "mc_inflation_bands.png",
        )

        # Monte Carlo bands for output gap
        plot_monte_carlo_bands(
            all_trajectories,
            variable="output_gap",
            title="Output Gap: Policy Comparison (Monte Carlo Bands)",
            save_path=plot_dir / "mc_output_gap_bands.png",
        )

        # Reward distribution boxplot
        reward_series = {
            name: [t.total_reward for t in trajs]
            for name, trajs in all_trajectories.items()
        }
        plot_reward_distribution(
            reward_series,
            title="Total Reward Distribution by Policy",
            save_path=plot_dir / "reward_distribution.png",
        )


if __name__ == "__main__":
    main()