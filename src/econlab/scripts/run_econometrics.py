from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from econlab.core.parameters import ModelParameters
from econlab.econometrics.dataset import (
    add_lags,
    build_pseudo_time_series,
    save_dataset,
    trajectories_to_panel,
)
from econlab.econometrics.local_projections import compare_irfs_across_regimes
from econlab.econometrics.regime_analysis import (
    build_regime_panel,
    estimate_regime_differences,
    plot_regime_means,
)
from econlab.econometrics.var import fit_var, plot_var_irf
from econlab.rl.sb3_train import load_model, model_policy_fn
from econlab.simulation.runner import run_policy_episode, run_taylor_episode
from econlab.policy.taylor import TaylorRule


VAR_VARIABLES = [
    "inflation", "output_gap", "unemployment",
    "credit_spread", "bank_leverage", "default_rate", "policy_rate",
]

LP_CONTROLS = [
    "output_gap", "unemployment", "credit_spread", "bank_leverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full econometrics pipeline on simulated data."
    )
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name of a trained RL run to include.")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--max-horizon", type=int, default=12,
                        help="Maximum horizon for local projections.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--plot-dir", type=str, default="results/figures/econometrics")
    parser.add_argument("--data-dir", type=str, default="results/data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParameters()
    plot_dir = Path(args.plot_dir)
    data_dir = Path(args.data_dir)

    print(f"\nEconometrics pipeline: {args.n_episodes} episodes, horizon={args.horizon}")
    print("=" * 60)

    # --- Simulate Taylor rule trajectories ---
    print("\nSimulating Taylor Rule episodes...")
    taylor_trajs = [
        run_taylor_episode(params=params, horizon=args.horizon, seed=i)
        for i in range(args.n_episodes)
    ]

    trajectories_by_policy: dict[str, list] = {"Taylor Rule": taylor_trajs}
    panels: dict[str, pd.DataFrame] = {}
    ts_dict: dict[str, pd.DataFrame] = {}

    # --- Simulate RL trajectories if model provided ---
    if args.run_name:
        print(f"\nLoading RL model: {args.run_name}")
        model = load_model(args.run_name, algorithm=args.algo)
        rl_fn = model_policy_fn(model)

        print(f"Simulating {args.algo} episodes...")
        rl_trajs = [
            run_policy_episode(
                policy_fn=rl_fn, params=params,
                horizon=args.horizon, seed=i
            )
            for i in range(args.n_episodes)
        ]
        trajectories_by_policy[args.algo] = rl_trajs

    # --- Build datasets ---
    print("\nBuilding datasets...")
    for policy_name, trajs in trajectories_by_policy.items():
        panel = trajectories_to_panel(trajs, policy_name=policy_name)
        panel = add_lags(panel, LP_CONTROLS + ["inflation", "policy_rate"], n_lags=4)
        panels[policy_name] = panel

        ts = build_pseudo_time_series(trajs, policy_name=policy_name)
        ts_dict[policy_name] = ts

        save_dataset(panel, data_dir / f"panel_{policy_name.replace(' ', '_').lower()}.parquet")
        save_dataset(ts, data_dir / f"ts_{policy_name.replace(' ', '_').lower()}.parquet")

    # --- Local projections ---
    print("\nEstimating local projections (LP-IRF)...")

    for outcome in ["inflation", "output_gap", "credit_spread"]:
        save_path = (
            plot_dir / f"lp_irf_{outcome}_vs_policy_rate.png"
            if args.plots else None
        )
        compare_irfs_across_regimes(
            panel_dict=panels,
            outcome_var=outcome,
            shock_var="policy_rate",
            control_vars=LP_CONTROLS,
            max_horizon=args.max_horizon,
            n_lags=2,
            save_path=save_path,
        )

    # --- VAR estimation ---
    print("\nFitting VARs...")
    var_results = []
    for policy_name, ts in ts_dict.items():
        try:
            vr = fit_var(
                ts, variables=VAR_VARIABLES, max_lags=4, policy_name=policy_name
            )
            var_results.append(vr)
        except Exception as e:
            print(f"  VAR failed for {policy_name}: {e}")

    if args.plots and len(var_results) > 0:
        for impulse, response in [
            ("policy_rate", "inflation"),
            ("policy_rate", "output_gap"),
            ("policy_rate", "credit_spread"),
        ]:
            plot_var_irf(
                var_results,
                impulse_var=impulse,
                response_var=response,
                periods=args.max_horizon,
                save_path=plot_dir / f"var_irf_{response}_to_{impulse}.png",
            )

    # --- Regime analysis ---
    if len(trajectories_by_policy) > 1:
        print("\nRunning regime analysis...")
        regime_panel = build_regime_panel(trajectories_by_policy)

        diff_table = estimate_regime_differences(
            regime_panel,
            outcome_vars=[
                "inflation", "output_gap", "credit_spread",
                "default_rate", "bank_leverage",
            ],
            baseline_policy="Taylor Rule",
        )

        print("\nREGIME DIFFERENCE TABLE (vs Taylor Rule baseline)")
        print(diff_table.to_string(index=False))

        table_path = plot_dir / "regime_differences.csv"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        diff_table.to_csv(table_path, index=False)
        print(f"\nTable saved: {table_path}")

        if args.plots:
            plot_regime_means(
                regime_panel,
                variables=["inflation", "output_gap", "credit_spread", "default_rate"],
                save_path=plot_dir / "regime_mean_comparison.png",
            )

    print("\nEconometrics pipeline complete.")


if __name__ == "__main__":
    main()