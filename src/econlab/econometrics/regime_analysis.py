from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from econlab.econometrics.dataset import trajectories_to_panel
from econlab.simulation.runner import Trajectory


def build_regime_panel(
    trajectories_by_policy: dict[str, list[Trajectory]],
) -> pd.DataFrame:
    """
    Build a panel DataFrame with a regime indicator for each policy.
    Used for difference-in-differences and regime switching analysis.
    """
    frames = []
    for policy_name, trajs in trajectories_by_policy.items():
        df = trajectories_to_panel(trajs, policy_name=policy_name)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    # Add regime dummy: 1 = RL policy, 0 = Taylor/baseline
    all_policies = panel["policy"].unique().tolist()
    baseline = "Taylor Rule"
    panel["is_rl_regime"] = (panel["policy"] != baseline).astype(int)

    return panel


def estimate_regime_differences(
    panel: pd.DataFrame,
    outcome_vars: list[str],
    baseline_policy: str = "Taylor Rule",
) -> pd.DataFrame:
    """
    For each outcome variable, estimate the mean difference between
    each policy regime and the baseline using simple OLS with episode
    fixed effects (within estimator).

    Returns a tidy DataFrame of regime coefficients and standard errors.
    """
    results = []

    policies = [p for p in panel["policy"].unique() if p != baseline_policy]

    for outcome in outcome_vars:
        for policy in policies:
            sub = panel[panel["policy"].isin([baseline_policy, policy])].copy()
            sub = sub.dropna(subset=[outcome])

            if len(sub) < 10:
                continue

            # Dummy: 1 = this policy
            sub["regime_dummy"] = (sub["policy"] == policy).astype(int)

            # Include period fixed effect via demeaning within episode
            sub["outcome_dm"] = sub[outcome] - sub.groupby("episode_id")[outcome].transform("mean")
            sub["regime_dm"] = sub["regime_dummy"] - sub.groupby("episode_id")["regime_dummy"].transform("mean")

            X = sm.add_constant(sub[["regime_dm"]])
            y = sub["outcome_dm"]

            try:
                res = sm.OLS(y, X).fit(cov_type="HC3")
                coef = res.params.get("regime_dm", np.nan)
                se = res.bse.get("regime_dm", np.nan)
                pval = res.pvalues.get("regime_dm", np.nan)
            except Exception:
                coef, se, pval = np.nan, np.nan, np.nan

            results.append(
                {
                    "outcome": outcome,
                    "policy": policy,
                    "baseline": baseline_policy,
                    "coef": round(coef, 6),
                    "std_err": round(se, 6),
                    "p_value": round(pval, 4),
                    "n_obs": len(sub),
                    "significant_90": pval < 0.10 if not np.isnan(pval) else False,
                }
            )

    return pd.DataFrame(results)


def plot_regime_means(
    panel: pd.DataFrame,
    variables: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Bar chart comparing mean values of key variables across policy regimes.
    Error bars show 95% CI of the mean.
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544", "#da7101"]
    policies = panel["policy"].unique().tolist()

    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4))
    if len(variables) == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    for ax_i, var in enumerate(variables):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        means = []
        errors = []
        labels = []

        for policy in policies:
            vals = panel[panel["policy"] == policy][var].dropna()
            mean = vals.mean()
            se = vals.std() / np.sqrt(len(vals))
            means.append(mean)
            errors.append(1.96 * se)
            labels.append(policy)

        colors = [palette[i % len(palette)] for i in range(len(labels))]
        bars = ax.bar(labels, means, yerr=errors, color=colors, alpha=0.75,
                      capsize=4, edgecolor="white")
        ax.set_title(var.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()