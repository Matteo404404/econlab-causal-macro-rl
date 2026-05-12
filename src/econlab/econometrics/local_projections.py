from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class IRFResult:
    """
    Impulse response function estimated via local projections.

    horizon: array of horizon indices [0, 1, ..., H]
    point:   point estimate at each horizon
    ci_low:  lower bound of confidence interval
    ci_high: upper bound of confidence interval
    outcome_var: name of the outcome variable
    shock_var:   name of the shock/treatment variable
    """

    horizon: np.ndarray
    point: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    outcome_var: str
    shock_var: str
    policy_name: str
    n_obs: int
    ci_level: float = 0.90

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "horizon": self.horizon,
                "irf": self.point,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "outcome_var": self.outcome_var,
                "shock_var": self.shock_var,
                "policy": self.policy_name,
            }
        )


def estimate_local_projections(
    df: pd.DataFrame,
    outcome_var: str,
    shock_var: str,
    control_vars: list[str],
    max_horizon: int = 12,
    n_lags: int = 2,
    ci_level: float = 0.90,
    policy_name: str = "",
) -> IRFResult:
    """
    Estimate impulse response function using Jorda (2005) local projections.

    For each horizon h = 0, 1, ..., max_horizon, estimates:
        y_{t+h} - y_{t-1} = alpha_h + beta_h * shock_t + gamma_h * X_t + e_{t+h}

    beta_h traces out the impulse response.

    Parameters
    ----------
    df           : panel DataFrame from dataset.py (must have episode_id, period columns)
    outcome_var  : dependent variable name (e.g. 'inflation')
    shock_var    : shock/treatment variable (e.g. 'policy_rate')
    control_vars : list of control variables (lags will be added automatically)
    max_horizon  : maximum forecast horizon H
    n_lags       : number of lags of controls to include
    ci_level     : confidence interval level (0.90 = 90%)
    policy_name  : label for this policy regime

    Returns
    -------
    IRFResult
    """
    df = df.copy().dropna(subset=[outcome_var, shock_var])

    # Add lags of outcome and shock for controls
    all_lag_vars = [outcome_var, shock_var] + control_vars
    for var in all_lag_vars:
        for lag in range(1, n_lags + 1):
            col = f"{var}_lag{lag}"
            if col not in df.columns:
                df[col] = df.groupby("episode_id")[var].shift(lag)

    from scipy.stats import norm

    horizons = np.arange(0, max_horizon + 1)
    point_estimates = np.zeros(len(horizons))
    ci_lows = np.zeros(len(horizons))
    ci_highs = np.zeros(len(horizons))
    n_obs_list = []

    z_crit = norm.ppf(1 - (1.0 - ci_level) / 2)
    lag_cols = []
    for var in all_lag_vars:
        for lag in range(1, n_lags + 1):
            lag_cols.append(f"{var}_lag{lag}")

    for h in horizons:
        # Dependent variable: level at t+h (or cumulative change)
        y_col = f"_lp_y_h{h}"
        df[y_col] = df.groupby("episode_id")[outcome_var].shift(-h)

        reg_df = df[[y_col, shock_var] + lag_cols].dropna()
        if len(reg_df) < 20:
            point_estimates[h] = np.nan
            ci_lows[h] = np.nan
            ci_highs[h] = np.nan
            continue

        X = sm.add_constant(reg_df[[shock_var] + lag_cols])
        y = reg_df[y_col]

        try:
            # Use HC3 heteroskedasticity-consistent standard errors
            res = sm.OLS(y, X).fit(cov_type="HC3")
            point_estimates[h] = res.params[shock_var]
            se = res.bse[shock_var]
            ci_lows[h] = res.params[shock_var] - z_crit * se
            ci_highs[h] = res.params[shock_var] + z_crit * se
            n_obs_list.append(int(res.nobs))
        except Exception:
            point_estimates[h] = np.nan
            ci_lows[h] = np.nan
            ci_highs[h] = np.nan

        # Clean up temp column
        df.drop(columns=[y_col], inplace=True)

    return IRFResult(
        horizon=horizons,
        point=point_estimates,
        ci_low=ci_lows,
        ci_high=ci_highs,
        outcome_var=outcome_var,
        shock_var=shock_var,
        policy_name=policy_name,
        n_obs=int(np.mean(n_obs_list)) if n_obs_list else 0,
        ci_level=ci_level,
    )


def plot_irf(
    results: list[IRFResult],
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Plot impulse response functions with confidence bands.
    Overlays multiple IRFResult objects for policy comparison.
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")

    for i, res in enumerate(results):
        color = palette[i % len(palette)]
        ax.plot(res.horizon, res.point, color=color, linewidth=2.0,
                label=f"{res.policy_name}")
        ax.fill_between(
            res.horizon, res.ci_low, res.ci_high,
            color=color, alpha=0.15,
            label=f"{res.policy_name} ({int(res.ci_level*100)}% CI)"
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Horizon (periods)")
    ax.set_ylabel(f"Response of {results[0].outcome_var if results else ''}")
    ax.set_title(
        title or
        f"LP-IRF: Response of {results[0].outcome_var} to {results[0].shock_var}"
        if results else "LP-IRF"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def compare_irfs_across_regimes(
    panel_dict: dict[str, pd.DataFrame],
    outcome_var: str,
    shock_var: str,
    control_vars: list[str],
    max_horizon: int = 12,
    n_lags: int = 2,
    save_path: Path | None = None,
) -> dict[str, IRFResult]:
    """
    Estimate and plot LP-IRFs for multiple policy regimes side by side.

    panel_dict: dict mapping policy_name -> panel DataFrame
    Returns dict mapping policy_name -> IRFResult
    """
    results = {}
    irf_list = []

    for policy_name, df in panel_dict.items():
        print(f"  Estimating LP-IRF: {policy_name} | {outcome_var} ~ {shock_var}")
        irf = estimate_local_projections(
            df=df,
            outcome_var=outcome_var,
            shock_var=shock_var,
            control_vars=control_vars,
            max_horizon=max_horizon,
            n_lags=n_lags,
            policy_name=policy_name,
        )
        results[policy_name] = irf
        irf_list.append(irf)

    plot_irf(
        irf_list,
        title=f"LP-IRF Comparison: {outcome_var} response to {shock_var}",
        save_path=save_path,
    )

    return results