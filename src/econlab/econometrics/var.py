from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR


@dataclass
class VARResult:
    """Wrapper around a fitted VAR model with convenience methods."""

    model_fit: object          # statsmodels VARResultsWrapper
    variables: list[str]
    policy_name: str
    n_lags: int
    n_obs: int

    @property
    def aic(self) -> float:
        return float(self.model_fit.aic)

    @property
    def bic(self) -> float:
        return float(self.model_fit.bic)

    def irf(self, periods: int = 12):
        return self.model_fit.irf(periods)

    def summary(self) -> str:
        return str(self.model_fit.summary())


def fit_var(
    df: pd.DataFrame,
    variables: list[str],
    max_lags: int = 4,
    policy_name: str = "",
) -> VARResult:
    """
    Fit a reduced-form VAR on pseudo-real simulated time series data.

    Parameters
    ----------
    df         : time-series DataFrame (global_period as index or sorted)
    variables  : list of variable names to include in the VAR
    max_lags   : maximum lag order — AIC selection used to pick optimal lags
    policy_name: label for this regime

    Returns
    -------
    VARResult
    """
    data = df[variables].dropna()

    if len(data) < 4 * max_lags:
        raise ValueError(
            f"Not enough observations ({len(data)}) for VAR with max_lags={max_lags}."
        )

    model = VAR(data)
    lag_order_res = model.select_order(maxlags=max_lags)
    best_lags = lag_order_res.aic
    best_lags = max(best_lags, 1)

    fit = model.fit(best_lags)
    print(
        f"  VAR({best_lags}) fitted | {policy_name} | "
        f"AIC={fit.aic:.2f} BIC={fit.bic:.2f} | n={len(data)}"
    )

    return VARResult(
        model_fit=fit,
        variables=variables,
        policy_name=policy_name,
        n_lags=best_lags,
        n_obs=len(data),
    )


def plot_var_irf(
    var_results: list[VARResult],
    impulse_var: str,
    response_var: str,
    periods: int = 12,
    save_path: Path | None = None,
) -> None:
    """
    Plot VAR impulse responses for a given impulse-response pair
    across multiple policy regimes.
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")

    for i, vr in enumerate(var_results):
        color = palette[i % len(palette)]
        irf = vr.irf(periods)

        # Get the index of impulse and response variables
        try:
            imp_idx = vr.variables.index(impulse_var)
            res_idx = vr.variables.index(response_var)
        except ValueError:
            print(f"  Variable not found in {vr.policy_name}: {impulse_var} or {response_var}")
            continue

        # irf.irfs has shape (periods+1, n_vars, n_vars)
        # [h, response, impulse]
        irf_vals = irf.irfs[:, res_idx, imp_idx]
        h = np.arange(len(irf_vals))

        ax.plot(h, irf_vals, color=color, linewidth=2.0, label=vr.policy_name)

        # Bootstrap confidence bands if available
        try:
            err = irf.cum_effect_stderr()
        except Exception:
            err = None

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Horizon (periods)")
    ax.set_ylabel(f"Response: {response_var}")
    ax.set_title(f"VAR IRF: {response_var} response to {impulse_var} shock")
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()