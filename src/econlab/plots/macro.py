from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from econlab.simulation.runner import Trajectory

INFLATION_TARGET = 0.02

# Clean style — no unnecessary gridlines or chart junk
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def plot_trajectory(
    traj: Trajectory,
    title: str = "Policy Episode",
    save_path: Path | None = None,
) -> None:
    """
    Plot the key macro variables from a single episode trajectory.
    Shows inflation vs target, output gap, policy rate, credit spread,
    bank leverage, and default rate in a 2x3 grid.
    """
    df = traj.to_dataframe()
    t = np.arange(len(df))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # Inflation vs target
    ax = axes[0, 0]
    ax.plot(t, df["inflation"] * 100, color="#01696f", linewidth=1.5, label="Inflation")
    ax.axhline(INFLATION_TARGET * 100, color="#a12c7b", linestyle="--",
               linewidth=1.0, label="Target (2%)")
    ax.set_title("Inflation (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)

    # Output gap
    ax = axes[0, 1]
    ax.plot(t, df["output_gap"] * 100, color="#006494", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Output Gap (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Policy rate
    ax = axes[0, 2]
    ax.plot(t, df["policy_rate"] * 100, color="#964219", linewidth=1.5)
    ax.set_title("Policy Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Credit spread
    ax = axes[1, 0]
    ax.plot(t, df["credit_spread"] * 100, color="#7a39bb", linewidth=1.5)
    ax.set_title("Credit Spread (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Bank leverage
    ax = axes[1, 1]
    ax.plot(t, df["bank_leverage"], color="#da7101", linewidth=1.5)
    ax.axhline(10.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Neutral (10x)")
    ax.set_title("Bank Leverage (x)")
    ax.legend(fontsize=8)

    # Default rate
    ax = axes[1, 2]
    ax.plot(t, df["default_rate"] * 100, color="#a13544", linewidth=1.5)
    ax.axhline(8.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Stress threshold (8%)")
    ax.set_title("Default Rate (%)")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Period")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def plot_policy_comparison(
    trajectories: dict[str, Trajectory],
    variable: str = "inflation",
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Overlay a single variable across multiple policy trajectories.
    Useful for direct visual comparison of Taylor vs RL policies.

    trajectories: dict mapping policy name to Trajectory
    variable: column name from Trajectory.to_dataframe()
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544", "#da7101"]

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (name, traj) in enumerate(trajectories.items()):
        df = traj.to_dataframe()
        t = np.arange(len(df))
        color = palette[i % len(palette)]
        ax.plot(t, df[variable], color=color, linewidth=1.6, label=name)

    if variable == "inflation":
        ax.axhline(INFLATION_TARGET, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="Target (2%)")

    if variable == "output_gap":
        ax.axhline(0.0, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.5)

    ax.set_xlabel("Period")
    ax.set_ylabel(variable.replace("_", " ").title())
    ax.set_title(title or f"Policy Comparison: {variable.replace('_', ' ').title()}")
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def plot_reward_distribution(
    reward_series: dict[str, list[float]],
    title: str = "Total Reward Distribution by Policy",
    save_path: Path | None = None,
) -> None:
    """
    Box plot comparing total reward distributions across policies.
    reward_series: dict mapping policy name to list of total rewards per episode.
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544"]

    fig, ax = plt.subplots(figsize=(8, 5))

    data = list(reward_series.values())
    labels = list(reward_series.keys())

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )

    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Total Episode Reward")
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def plot_monte_carlo_bands(
    all_trajectories: dict[str, list[Trajectory]],
    variable: str = "inflation",
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """
    For each policy, plot the median trajectory plus a shaded 10th-90th percentile band.
    This is the main research-quality figure for comparing policies.

    all_trajectories: dict mapping policy name to list of Trajectory objects
    """
    palette = ["#01696f", "#964219", "#7a39bb", "#006494", "#a13544", "#da7101"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (name, trajs) in enumerate(all_trajectories.items()):
        color = palette[i % len(palette)]
        min_len = min(len(t.to_dataframe()) for t in trajs)

        matrix = np.array(
            [t.to_dataframe()[variable].values[:min_len] for t in trajs]
        )

        t = np.arange(min_len)
        median = np.median(matrix, axis=0)
        p10 = np.percentile(matrix, 10, axis=0)
        p90 = np.percentile(matrix, 90, axis=0)

        ax.plot(t, median, color=color, linewidth=2.0, label=f"{name} (median)")
        ax.fill_between(t, p10, p90, color=color, alpha=0.15, label=f"{name} (10-90th pct)")

    if variable == "inflation":
        ax.axhline(INFLATION_TARGET, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="Target")

    if variable == "output_gap":
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)

    ax.set_xlabel("Period")
    ax.set_ylabel(variable.replace("_", " ").title())
    ax.set_title(title or f"Monte Carlo: {variable.replace('_', ' ').title()}")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()