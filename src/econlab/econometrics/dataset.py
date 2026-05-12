from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from econlab.simulation.runner import Trajectory


def trajectories_to_panel(
    trajectories: list[Trajectory],
    policy_name: str,
) -> pd.DataFrame:
    """
    Convert a list of trajectories into a long-format panel DataFrame.

    Columns: episode_id, period, policy_name, + all macro variables.
    This is the canonical input format for all econometric estimators.
    """
    frames = []
    for ep_id, traj in enumerate(trajectories):
        df = traj.to_dataframe()
        df.insert(0, "period", np.arange(len(df)))
        df.insert(0, "episode_id", ep_id)
        df.insert(0, "policy", policy_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_pseudo_time_series(
    trajectories: list[Trajectory],
    policy_name: str,
) -> pd.DataFrame:
    """
    Concatenate all trajectories end-to-end into a single long time series.
    Treats each episode as a continuation of the previous one.
    Used for VAR and LP estimation that assume a single long series.
    """
    all_dfs = []
    global_t = 0
    for ep_id, traj in enumerate(trajectories):
        df = traj.to_dataframe()
        df.insert(0, "global_period", np.arange(global_t, global_t + len(df)))
        df.insert(0, "episode_id", ep_id)
        df.insert(0, "policy", policy_name)
        global_t += len(df)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def add_lags(df: pd.DataFrame, variables: list[str], n_lags: int = 4) -> pd.DataFrame:
    """
    Add lag columns for each variable in variables.
    Operates within episode_id so lags don't bleed across episodes.
    """
    df = df.copy()
    for var in variables:
        for lag in range(1, n_lags + 1):
            df[f"{var}_lag{lag}"] = (
                df.groupby("episode_id")[var].shift(lag)
            )
    return df


def add_leads(df: pd.DataFrame, variables: list[str], n_leads: int = 12) -> pd.DataFrame:
    """
    Add forward-looking lead columns for LP-IRF estimation.
    h steps ahead: y_{t+h} for h = 1 ... n_leads.
    """
    df = df.copy()
    for var in variables:
        for h in range(1, n_leads + 1):
            df[f"{var}_lead{h}"] = (
                df.groupby("episode_id")[var].shift(-h)
            )
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Dataset saved: {path} ({len(df):,} rows)")


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)