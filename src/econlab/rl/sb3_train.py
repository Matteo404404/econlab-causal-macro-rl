from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from econlab.core.parameters import ModelParameters
from econlab.envs.cb_env import CentralBankEnv


MODELS_DIR = Path("results/models")
LOGS_DIR = Path("results/logs")


@dataclass
class TrainingConfig:
    """
    All hyperparameters for one training run in one place.
    Change this dataclass — don't scatter magic numbers in the code.
    """

    algorithm: str = "PPO"       # "PPO", "A2C", or "DQN"
    total_timesteps: int = 200_000
    horizon: int = 120
    n_envs: int = 4              # parallel environments for PPO/A2C

    # PPO-specific
    learning_rate: float = 3e-4
    n_steps: int = 512           # rollout buffer size per env
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01       # entropy bonus to encourage exploration

    # Evaluation
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    eval_seed: int = 9999

    # Saving
    run_name: str = "ppo_cb_v1"


class EpisodeRewardLogger(BaseCallback):
    """
    Logs mean episode reward every n_steps to stdout.
    Keeps it simple — no tensorboard dependency required.
    """

    def __init__(self, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.log_interval == 0:
            if "episode" in self.locals.get("infos", [{}])[0]:
                ep_rew = self.locals["infos"][0]["episode"]["r"]
                print(f"  step {self._step_count:>8d} | episode reward: {ep_rew:.4f}")
        return True


def make_env(params: ModelParameters, horizon: int, seed: int):
    """Factory for a monitored environment instance."""
    def _init():
        env = CentralBankEnv(params=params, horizon=horizon)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(
    config: TrainingConfig | None = None,
    params: ModelParameters | None = None,
    save: bool = True,
) -> tuple:
    """
    Train a policy agent and return (model, training_config).

    Parameters
    ----------
    config : TrainingConfig, optional
    params : ModelParameters, optional
    save   : bool — whether to save the model to disk

    Returns
    -------
    (model, config)
    """
    config = config or TrainingConfig()
    params = params or ModelParameters()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Build vectorized training environments
    train_env = DummyVecEnv(
        [make_env(params, config.horizon, seed=i) for i in range(config.n_envs)]
    )

    # Build a separate deterministic eval environment
    eval_env = DummyVecEnv(
        [make_env(params, config.horizon, seed=config.eval_seed)]
    )

    # Reward threshold callback — stops training if eval mean reward is good enough
    reward_threshold_cb = StopTrainingOnRewardThreshold(
        reward_threshold=-0.0005, verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / config.run_name),
        log_path=str(LOGS_DIR / config.run_name),
        eval_freq=max(config.eval_freq // config.n_envs, 1),
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        callback_on_new_best=reward_threshold_cb,
        verbose=1,
    )

    logger_cb = EpisodeRewardLogger(log_interval=10_000)

    # Construct the model
    if config.algorithm == "PPO":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=0,
        )
    elif config.algorithm == "A2C":
        model = A2C(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            verbose=0,
        )
    elif config.algorithm == "DQN":
        # DQN requires a single env
        single_env = DummyVecEnv([make_env(params, config.horizon, seed=0)])
        model = DQN(
            policy="MlpPolicy",
            env=single_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    print(f"\nTraining {config.algorithm} for {config.total_timesteps:,} timesteps")
    print(f"  horizon={config.horizon}, n_envs={config.n_envs}")
    print(f"  run_name={config.run_name}")
    print("-" * 50)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_cb, logger_cb],
        progress_bar=False,
    )

    if save:
        final_path = MODELS_DIR / config.run_name / "final_model"
        model.save(str(final_path))
        print(f"\nFinal model saved to: {final_path}")

    return model, config


def load_model(run_name: str, algorithm: str = "PPO"):
    """Load the best saved model for a given run."""
    best_path = MODELS_DIR / run_name / "best_model"
    final_path = MODELS_DIR / run_name / "final_model"

    path = best_path if best_path.with_suffix(".zip").exists() else final_path

    cls_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    cls = cls_map.get(algorithm, PPO)

    model = cls.load(str(path))
    print(f"Loaded model from: {path}")
    return model


def model_policy_fn(model):
    """
    Wrap a trained SB3 model into a simple callable
    that takes an obs array and returns an action int.
    Compatible with run_policy_episode() in runner.py.
    """
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy_fn