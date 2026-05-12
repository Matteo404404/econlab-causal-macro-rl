from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from econlab.core.parameters import ModelParameters
from econlab.envs.cb_env import CentralBankEnv


MODELS_DIR = Path("results/models")
LOGS_DIR  = Path("results/logs")

DISCRETE_ALGOS = {"PPO", "A2C", "DQN"}
CONTINUOUS_ALGOS = {"SAC", "TD3"}


@dataclass
class TrainingConfig:
    algorithm: str = "SAC"
    total_timesteps: int = 300_000
    horizon: int = 120
    n_envs: int = 1          # SAC/TD3 don't use vectorized envs

    # PPO/A2C
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001

    # SAC-specific
    buffer_size: int = 300_000
    learning_starts: int = 5_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1

    net_arch: list = field(default_factory=lambda: [256, 256, 128])
    device: str = "cuda"

    eval_freq: int = 10_000
    n_eval_episodes: int = 30
    eval_seed: int = 9999

    run_name: str = "sac_cb_v1"


class StepLogger(BaseCallback):
    def __init__(self, log_interval: int = 20_000):
        super().__init__()
        self.log_interval = log_interval
        self._n = 0

    def _on_step(self) -> bool:
        self._n += 1
        if self._n % self.log_interval == 0 and len(self.model.ep_info_buffer) > 0:
            mean_r = sum(e["r"] for e in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            print(f"  step {self._n:>8d} | mean ep reward: {mean_r:.4f}")
        return True


def make_env(params, horizon, seed, use_discrete=False):
    def _init():
        env = CentralBankEnv(params=params, horizon=horizon, use_discrete=use_discrete)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(config: TrainingConfig | None = None,
          params: ModelParameters | None = None,
          save: bool = True) -> tuple:
    config = config or TrainingConfig()
    params = params or ModelParameters()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    is_continuous = config.algorithm in CONTINUOUS_ALGOS
    use_discrete = not is_continuous

    # SAC/TD3: single env (no VecEnv needed, uses replay buffer)
    if is_continuous:
        train_env = make_env(params, config.horizon, seed=0, use_discrete=False)()
        eval_env  = make_env(params, config.horizon, seed=config.eval_seed, use_discrete=False)()
    else:
        raw = DummyVecEnv([make_env(params, config.horizon, seed=i, use_discrete=True)
                           for i in range(config.n_envs)])
        train_env = VecNormalize(raw, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0, clip_reward=10.0)
        raw_eval = DummyVecEnv([make_env(params, config.horizon,
                                         seed=config.eval_seed, use_discrete=True)])
        eval_env = VecNormalize(raw_eval, norm_obs=True, norm_reward=False, training=False)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / config.run_name),
        log_path=str(LOGS_DIR / config.run_name),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    policy_kwargs = dict(net_arch=config.net_arch)

    if config.algorithm == "SAC":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            ent_coef="auto",        # SAC auto-tunes entropy
            policy_kwargs=policy_kwargs,
            device=config.device,
            verbose=0,
        )
    elif config.algorithm == "TD3":
        model = TD3(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            policy_kwargs=policy_kwargs,
            device=config.device,
            verbose=0,
        )
    elif config.algorithm == "PPO":
        model = PPO("MlpPolicy", train_env,
                    learning_rate=config.learning_rate,
                    n_steps=config.n_steps, batch_size=config.batch_size,
                    n_epochs=config.n_epochs, gamma=config.gamma,
                    gae_lambda=config.gae_lambda, clip_range=config.clip_range,
                    ent_coef=config.ent_coef, policy_kwargs=policy_kwargs,
                    device=config.device, verbose=0)
    elif config.algorithm == "A2C":
        model = A2C("MlpPolicy", train_env,
                    learning_rate=config.learning_rate, gamma=config.gamma,
                    ent_coef=config.ent_coef, policy_kwargs=policy_kwargs,
                    device=config.device, verbose=0)
    elif config.algorithm == "DQN":
        model = DQN("MlpPolicy", train_env,
                    learning_rate=config.learning_rate, gamma=config.gamma,
                    policy_kwargs=policy_kwargs, device=config.device, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    print(f"\nTraining {config.algorithm} for {config.total_timesteps:,} timesteps")
    print(f"  action_space={'continuous' if is_continuous else 'discrete'}")
    print(f"  horizon={config.horizon}, device={config.device}")
    print(f"  net_arch={config.net_arch}, lr={config.learning_rate}")
    print(f"  run_name={config.run_name}")
    print("-" * 55)

    model.learn(total_timesteps=config.total_timesteps,
                callback=[eval_cb, StepLogger()],
                progress_bar=False)

    if save:
        final_path = MODELS_DIR / config.run_name / "final_model"
        model.save(str(final_path))
        print(f"\nFinal model saved: {final_path}")

    return model, config


def load_model(run_name: str, algorithm: str = "SAC"):
    best  = MODELS_DIR / run_name / "best_model"
    final = MODELS_DIR / run_name / "final_model"
    path  = best if best.with_suffix(".zip").exists() else final
    cls   = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC, "TD3": TD3}.get(algorithm, SAC)
    model = cls.load(str(path))
    print(f"Loaded model from: {path}")
    return model


def model_policy_fn(model):
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        # For continuous: action is array; for discrete: scalar
        if hasattr(action, "__len__"):
            return action  # runner will handle float rate directly
        return int(action)
    return policy_fn
