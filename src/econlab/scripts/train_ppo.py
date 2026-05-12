from __future__ import annotations

import argparse

from econlab.core.parameters import ModelParameters
from econlab.rl.sb3_train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="SAC", choices=["PPO", "A2C", "DQN", "SAC", "TD3"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="sac_cb_v1")
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        algorithm=args.algo,
        total_timesteps=args.timesteps,
        horizon=args.horizon,
        n_envs=args.n_envs,
        run_name=args.run_name,
        ent_coef=args.ent_coef,
        device=args.device,
    )
    train(config=config, params=ModelParameters(), save=not args.no_save)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
