from __future__ import annotations

from statistics import mean

from stable_baselines3.common.env_checker import check_env

from econlab.core.parameters import ModelParameters
from econlab.envs.cb_env import CentralBankEnv
from econlab.policy.taylor import TaylorRule


def run_random_episode() -> None:
    env = CentralBankEnv()
    obs, info = env.reset(seed=42)

    rewards = []
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    print("Random policy episode")
    print(f"  steps: {len(rewards)}")
    print(f"  avg reward: {mean(rewards):.6f}")
    print(f"  final inflation: {info['inflation']:.4f}")
    print(f"  final output gap: {info['output_gap']:.4f}")
    print(f"  final policy rate: {info['policy_rate']:.4f}")


def run_taylor_episode() -> None:
    params = ModelParameters()
    env = CentralBankEnv(params=params)
    rule = TaylorRule()

    obs, info = env.reset(seed=42)
    rewards = []
    done = False

    while not done:
        assert env.state is not None
        next_rate = rule(env.state, params)

        action = min(
            range(len(env.rate_grid)),
            key=lambda i: abs(float(env.rate_grid[i]) - next_rate),
        )

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    print("Taylor rule episode")
    print(f"  steps: {len(rewards)}")
    print(f"  avg reward: {mean(rewards):.6f}")
    print(f"  final inflation: {info['inflation']:.4f}")
    print(f"  final output gap: {info['output_gap']:.4f}")
    print(f"  final policy rate: {info['policy_rate']:.4f}")


def main() -> None:
    env = CentralBankEnv()
    check_env(env)

    print("Environment check passed.")
    print("-" * 40)

    run_random_episode()
    print("-" * 40)
    run_taylor_episode()


if __name__ == "__main__":
    main()