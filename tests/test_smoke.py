from econlab.envs.cb_env import CentralBankEnv


def test_env_reset_and_step():
    env = CentralBankEnv(horizon=5)
    obs, info = env.reset(seed=123)

    assert obs.shape == (9,)

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert truncated is True