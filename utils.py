import gymnasium as gym


def make_env(env_id, ep_max_timesteps, n_predator, prefix, seed):
    import pettingzoo_wolfpack  # noqa
    env = gym.make(env_id, env_name=env_id, ep_max_timesteps=ep_max_timesteps,
                   n_predator=n_predator, prefix=prefix, seed=seed)
    env._max_episode_steps = ep_max_timesteps

    return env
