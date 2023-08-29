import random
import numpy as np
from pettingzoo_wolfpack.wolfpack.wolfpack_env import WolfPackEnv


def main(env_id, ep_max_timesteps, n_predator, prefix, seed, obs):
    # Create directories
    # if not os.path.exists("./logs"):
    #     os.makedirs("./logs")

    # Set logs
    # log = set_log(args)

    # Create env
    env = WolfPackEnv(env_id, ep_max_timesteps, n_predator, prefix, seed, obs)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Visualize environment
    observations, infos = env.reset()

    while True:
        # env.render()

        prey_action = env.action_space.sample()
        predator1_action = env.action_space.sample()
        predator2_action = env.action_space.sample()
        actions = [prey_action, predator1_action, predator2_action]

        observations, reward, done, truncated, _ = env.step(actions)

        if any(done):
            break

        if any(truncated):
            break


if __name__ == "__main__":
    m_env_id = "wolfpack-v0"
    m_ep_max_timesteps = 150
    m_n_predator = 2
    m_prefix = ""
    m_seed = 1
    m_obs = "vector"
    main(m_env_id, m_ep_max_timesteps, m_n_predator, m_prefix, m_seed, m_obs)
