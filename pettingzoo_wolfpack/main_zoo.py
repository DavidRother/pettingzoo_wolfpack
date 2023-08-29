import random
import numpy as np
from pettingzoo_wolfpack.wolfpack.wolfpack_zoo_env import parallel_env


def main(env_id, ep_max_timesteps, n_predator, prefix, seed, obs):
    # Create directories
    # if not os.path.exists("./logs"):
    #     os.makedirs("./logs")

    # Set logs
    # log = set_log(args)

    # Create env
    env = parallel_env(env_name=env_id, ep_max_timesteps=ep_max_timesteps, n_predator=n_predator,
                       prefix=prefix, seed=seed, obs=obs)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    accumulated_rewards = {"player_0": 0, "player_1": 0, "player_2": 0}

    # Visualize environment
    observations, infos = env.reset()
    steps = 0
    while True:
        # env.render()
        steps += 1
        prey_action = env.action_space("player_0").sample()
        predator1_action = env.action_space("player_1").sample()
        predator2_action = env.action_space("player_2").sample()
        actions = {"player_0": prey_action, "player_1": predator1_action, "player_2": predator2_action}

        observations, reward, done, truncated, _ = env.step(actions)

        for agent_id in accumulated_rewards.keys():
            accumulated_rewards[agent_id] += reward[agent_id]

        if any(done.values()):
            break

        if any(truncated.values()):
            break

    print(accumulated_rewards)
    print(steps)


if __name__ == "__main__":
    m_env_id = "wolfpack-v0"
    m_ep_max_timesteps = 150
    m_n_predator = 2
    m_prefix = ""
    m_seed = 1
    m_obs = "vector"
    main(m_env_id, m_ep_max_timesteps, m_n_predator, m_prefix, m_seed, m_obs)
