import random
import numpy as np
from pettingzoo_wolfpack.wolfpack.wolfpack_zoo_env import parallel_env
from pettingzoo_wolfpack.env_agents.prey_heuristic_agents import PreyAgentH1


def main(env_id, ep_max_timesteps, n_predator, prefix, seed, obs):
    # Create directories
    # if not os.path.exists("./logs"):
    #     os.makedirs("./logs")

    # Set logs
    # log = set_log(args)

    # Create env
    env = parallel_env(env_name=env_id, ep_max_timesteps=ep_max_timesteps, n_predator=n_predator,
                       prefix=prefix, seed=seed, obs=obs, agent_respawn_rate=0.1, grace_period=20,
                       agent_despawn_rate=0.1)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    accumulated_rewards = {"player_0": 0, "player_1": 0, "player_2": 0, "player_3": 0}
    prey_agent = PreyAgentH1()

    # Visualize environment
    observations, infos = env.reset()
    done = {"player_0": False, "player_1": False, "player_2": False, "player_3": False}
    truncated = {"player_0": False, "player_1": False, "player_2": False, "player_3": False}
    steps = 0
    while True:
        # env.render()
        steps += 1
        actions = {}
        if "player_0" in observations and not truncated["player_0"] and not done["player_0"]:
            prey_action = env.action_space("player_0").sample()
            actions["player_0"] = prey_action
        if "player_1" in observations and not truncated["player_1"] and not done["player_1"]:
            predator1_action = env.action_space("player_1").sample()
            actions["player_1"] = predator1_action
        if "player_2" in observations and not truncated["player_2"] and not done["player_2"]:
            predator2_action = env.action_space("player_2").sample()
            actions["player_2"] = predator2_action
        if "player_3" in observations and not truncated["player_3"] and not done["player_3"]:
            predator3_action = env.action_space("player_3").sample()
            actions["player_3"] = predator3_action

        observations, reward, done, truncated, infos = env.step(actions)
        print(truncated)
        for agent_id in accumulated_rewards:
            accumulated_rewards[agent_id] += reward[agent_id]

        if all(done.values()):
            break

        if all(truncated.values()):
            break

    print(accumulated_rewards)
    print(steps)


if __name__ == "__main__":
    m_env_id = "wolfpack-v0"
    m_ep_max_timesteps = 150
    m_n_predator = 3
    m_prefix = ""
    m_seed = 1
    m_obs = "vector"
    main(m_env_id, m_ep_max_timesteps, m_n_predator, m_prefix, m_seed, m_obs)
