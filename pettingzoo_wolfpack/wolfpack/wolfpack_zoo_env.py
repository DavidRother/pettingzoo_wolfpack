from __future__ import annotations

import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gymnasium.utils import seeding
import gymnasium as gym
from pettingzoo.utils.env import ObsType, ActionType

from pettingzoo_wolfpack.wolfpack.wolfpack_env import WolfPackEnv


def env(env_name, ep_max_timesteps, n_predator, prefix, seed, obs="vector", agent_respawn_rate=0.0,
                 grace_period=20, agent_despawn_rate=0.0):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = WolfpackEnvironment(env_name, ep_max_timesteps, n_predator, prefix, seed, obs, agent_respawn_rate,
                                   grace_period, agent_despawn_rate)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class WolfpackEnvironment(AECEnv):
    """Environment object for Level Based Foraging."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        'render.modes': ['human', "rgb_array"],
        "name": "lbf_v1",
        "is_parallelizable": True,
        "render_fps": 20,
    }

    def __init__(self, env_name, ep_max_timesteps, n_predator, prefix, seed, obs, agent_respawn_rate=0.0,
                 grace_period=20, agent_despawn_rate=0.0):
        super().__init__()
        self.wolfpack_env = WolfPackEnv(env_name, ep_max_timesteps, n_predator, prefix, seed, obs, agent_respawn_rate,
                                        grace_period, agent_despawn_rate)
        self.possible_agents = ["player_" + str(r) for r in range(n_predator + 1)]
        self.agents = self.possible_agents[:]
        self.t = 0

        self.termination_info = ""
        self.observation_spaces = {agent: self.wolfpack_env.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.wolfpack_env.action_space for agent in self.possible_agents}
        self.has_reset = True
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.render_mode = "human"
        self.np_random = None
        self.agent_observations = {agent: None for agent in self.possible_agents}

    def step(self, action: ActionType) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        nobs, nreward, nterminated, ntruncated, ninfo = self.wolfpack_env.step(actions)

        for idx, agent in enumerate(self.agents):
            if idx and not self.wolfpack_env.active_predators[idx - 1]:
                self.agents.remove(agent)
                continue

            self.rewards[agent] = nreward[idx]
            self.terminations[agent] = nterminated[idx]
            self.truncations[agent] = ntruncated[idx]
            self.infos[agent] = ninfo[idx]
            self._cumulative_rewards[agent] += nreward[idx]
            self.agent_observations[agent] = nobs[idx]

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.t = 0

        # For tracking data during an episode.
        self.termination_info = ""

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        nobs, ninfo = self.wolfpack_env.reset()

        # Get an image observation
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, ninfo))
        self.accumulated_actions = []
        self.agent_observations = {agent: nobs[idx] for idx, agent in enumerate(self.possible_agents)}

    def observe(self, agent: str) -> ObsType | None:
        return self.agent_observations[agent]

    def render(self) -> None | np.ndarray | str | list:
        return self.wolfpack_env.render()

    def state(self) -> np.ndarray:
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
