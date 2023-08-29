import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import gym
from pettingzoo_wolfpack.wolfpack.config import Config
from pettingzoo_wolfpack.wolfpack.agent import Agent


class WolfPackEnv:
    REWARD_LONELY = 1.
    REWARD_TEAM = 5.
    CAPTURE_RADIUS = 6.

    def __init__(self, env_name, ep_max_timesteps, n_predator, prefix, seed, obs="vector"):
        self.env_name = env_name
        self.ep_max_timesteps = ep_max_timesteps
        self.n_predator = n_predator
        self.prefix = prefix
        self.seed = seed
        self.config = Config()
        self.obs = obs
        if obs == "vector":
            self.observation_shape = (self.n_predator * 2 + 2, 2)
        else:
            self.observation_shape = (11, 11, 3)  # Format: (height, width, channel)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.observation_shape)
        self.action_space = gym.spaces.Discrete(len(self.config.action_dict))
        self.pad = np.max(self.observation_shape) - 2

        self.base_gridmap_array = self._load_gridmap_array()
        self.base_gridmap_image = self._to_image(self.base_gridmap_array)
        self.agents = []
        self.current_step = 0

    def reset(self, **kwargs):
        self._reset_agents()

        gridmap_image = self._render_gridmap()

        observations = []
        for agent in self.agents:
            if self.obs == "vector":
                observation = self._get_observation_vector(agent)
            else:
                observation = self._get_observation(agent, gridmap_image)
            observations.append(observation)

        infos = [{} for _ in range(len(self.agents))]

        return observations, infos

    def step(self, actions):
        assert len(actions) == self.n_predator + 1

        # Compute next locations
        for agent, action in zip(self.agents, actions):
            action = list(self.config.action_dict.keys())[action]

            if "spin" not in action: 
                next_location = agent.location + self.config.action_dict[action]
                next_orientation = agent.orientation
            else:
                next_location = agent.location
                next_orientation = agent.orientation + self.config.action_dict[action]
            agent.location = next_location
            agent.orientation = next_orientation

        # Get next observations
        gridmap_image = self._render_gridmap()

        observations = []
        for agent in self.agents:
            if self.obs == "vector":
                observation = self._get_observation_vector(agent)
            else:
                observation = self._get_observation(agent, gridmap_image)
            observations.append(observation)

        # Find who succeeded in hunting
        hunted_predator = None
        for predator in self.agents[1:]:
            if np.array_equal(self.agents[0].location, predator.location):
                hunted_predator = predator

        # Find nearby predators to the one succeeded in hunting
        nearby_predators = []
        if hunted_predator is not None:
            for predator in self.agents[1:]:
                if predator.id != hunted_predator.id:
                    dist = np.linalg.norm(predator.location - hunted_predator.location)
                    if dist < self.CAPTURE_RADIUS:
                        nearby_predators.append(predator)

        # Compute reward
        rewards = [0. for _ in range(len(self.agents))]
        if hunted_predator is not None:
            if len(nearby_predators) == 0:
                rewards[hunted_predator.id] = self.REWARD_LONELY
            else:
                rewards[hunted_predator.id] = self.REWARD_TEAM
                for neaby_predator in nearby_predators:
                    rewards[neaby_predator.id] = self.REWARD_TEAM

        # Compute done
        if hunted_predator is not None:
            done = [True] * len(self.agents)
        else:
            done = [False] * len(self.agents)

        self.current_step += 1
        if self.current_step >= self.ep_max_timesteps:
            truncated = [True] * len(self.agents)
        else:
            truncated = [False] * len(self.agents)
        info = [{} for _ in range(len(self.agents))]
        return observations, rewards, done, truncated, info
  
    def render(self, mode='human'):
        gridmap_image = self._render_gridmap()

        plt.figure(1)
        plt.clf()
        plt.imshow(gridmap_image)
        plt.axis('off')
        plt.pause(0.00001)

    def _load_gridmap_array(self):
        # Ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/envs/gridworld_env.py
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "maze.txt")
        with open(path, 'r') as f:
            gridmap = f.readlines()

        gridmap_array = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), gridmap)))
        return gridmap_array

    def _to_image(self, gridmap_array):
        image = np.zeros((gridmap_array.shape[0], gridmap_array.shape[1], 3), dtype=np.float32)

        for row in range(gridmap_array.shape[0]):
            for col in range(gridmap_array.shape[1]):
                grid = gridmap_array[row, col]

                if grid == self.config.grid_dict["empty"]:
                    image[row, col] = self.config.color_dict["empty"]
                elif grid == self.config.grid_dict["wall"]:
                    image[row, col] = self.config.color_dict["wall"]
                elif grid == self.config.grid_dict["prey"]:
                    image[row, col] = self.config.color_dict["prey"]
                elif grid == self.config.grid_dict["predator"]:
                    image[row, col] = self.config.color_dict["predator"]
                elif grid == self.config.grid_dict["orientation"]:
                    image[row, col] = self.config.color_dict["orientation"]
                else:
                    raise ValueError()

        return image

    def _render_gridmap(self):
        gridmap_image = np.copy(self.base_gridmap_image)

        # Render orientation
        for agent in self.agents:
            orientation_location = agent.orientation_location
            gridmap_image[orientation_location[0], orientation_location[1]] = self.config.color_dict["orientation"]

        # Render location
        for agent in self.agents:
            location = agent.location
            gridmap_image[location[0], location[1]] = self.config.color_dict[agent.type]

        # Pad image
        pad_width = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        gridmap_image = np.pad(gridmap_image, pad_width, mode="constant")

        return gridmap_image

    def _reset_agents(self):
        self.agents = []
        for i_agent, agent_type in enumerate(["prey"] + ["predator" for _ in range(self.n_predator)]):
            agent = Agent(i_agent, agent_type, self.base_gridmap_array)
            self.agents.append(agent)

    def _get_observation(self, agent, gridmap_image):
        """As in  Leibo et al., AAMAS-17 (https://arxiv.org/pdf/1702.03037.pdf),
        the observation depends on each player’s current position and orientation.
        Specifically, depending on the orientation, the image is cropped and then
        post-processed such that the player's location is always at the bottom center.
        """
        row, col = agent.location[0] + self.pad, agent.location[1] + self.pad
        height, half_width = self.observation_shape[0], int(self.observation_shape[1] / 2)

        if agent.orientation == self.config.orientation_dict["up"]:
            observation = gridmap_image[
                          row - height + 1: row + 1,
                          col - half_width: col + half_width + 1, :]
        elif agent.orientation == self.config.orientation_dict["right"]:
            observation = gridmap_image[
                          row - half_width: row + half_width + 1,
                          col: col + height, :]
            observation = np.rot90(observation, k=1)
        elif agent.orientation == self.config.orientation_dict["down"]:
            observation = gridmap_image[
                          row: row + height,
                          col - half_width: col + half_width + 1, :]
            observation = np.rot90(observation, k=2)
        elif agent.orientation == self.config.orientation_dict["left"]:
            observation = gridmap_image[
                          row - half_width: row + half_width + 1,
                          col - height + 1: col + 1, :]
            observation = np.rot90(observation, k=3)
        else:
            raise ValueError()

        assert observation.shape == self.observation_shape

        return observation

    def _get_observation_vector(self, agent):
        observation_vector = []
        agents_in_order = [agent] + [a for a in self.agents if a != agent]
        for agent in agents_in_order:
            observation_vector.append(agent.location[0])
            observation_vector.append(agent.location[1])
            observation_vector.append(agent.orientation)
        return observation_vector
