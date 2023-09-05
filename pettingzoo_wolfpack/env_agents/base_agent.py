from abc import abstractmethod
import os
import numpy as np


class BaseAgent:

    def __init__(self, gridmap="../maps/maze.txt"):
        self.gridmap = gridmap
        self.gridmap_array = self._load_gridmap_array(gridmap)
        self.action_dict = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
            5: +1,
            6: -1,
        }

    @abstractmethod
    def step(self, observation) -> int:
        pass

    def _load_gridmap_array(self, gridmap):
        # Ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/envs/gridworld_env.py
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), gridmap)
        with open(path, 'r') as f:
            gridmap = f.readlines()

        gridmap_array = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), gridmap)))
        return gridmap_array
