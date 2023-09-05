from pettingzoo_wolfpack.env_agents.base_agent import BaseAgent
import numpy as np


class PreyAgentH1(BaseAgent):

    def __init__(self, gridmap="../maps/maze.txt"):
        super().__init__(gridmap)
        self.name = "prey_h1"

    def step(self, observation) -> int:
        own_pos = observation[0:2]
        other_pos = [observation[i*3:i*3+2] for i in range(1, len(observation)//3) if len(observation[i*3:i*3+2]) == 2]
        move = self.best_move(own_pos, other_pos)
        return move

    def valid_moves(self, pos):
        """
        Returns a list of valid moves from the current position.
        """
        valid_actions = {}
        for action, vector in self.action_dict.items():
            # For movement actions
            if isinstance(vector, np.ndarray):
                new_pos = tuple(pos + vector)
                if self.gridmap_array.shape[0] > new_pos[0] >= 0 == self.gridmap_array[new_pos] and \
                        0 <= new_pos[1] < self.gridmap_array.shape[1]:
                    valid_actions[action] = new_pos

        return valid_actions

    def sum_distances(self, pos, agents):
        """
        Returns the sum of the distances from the given position to all agents.
        """
        return sum(np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2) for x, y in agents)

    def best_move(self, my_pos, agents):
        """
        Returns the best move to maximize distance from other agents.
        """
        moves = self.valid_moves(my_pos)

        # If there are no valid moves, stay in place.
        if not moves:
            return 0

        distances = {action: self.sum_distances(new_pos, agents) for action, new_pos in moves.items()}

        # Get the move with the maximum summed distance.
        best_action = max(distances, key=distances.get)

        return best_action
