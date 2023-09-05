from pettingzoo_wolfpack.env_agents.base_agent import BaseAgent
import numpy as np


class HunterAgent(BaseAgent):

    def __init__(self, heuristic="H1", gridmap="../maps/maze.txt"):
        super().__init__(gridmap)
        self.name = "hunter"
        self.heuristic = heuristic

    def step(self, observation) -> int:
        own_pos = observation[0:2]
        prey_pos = observation[3:5]
        if self.heuristic == "H1":
            patrol_points = np.asarray([[2, 2], [2, 18], [18, 2], [18, 18]])
            move = self._patrol_strategy(own_pos, prey_pos, patrol_points)
        elif self.heuristic == "H2":
            move = self._limit_prey_options(own_pos, prey_pos)
        elif self.heuristic == "H3":
            zone_min = np.asarray([0, 0])
            zone_max = np.asarray([10, 10])
            move = self._zone_control(own_pos, prey_pos, zone_min, zone_max)
        elif self.heuristic == "H4":
            move = self._noise_based_decision(own_pos, prey_pos)
        elif self.heuristic == "H5":
            move = self.best_move(own_pos, [prey_pos])
        else:
            raise ValueError("Invalid heuristic.")
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

    def distance_to_prey(self, pos, prey_pos):
        """
        Returns the distance from the given position to the prey.
        """
        return np.sqrt((prey_pos[0] - pos[0]) ** 2 + (prey_pos[1] - pos[1]) ** 2)

    def best_move(self, my_pos, prey_positions):
        """
        Returns the best move to minimize distance to the prey.
        """
        moves = self.valid_moves(my_pos)
        if not moves:  # No valid moves
            return 0

        # Distance to the prey for each possible move
        distances = {action: self.distance_to_prey(new_pos, prey_positions[0]) for action, new_pos in moves.items()}

        # Get the move with the minimum distance to the prey
        best_action = min(distances, key=distances.get)

        return best_action

    def _patrol_strategy(self, my_pos, prey_pos, patrol_points, detect_radius=5):
        distance_to_prey = np.linalg.norm(my_pos - prey_pos)

        # If prey is within detection radius, pursue directly.
        if distance_to_prey < detect_radius:
            return self.best_move(my_pos, [prey_pos])

        # Otherwise, move to the next patrol point.
        next_patrol_point = patrol_points[0]
        min_distance = np.linalg.norm(my_pos - next_patrol_point)
        for point in patrol_points[1:]:
            distance = np.linalg.norm(my_pos - point)
            if distance < min_distance:
                min_distance = distance
                next_patrol_point = point

        return self.best_move(my_pos, [next_patrol_point])

    def _limit_prey_options(self, my_pos, prey_pos):
        prey_moves = self.valid_moves(prey_pos)

        # If prey has only one valid move, chase directly.
        if len(prey_moves) <= 1:
            return self.best_move(my_pos, [prey_pos])

        # Try to block one of the prey's valid moves.
        for _, block_pos in prey_moves.items():
            if np.linalg.norm(my_pos - block_pos) < 2: # Within striking distance to block.
                return self.best_move(my_pos, [block_pos])

        # Default to direct pursuit.
        return self.best_move(my_pos, [prey_pos])

    def _zone_control(self, my_pos, prey_pos, zone_min, zone_max):
        if zone_min <= prey_pos[0] <= zone_max[0] and zone_min <= prey_pos[1] <= zone_max[1]:
            # If prey is within zone, use a basic strategy like direct pursuit.
            return self.best_move(my_pos, [prey_pos])
        else:
            # Find the nearest point in the zone to the prey.
            target = np.array([
                min(max(prey_pos[0], zone_min[0]), zone_max[0]),
                min(max(prey_pos[1], zone_min[1]), zone_max[1])
            ])
            return self.best_move(my_pos, [target])

    def _noise_based_decision(self, my_pos, prey_pos, epsilon=0.2):
        if np.random.rand() < epsilon:
            # Choose a random move
            valid_moves = list(self.valid_moves(my_pos).keys())
            return np.random.choice(valid_moves)
        else:
            # Choose the best move
            return self.best_move(my_pos, [prey_pos])

