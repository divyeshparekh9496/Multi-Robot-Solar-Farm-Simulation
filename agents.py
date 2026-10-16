"""Random and shortest-path baselines; neither baseline learns."""
from collections import deque

import numpy as np

from solar_env import MOVES


class RandomAgent:
    def __init__(self, n_robots, seed=0):
        self.n_robots = n_robots
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return self.rng.integers(0, 6, self.n_robots), None


class GreedyAgent:
    """BFS routes with reserved destinations and adjacency-aware build sites."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        env = self.env
        occupied = set(env.robot_positions)
        reserved, targets, actions = set(), set(), []
        for i, start in enumerate(env.robot_positions):
            routes = {start: (0, 5)}
            todo = deque([start])
            while todo:
                cell = todo.popleft()
                distance, first = routes[cell]
                for action, (dr, dc) in MOVES.items():
                    dest = (cell[0] + dr, cell[1] + dc)
                    r, c = dest
                    if (0 <= r < env.grid_size and 0 <= c < env.grid_size
                            and not env.obstacle_map[dest] and dest not in routes
                            and dest not in occupied and dest not in reserved):
                        routes[dest] = (distance + 1, action if distance == 0 else first)
                        todo.append(dest)
            candidates = []
            for cell, (distance, first) in routes.items():
                if cell in targets:
                    continue
                if env.inventory[i] == 0:
                    if env.resource_map[cell]:
                        candidates.append((float(distance), cell, first))
                elif not env.panel_map[cell]:
                    # A new neighbor can increase both panels' production.
                    adjacent = sum(int(env.panel_map[p]) for p in env.neighbors(cell))
                    candidates.append((distance - 2.5 * adjacent, cell, 4 if distance == 0 else first))
            if candidates:
                _, target, action = min(candidates)
                targets.add(target)
            else:
                action = 5
            if action in MOVES:
                dr, dc = MOVES[action]
                reserved.add((start[0] + dr, start[1] + dc))
            actions.append(action)
        return np.array(actions, np.int64), None
