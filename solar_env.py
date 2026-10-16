"""Deterministic, headless Gymnasium task for centralized robot coordination."""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = ("NORTH", "EAST", "SOUTH", "WEST", "BUILD", "WAIT")


class MultiRobotSolarFarmEnv(gym.Env):
    """Joint MultiDiscrete actions; global normalized observation; shared reward.

    Moves into any cell occupied at the start of a tick are blocked, including
    swaps. Competing destinations go to the lower robot index. Resource pickup
    is automatic; each panel costs one material. Energy units are fictional.
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size=8, n_robots=2, max_steps=300, n_resources=12):
        super().__init__()
        if grid_size < 4 or not 1 <= n_robots <= 4 or max_steps < 1:
            raise ValueError("Use grid_size >= 4, 1..4 robots, and positive max_steps")
        if not 1 <= n_resources <= grid_size * grid_size - n_robots - grid_size:
            raise ValueError("Resource count must fit the grid")
        self.grid_size, self.n_robots = grid_size, n_robots
        self.max_steps, self.n_resources = max_steps, n_resources
        self.capacity = 2
        self.action_space = spaces.MultiDiscrete(np.full(n_robots, 6))
        # Obstacle/resource/panel maps, one position map per robot, inventories,
        # and remaining episode time. Shape is fixed throughout an episode.
        length = (3 + n_robots) * grid_size**2 + n_robots + 1
        self.observation_space = spaces.Box(0, 1, (length,), np.float32)
        self.reset()

    def neighbors(self, cell):
        for dr, dc in MOVES.values():
            r, c = cell[0] + dr, cell[1] + dc
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                yield (r, c)

    def _connected(self, free):
        seen = {next(iter(free))}
        todo = deque(seen)
        while todo:
            for cell in self.neighbors(todo.popleft()):
                if cell in free and cell not in seen:
                    seen.add(cell)
                    todo.append(cell)
        return len(seen) == len(free)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self.grid_size
        cells = [(r, c) for r in range(n) for c in range(n)]
        # Only retain an obstacle if all remaining terrain stays connected.
        free, obstacles = set(cells), []
        for idx in self.np_random.permutation(len(cells)):
            cell = cells[idx]
            candidate = free - {cell}
            if self._connected(candidate):
                free = candidate
                obstacles.append(cell)
            if len(obstacles) == n:
                break
        self.obstacle_map = np.zeros((n, n), np.int8)
        self.resource_map = np.zeros((n, n), np.int8)
        self.panel_map = np.zeros((n, n), np.int8)
        for cell in obstacles:
            self.obstacle_map[cell] = 1
        available = sorted(free)
        picks = self.np_random.choice(len(available), self.n_robots + self.n_resources, replace=False)
        self.robot_positions = [available[i] for i in picks[:self.n_robots]]
        for i in picks[self.n_robots:]:
            self.resource_map[available[i]] = 1
        self.inventory = np.zeros(self.n_robots, np.int8)
        self.step_count, self.score, self.total_energy = 0, 0.0, 0.0
        self.invalid_actions, self.collected = 0, 0
        self.last_actions = [5] * self.n_robots
        self.events = ["New expedition. Gather salvage and build a solar grove."]
        self.finished = False
        return self._get_observation(), self._info()

    def production(self):
        return sum(1 + int(any(self.panel_map[p] for p in self.neighbors(tuple(cell))))
                   for cell in np.argwhere(self.panel_map))

    def step(self, actions):
        if self.finished:
            raise RuntimeError("Episode finished; call reset before step")
        actions = np.asarray(actions)
        if not self.action_space.contains(actions):
            raise ValueError(f"Expected {self.n_robots} integer actions in 0..5")
        self.step_count += 1
        self.last_actions = actions.tolist()
        occupied = set(self.robot_positions)
        claimed = set()
        invalid, moves = 0, 0
        self.events = []
        for i, action in enumerate(actions):
            cell = self.robot_positions[i]
            if action in MOVES:
                dr, dc = MOVES[action]
                dest = (cell[0] + dr, cell[1] + dc)
                r, c = dest
                if (not (0 <= r < self.grid_size and 0 <= c < self.grid_size)
                        or self.obstacle_map[dest] or dest in occupied or dest in claimed):
                    invalid += 1
                else:
                    self.robot_positions[i] = dest
                    claimed.add(dest)
                    moves += 1
            elif action == 4:
                if self.inventory[i] > 0 and not self.panel_map[cell] and not self.obstacle_map[cell]:
                    self.panel_map[cell] = 1
                    self.inventory[i] -= 1
                    self.events.append(f"BOT {i+1} planted a solar panel.")
                else:
                    invalid += 1
            cell = self.robot_positions[i]
            if self.resource_map[cell] and self.inventory[i] < self.capacity:
                self.resource_map[cell] = 0
                self.inventory[i] += 1
                self.collected += 1
                self.events.append(f"BOT {i+1} collected one material.")
        energy = self.production()
        self.total_energy += energy
        # No collection bonus: materials are useful only when converted to panels.
        reward = float(0.1 * energy - 0.02 * moves - 0.05 * invalid)
        self.score += reward
        self.invalid_actions += invalid
        self.finished = self.step_count >= self.max_steps
        return self._get_observation(), reward, False, self.finished, self._info()

    def _get_observation(self):
        robot_maps = np.zeros((self.n_robots, self.grid_size, self.grid_size), np.float32)
        for i, cell in enumerate(self.robot_positions):
            robot_maps[i, cell[0], cell[1]] = 1
        maps = np.concatenate([self.obstacle_map.ravel(), self.resource_map.ravel(),
                               self.panel_map.ravel(), robot_maps.ravel()])
        return np.concatenate([maps, self.inventory / self.capacity,
                               [1 - self.step_count / self.max_steps]]).astype(np.float32)

    def _info(self):
        return {"energy": float(self.total_energy), "production": int(self.production()),
                "panels": int(self.panel_map.sum()), "collected": self.collected,
                "invalid_actions": self.invalid_actions, "score": float(self.score)}
