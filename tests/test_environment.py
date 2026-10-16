import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from agents import GreedyAgent
from solar_env import MultiRobotSolarFarmEnv


class EnvironmentTests(unittest.TestCase):
    def blank(self):
        env = MultiRobotSolarFarmEnv()
        env.obstacle_map[:] = 0
        env.resource_map[:] = 0
        env.panel_map[:] = 0
        env.robot_positions = [(2, 2), (5, 5)]
        return env

    def test_gym_contract(self):
        check_env(MultiRobotSolarFarmEnv(), warn=True)

    def test_reset_is_reproducible_connected_and_disjoint(self):
        env = MultiRobotSolarFarmEnv()
        for seed in range(50):
            a, _ = env.reset(seed=seed)
            positions = list(env.robot_positions)
            b, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(positions, env.robot_positions)
            self.assertEqual(len(set(positions)), 2)
            self.assertEqual(env.resource_map.sum(), env.n_resources)
            self.assertFalse(np.any(env.resource_map & env.obstacle_map))
            for cell in positions:
                self.assertFalse(env.obstacle_map[cell] or env.resource_map[cell])
            free = {tuple(cell) for cell in np.argwhere(env.obstacle_map == 0)}
            self.assertTrue(env._connected(free))

    def test_build_requires_material_and_energy_matches_reward(self):
        env = self.blank()
        _, reward, _, _, _ = env.step([4, 5])
        self.assertEqual(env.panel_map.sum(), 0)
        self.assertAlmostEqual(reward, -0.05)
        env.resource_map[2, 3] = 1
        env.step([1, 5])
        self.assertEqual(env.inventory[0], 1)
        self.assertEqual(env.resource_map.sum(), 0)
        _, reward, _, _, _ = env.step([4, 5])
        self.assertEqual(env.inventory[0], 0)
        self.assertEqual(env.production(), 1)
        self.assertAlmostEqual(reward, 0.1)
        env.panel_map[2, 4] = 1
        _, reward, _, _, _ = env.step([5, 5])
        self.assertEqual(env.production(), 4)
        self.assertAlmostEqual(reward, 0.4)

    def test_collision_swap_obstacle_and_bounds(self):
        env = self.blank()
        env.robot_positions = [(2, 2), (2, 4)]
        env.step([1, 3])
        self.assertEqual(env.robot_positions, [(2, 3), (2, 4)])
        env.step([1, 3])
        self.assertEqual(env.robot_positions, [(2, 3), (2, 4)])
        env.robot_positions = [(0, 0), (5, 5)]
        env.step([0, 5])
        self.assertEqual(env.robot_positions[0], (0, 0))
        env.obstacle_map[0, 1] = 1
        env.step([1, 5])
        self.assertEqual(env.robot_positions[0], (0, 0))

    def test_capacity_and_time_limit(self):
        env = self.blank()
        env.inventory[0] = env.capacity
        env.resource_map[2, 3] = 1
        env.step([1, 5])
        self.assertEqual(env.resource_map[2, 3], 1)
        env.step_count = env.max_steps - 1
        obs, _, terminated, truncated, _ = env.step([5, 5])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(env.observation_space.contains(obs))
        with self.assertRaises(RuntimeError):
            env.step([5, 5])

    def test_material_conservation_and_fixed_shape(self):
        env = MultiRobotSolarFarmEnv()
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            agent = GreedyAgent(env)
            for _ in range(env.max_steps):
                action, _ = agent.predict(obs)
                obs, _, _, _, _ = env.step(action)
                self.assertEqual(env.resource_map.sum() + env.inventory.sum() +
                                 env.panel_map.sum(), env.n_resources)
                self.assertEqual(len(set(env.robot_positions)), env.n_robots)
                self.assertTrue(env.observation_space.contains(obs))
            self.assertGreater(env.panel_map.sum(), 0)

    def test_invalid_joint_action_rejected(self):
        env = self.blank()
        for actions in ([0], [0, 6], [0.2, 1]):
            with self.assertRaises(ValueError):
                env.step(actions)


if __name__ == '__main__':
    unittest.main()
