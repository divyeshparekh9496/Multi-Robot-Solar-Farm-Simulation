import unittest
import tempfile
from pathlib import Path
import numpy as np
from stable_baselines3.common.env_checker import check_env
from array_rl import ArrayDesignEnv, LearnedDesigner, ROWS
from solar_target import Farm, Panel


class ArrayRLTests(unittest.TestCase):
    def test_gym_contract_and_seed_reproducibility(self):
        env = ArrayDesignEnv()
        check_env(env, warn=True)
        first, _ = env.reset(seed=123)
        farm = (env.farm.region, env.farm.hour, env.farm.target, dict(env.farm.obstacles))
        second, _ = env.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(farm, (env.farm.region, env.farm.hour, env.farm.target, env.farm.obstacles))

    def test_rl_reward_uses_same_physics_as_game(self):
        farm = Farm()
        farm.autonomous = True
        env = ArrayDesignEnv()
        env.reset(options={'farm': farm})
        for row, action in enumerate((12, 11, 10)):
            _, reward, done, truncated, info = env.step(action)
            for pos in ROWS[row]:
                farm.panels[pos] = Panel(row, action//5, action%5*15, True)
            self.assertAlmostEqual(info['power'], farm.power(), places=3)
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(done, row == 2)
            self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_skip_invalid_action_and_terminal_shape(self):
        env = ArrayDesignEnv()
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(21)
        for _ in range(3):
            obs, _, _, _, info = env.step(20)
        self.assertEqual(info['power'], 0)
        self.assertEqual(info['panels'], 0)
        self.assertTrue(env.observation_space.contains(obs))

    def test_saved_policy_is_loaded_and_robots_execute_its_design(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        env = ArrayDesignEnv()
        # A real serialization round trip; training quality is measured by the
        # separate held-out benchmark, not asserted by a unit test.
        model = PPO('MlpPolicy', env, n_steps=12, batch_size=12, n_epochs=1,
                    policy_kwargs={'net_arch':[16]}, seed=7, device='cpu')
        model.learn(24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy'
            model.save(path)
            designer = LearnedDesigner(path.with_suffix('.zip'))
            farm = Farm(designer=designer)
            farm.command('produce 1 kW')
            expected = farm.predicted
            for _ in range(200):
                farm.step()
                self.assertEqual(len(set(farm.robots)), 2)
            self.assertAlmostEqual(farm.power(), expected, places=3)
            self.assertTrue(all(p.built for p in farm.panels.values()))
            self.assertIn('PPO', farm.notice)
