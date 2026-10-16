"""Goal-conditioned PPO environment for the production challenge."""
from pathlib import Path
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from solar_target import Farm, Panel, REGIONS, VECTORS, sun_vector

ROWS = tuple(tuple((x, y) for x in range(3, 7)) for y in (3, 5, 7))
VERSION = 1


def scenario(seed):
    rng = np.random.default_rng(seed)
    farm = Farm()
    farm.region = int(rng.integers(4))
    farm.hour = float(rng.uniform(7, 17))
    farm.target = float(rng.uniform(300, 1200))
    farm.obstacles = {pos: float(rng.uniform(1, 4)) for pos in farm.obstacles}
    return farm


class ArrayDesignEnv(gym.Env):
    """Three decisions: skip a rack, or choose one of 20 direction/tilt pairs.

    Robots execute the learned design using deterministic navigation afterwards.
    Training reward uses the same ideal array production as the rendered game.
    """
    metadata = {'render_modes': []}

    def __init__(self, pool=None):
        super().__init__()
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(-1, 1, (21,), np.float32)
        self.pool = pool
        self.farm = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'farm' in options:
            self.farm = options['farm']
            self.sun = np.asarray(sun_vector(REGIONS[self.farm.region][1], self.farm.hour), dtype=np.float32)
            self.light = np.asarray([[not self.farm.shadowed(p) for p in row] for row in ROWS], dtype=np.float32)
        elif self.pool is not None:
            template, sun, light = self.pool[int(self.np_random.integers(len(self.pool)))]
            self.farm, self.sun, self.light = template, sun, light
        else:
            self.farm = scenario(int(self.np_random.integers(0, 2**31)))
            self.sun = np.asarray(sun_vector(REGIONS[self.farm.region][1], self.farm.hour), dtype=np.float32)
            self.light = np.asarray([[not self.farm.shadowed(p) for p in row] for row in ROWS], dtype=np.float32)
        self.index, self.watts, self.count = 0, 0.0, 0
        self.design = []
        return self.observation(), {}

    def observation(self):
        stage = np.zeros(3, dtype=np.float32)
        if self.index < 3:
            stage[self.index] = 1
        return np.concatenate((self.sun, self.light.ravel(),
                               [min(1, self.farm.target/1200)], stage,
                               [min(1, self.watts/1200), self.count/3])).astype(np.float32)

    def step(self, action):
        if self.index >= 3:
            raise RuntimeError('Episode finished; reset before stepping again.')
        if not self.action_space.contains(action):
            raise ValueError('Action must be an integer from 0 to 20.')
        action = int(action)
        before = min(1, self.watts/self.farm.target)
        cost = 0.0
        if action < 20:
            facing, tilt = action//5, (action%5)*15
            dx, dy = VECTORS[facing]
            angle = math.radians(tilt)
            east, south, up = self.sun
            incidence = max(0, dx*math.sin(angle)*east+dy*math.sin(angle)*south+math.cos(angle)*up)
            output = 400*incidence*max(0, float(up))**.15*float(self.light[self.index].min())
            self.watts += output
            self.count += 1
            self.design.append((self.index, facing, tilt))
            cost = .04
        self.index += 1
        terminated = self.index == 3
        reward = 2*(min(1, self.watts/self.farm.target)-before)-cost
        if terminated:
            reward += .5 if self.watts+1e-5 >= self.farm.target else -.5*(1-min(1, self.watts/self.farm.target))
        return self.observation(), float(reward), terminated, False, {
            'power': float(self.watts), 'target': self.farm.target,
            'panels': self.count*4, 'success': bool(self.watts+1e-5 >= self.farm.target)}


def training_pool(size=2048):
    pool = []
    for seed in range(size):
        farm = scenario(seed)
        sun = np.asarray(sun_vector(REGIONS[farm.region][1], farm.hour), dtype=np.float32)
        light = np.asarray([[not farm.shadowed(p) for p in row] for row in ROWS], dtype=np.float32)
        pool.append((farm, sun, light))
    return pool


class LearnedDesigner:
    def __init__(self, path='checkpoints/array_ppo.zip'):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        self.model = PPO.load(str(Path(path)), device='cpu')
        if self.model.observation_space.shape != (21,) or self.model.action_space.n != 21:
            raise ValueError('Checkpoint does not match the production-challenge observation and action spaces.')

    def apply(self, farm):
        env = ArrayDesignEnv()
        obs, _ = env.reset(options={'farm': farm})
        for _ in range(3):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(int(action))
        previous = farm.panels
        farm.panels = {}
        for group, (row, facing, tilt) in enumerate(env.design):
            for pos in ROWS[row]:
                old = previous.get(pos)
                built = bool(old and old.built and (old.group, old.facing, old.tilt) == (group, facing, tilt))
                farm.panels[pos] = Panel(group, facing, tilt, built)
        farm.jobs, farm.progress = [None, None], [0, 0]
        farm.predicted = float(env.watts)
        farm.notice = f'PPO chose {len(env.design)} racks. Expected {env.watts:.0f} W. Robots deploying.'
        if env.watts < farm.target:
            farm.notice = f'PPO design predicts {env.watts:.0f} W for {farm.target:.0f} W target.'
        env.close()
