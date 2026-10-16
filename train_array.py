"""Train the meadow array policy from reward; no planner demonstrations or fallback."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from array_rl import ArrayDesignEnv, training_pool, VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='models/array_ppo')
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    torch.set_num_threads(1)
    pool = training_pool()
    check_env(ArrayDesignEnv(pool), warn=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: ArrayDesignEnv(pool), n_envs=8, seed=args.seed,
                       monitor_dir=str(output.parent/(output.name+'_logs')))
    model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', n_steps=256,
                batch_size=256, n_epochs=5, learning_rate=3e-4, gamma=.99,
                ent_coef=.01, policy_kwargs={'net_arch':[64,64]}, verbose=1)
    try:
        model.learn(total_timesteps=args.steps, log_interval=10)
        model.save(str(output))
        output.with_suffix('.json').write_text(json.dumps({
            'algorithm':'PPO', 'environment':'ArrayDesignEnv', 'version':VERSION,
            'training_seed':args.seed, 'actual_timesteps':model.num_timesteps,
            'scenario_seeds':[0,2047], 'observation_size':21, 'actions':21,
            'decisions_per_episode':3, 'network':[64,64],
            'reward':'2 * target coverage increase - 0.04 per rack + terminal success/shortfall',
        }, indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
