"""Train centralized PPO; checkpoints are compatible with game.py --agent ppo."""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from solar_env import MultiRobotSolarFarmEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/solar_ppo")
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.envs < 1:
        parser.error("steps and envs must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.parent / (output.name + "_logs")
    run_dir.mkdir(exist_ok=True)
    check_env(MultiRobotSolarFarmEnv(), warn=True)
    env = make_vec_env(MultiRobotSolarFarmEnv, n_envs=args.envs, seed=args.seed,
                       monitor_dir=str(run_dir / "train"))
    evaluation = Monitor(MultiRobotSolarFarmEnv())
    evaluation.reset(seed=10_000)
    callback = EvalCallback(evaluation, best_model_save_path=str(run_dir / "best"),
                            log_path=str(run_dir), eval_freq=max(10_000 // args.envs, 1),
                            n_eval_episodes=5, deterministic=True)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", n_steps=512,
                batch_size=128, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, policy_kwargs={"net_arch": [128, 128]},
                verbose=1)
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(str(output))
        output.with_suffix(".json").write_text(json.dumps({
            "algorithm": "PPO", "seed": args.seed, "timesteps": model.num_timesteps,
            "grid_size": 8, "n_robots": 2, "max_steps": 300,
            "n_resources": 12, "environment_version": 1,
        }, indent=2) + "\n")
        print(f"Saved {output}.zip; evaluate.py measures performance on held-out maps.")
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
