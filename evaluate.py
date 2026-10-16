"""Paired evaluation on identical map seeds; exports raw per-episode metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from agents import GreedyAgent, RandomAgent
from solar_env import MultiRobotSolarFarmEnv


def evaluate(episodes, start_seed, model_path=None):
    model = None
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
    rows = []
    for name in ["random", "greedy"] + (["ppo"] if model else []):
        env = MultiRobotSolarFarmEnv()
        for seed in range(start_seed, start_seed + episodes):
            obs, _ = env.reset(seed=seed)
            agent = (RandomAgent(env.n_robots, seed) if name == "random"
                     else GreedyAgent(env) if name == "greedy" else model)
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rows.append({"agent": name, "seed": seed, **info})
        env.close()
    summary = {}
    for name in sorted({row["agent"] for row in rows}):
        subset = [r for r in rows if r["agent"] == name]
        summary[name] = {key: {"mean": float(np.mean([r[key] for r in subset])),
                               "std": float(np.std([r[key] for r in subset]))}
                         for key in ("energy", "production", "panels", "score", "invalid_actions")}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional PPO checkpoint; omitted means baselines only")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=20_000)
    parser.add_argument("--output", default="runs/evaluation")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    rows, summary = evaluate(args.episodes, args.start_seed, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    model_metadata = None
    if args.model:
        model_path = Path(args.model)
        if model_path.suffix != ".zip":
            model_path = Path(str(model_path) + ".zip")
        model_metadata = {"sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        config_path = model_path.with_suffix(".json")
        if config_path.exists():
            model_metadata["training"] = json.loads(config_path.read_text())
    output.with_suffix(".json").write_text(json.dumps({
        "episodes": args.episodes, "start_seed": args.start_seed,
        "model": args.model, "model_metadata": model_metadata, "summary": summary,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
