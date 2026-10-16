"""Paired evaluation on scenario seeds never used to construct the training pool."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from array_rl import ArrayDesignEnv, LearnedDesigner, scenario


def evaluate(model_path, episodes=200, start_seed=20000):
    designer = LearnedDesigner(model_path)
    records = []
    for seed in range(start_seed, start_seed+episodes):
        reference = scenario(seed)
        reference.autonomous = True
        reference.plan()
        feasible = reference.predicted+1e-5 >= reference.target
        for controller in ('ppo', 'random', 'fixed_south', 'planner'):
            farm = scenario(seed)
            farm.autonomous = True
            if controller == 'planner':
                farm.plan()
                power, panels = farm.predicted, len(farm.panels)
            else:
                env = ArrayDesignEnv()
                obs, _ = env.reset(options={'farm':farm})
                rng = np.random.default_rng(seed+100000)
                for _ in range(3):
                    if controller == 'ppo':
                        action, _ = designer.model.predict(obs, deterministic=True)
                        action = int(action)
                    elif controller == 'random':
                        action = int(rng.integers(21))
                    else:
                        action = 12  # south, 30-degree tilt, all three racks
                    obs, _, _, _, info = env.step(action)
                power, panels = info['power'], info['panels']
                env.close()
            records.append({'seed':seed, 'controller':controller,
                            'region':farm.region, 'solar_hour':farm.hour,
                            'target_w':farm.target, 'power_w':float(power), 'panels':panels,
                            'success':bool(power+1e-5 >= farm.target), 'feasible':bool(feasible)})
    summary = {}
    for controller in ('ppo', 'random', 'fixed_south', 'planner'):
        rows = [r for r in records if r['controller'] == controller]
        feasible = [r for r in rows if r['feasible']]
        summary[controller] = {
            'mean_power_w':float(np.mean([r['power_w'] for r in rows])),
            'std_power_w':float(np.std([r['power_w'] for r in rows])),
            'success_rate':float(np.mean([r['success'] for r in rows])),
            'success_rate_on_feasible_targets':float(np.mean([r['success'] for r in feasible])) if feasible else None,
            'feasible_targets':len(feasible),
            'mean_panels':float(np.mean([r['panels'] for r in rows])),
            'mean_target_coverage':float(np.mean([min(1,r['power_w']/r['target_w']) for r in rows])),
        }
    return records, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='checkpoints/array_ppo.zip')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--start-seed', type=int, default=20000)
    p.add_argument('--output', default='docs/array-ppo-evaluation')
    args = p.parse_args()
    if args.episodes < 1 or args.start_seed < 2048:
        p.error('Use at least one episode and held-out seeds >= 2048.')
    records, summary = evaluate(args.model, args.episodes, args.start_seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {'model':args.model, 'episodes':args.episodes,
               'held_out_seed_range':[args.start_seed,args.start_seed+args.episodes-1],
               'evaluation':'Deterministic PPO; same scenarios for every controller; design output before robot animation.',
               'summary':summary}
    output.with_suffix('.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
