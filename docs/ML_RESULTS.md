# Solar Grove: learned photovoltaic array design

The playable meadow mode uses **PPO**, a model-free reinforcement-learning technique, to choose array deployment and orientation. It is a centralized high-level controller: a neural policy makes three sequential rack decisions, and two robots execute the result through deterministic BFS navigation. Robot locomotion and task allocation are not learned; no claim of MAPPO, real robot autonomy, real solar-yield prediction or standards certification is made.

## Experiment

`ArrayDesignEnv` is a Gymnasium environment with 21 observations and 21 discrete actions per rack (skip or four compass directions × five tilt angles). A three-decision episode receives incremental target-coverage reward, material cost and terminal target feedback. Power comes from the same equinox illumination and tree-shadow model used in the game. Each four-module string only connects when fully built.

Training used Stable-Baselines3 PPO, an actor-critic MLP with two 64-unit layers, eight vector environments, learning rate 0.0003, rollout length 256 per worker, batch size 256, five optimization epochs, entropy coefficient 0.01, discount 0.99, seed 42, and **200,704 actual environment steps**. The training pool comprises scenario seeds 0–2047. Scenario generation varies region, local solar hour (7–17), production target (300–1200 W) and tree heights (1–4 abstract units).

The policy is learned exclusively from reward. The observation contains sun and shade information; it does not contain planner actions or an optimal direction label. There is no planner fallback inside policy inference. The finite shadow model and discrete action set are explicit simulation assumptions.

## Paired held-out results

200 scenarios, seeds 20000–20199, were evaluated with deterministic PPO inference. The same scenarios were used for all controllers. The planner can meet 114 of the 200 targets; 86 exceed its available capacity under the sampled shadows and sunlight. These feasibility labels come from the comparison planner, not from the PPO output.

| Controller | Overall target success | Success on 114 feasible targets | Mean power | Mean panels |
|---|---:|---:|---:|---:|
| PPO | 53.5% | 93.9% (107/114) | 767.3 W | 12.00 |
| Random direction/tilt/skip | 23.5% | 41.2% | 441.9 W | 11.52 |
| Fixed south / 30° / all racks | 27.5% | 48.2% | 514.2 W | 12.00 |
| Deterministic planner | 57.0% | 100.0% | 685.2 W | 7.92 |

PPO outperforms the random and fixed-orientation baselines in this experiment. The planner has better target success and material efficiency. Its lower mean power reflects its policy of stopping at the minimum rack count meeting a target; it is not evidence that PPO is a better optimizer.

The learned policy deploys all three racks on these evaluation cases. Although target and material cost are present during learning, efficient target-conditioned deployment has not emerged in this checkpoint. Stronger claims would require multiple independent training seeds, uncertainty estimates, further held-out environments, and improved material-efficiency results. The evaluation compares completed design output before navigation animation; build latency is not a learned objective or benchmark metric.

## Reproduce

Use `requirements-lock.txt` for the validated dependency versions. The bundled model is `checkpoints/array_ppo.zip`; `checkpoints/array_ppo.json` records the training metadata, package versions and SHA-256. Raw paired records, summaries and an aggregated monitor learning trace are adjacent to this document.

```bash
python train_array.py --steps 200000 --seed 42 --output models/array_ppo
python evaluate_array.py --model models/array_ppo.zip --episodes 200 --output runs/array-comparison
python game.py --designer ppo --array-model models/array_ppo.zip
python -m unittest discover -s tests -v
```

Exact trained parameters may vary across platforms. The included evaluation belongs to the bundled checkpoint hash; retraining produces a new artifact and requires a new evaluation.

## Accurate resume wording

Developed a goal-conditioned PPO policy in PyTorch/Stable-Baselines3 for simulated solar-array deployment, integrating sunlight and shadow observations with a two-robot construction game; achieved 93.9% target success on 114 feasible scenarios within a 200-scenario held-out benchmark, compared with 41.2% for a random policy, while retaining deterministic navigation for execution.

## References

- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [NOAA solar geometry](https://gml.noaa.gov/grad/solcalc/solareqns.PDF)
- [Sandia array series/parallel principles](https://pvpmc.sandia.gov/modeling-guide/3-dc-array-iv/)
