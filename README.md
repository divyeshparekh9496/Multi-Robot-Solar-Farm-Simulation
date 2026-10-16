# Solar Grove · Multi-Robot Solar Farm

An indie-style robot expedition and a trainable **Proximal Policy Optimization (PPO)** environment. Two robots collect salvage, spend materials on solar panels, and coordinate movement to maximize shared cumulative production.

![Solar Grove pixel-art interface, running the greedy baseline](docs/solar-grove.png)

The screenshot shows the **greedy baseline**, not PPO performance. The game works immediately with that baseline; PPO requires training a checkpoint first. Energy is an abstract game unit, not a physical solar-yield estimate.

## Play

Python **3.10+** recommended (validated with Python 3.12).

```bash
git clone https://github.com/divyeshparekh9496/Multi-Robot-Solar-Farm-Simulation.git
cd Multi-Robot-Solar-Farm-Simulation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python game.py
```

`requirements-lock.txt` records the versions used for validation; use `pip install -r requirements-lock.txt` to reproduce that dependency set on a supported platform.

The interface uses procedurally drawn pixel terrain, animated robots, salvage crates, solar arrays, and a live expedition HUD. No external sprites or API keys are required.

| Control | Action |
| --- | --- |
| Space | Pause / resume |
| R | Restart the same seeded map |
| N | Start the next map |
| Tab | Cycle available controllers (PPO appears when launched with a checkpoint) |
| + / - | Change simulation speed |
| E | Enter a fresh-map editor / resume the edited map |
| 1 / 2 / 3, then left-click | Paint rock / salvage / erase in the editor |
| Esc | Exit |

Editing starts a new episode and keeps the robot count fixed. Robot tiles cannot be painted. Custom editor maps may be disconnected and are for experimentation; benchmark maps are generated with connected terrain. Episodes stop at their time limit so their results remain visible.

## Machine learning: centralized PPO

One neural policy observes the full map and selects **one action for each robot**. Stable-Baselines3 supplies the actual PPO implementation, backed by PyTorch; this is not a random agent labeled as reinforcement learning.

PPO collects trajectories, estimates advantages, and optimizes a clipped policy objective alongside a value function. It suits this sequential task because building a panel earlier can pay off for many later ticks. It supports the joint `MultiDiscrete` action space without enumerating a separate action for every possible joint combination.

This is **centralized control with a shared reward**, not decentralized multi-agent PPO (MAPPO). Both robots' actions are produced from the same global state.

| RL component | Implementation |
| --- | --- |
| Environment | Headless `gymnasium.Env` in `solar_env.py` |
| Observation | Obstacle, salvage and panel maps; one occupancy map per robot; normalized inventories; remaining time |
| Shape | 323 float32 values in `[0, 1]` for the default 8×8 map and two robots |
| Actions | `MultiDiscrete([6, 6])`: north, east, south, west, build, wait for each robot |
| Policy | PPO `MlpPolicy`, two hidden layers of 128 units |
| Reward | `0.1 × production − 0.02 × successful moves − 0.05 × invalid actions` per tick |
| Episode | 300 ticks; time-limit truncation, no early terminal state |
| Training | Four vectorized environments; seeded training; separate evaluation stream and best-checkpoint callback |

Collection is automatic when a robot has capacity. Each robot holds at most two materials; each panel consumes one. A panel yields one energy unit per tick, or two if it has at least one orthogonal panel neighbor. Building is not allowed on existing panels. Total return in the HUD is exactly the sum of learning rewards; raw cumulative energy is reported separately.

Movement into a robot's starting cell is blocked for that tick, including swaps and following into a vacated cell. If both robots choose the same empty destination, the lower robot index wins. These deliberately conservative rules prevent overlapping robots. Reset places distinct robots, salvage and obstacles on connected terrain using the environment's seeded RNG.

### Train and load PPO

```bash
python train.py --steps 500000 --seed 42 --output models/solar_ppo
python game.py --agent ppo --model models/solar_ppo.zip
```

Training writes the final model, configuration metadata, episode-monitor CSVs and periodic evaluation records. The best evaluation checkpoint is saved under `models/solar_ppo_logs/best/best_model.zip`; you can load or evaluate it independently. Evaluation runs during training use a separate random stream starting at seed 10000. Final benchmark episodes explicitly reset to paired seeds starting at 20000.

Key settings: learning rate `3e-4`, rollout length `512` per environment, batch size `128`, discount `0.99`, GAE lambda `0.95`, clip range `0.2`, entropy coefficient `0.01`. Requested training steps round up to complete rollouts; metadata records the actual count. Checkpoints and training logs are local generated artifacts, excluded from Git.

For a quick integration check rather than a performance experiment:

```bash
python train.py --steps 2048 --output models/smoke
python game.py --agent ppo --model models/smoke.zip
```

**A working training loop does not imply a strong learned policy.** The flattened-map policy and delayed construction rewards can make generalization difficult. Measure held-out performance before claiming that PPO improves coordination or beats a planner. No pretrained model is bundled.

## Compare against baselines

```bash
# Baselines only; no checkpoint required
python evaluate.py --episodes 30

# Random, greedy, and PPO on exactly the same 30 map seeds
python evaluate.py --model models/solar_ppo.zip --episodes 30 --output runs/comparison
```

- **Random:** independently samples six actions per robot from a seeded RNG.
- **Greedy planner:** breadth-first shortest paths, resource-target assignment, reserved movement destinations, and adjacency-aware panel placement. It does not learn.
- **PPO:** deterministic inference from the supplied checkpoint, with no planner fallback.

Evaluation exports per-episode CSV data and JSON summaries with population mean and standard deviation for energy, production, panels, return and invalid actions. All controllers share the same information and environment rules. Single-checkpoint results do not establish robustness across training seeds; repeat training seeds for stronger claims.

## Verify and preview

```bash
python -m unittest discover -s tests -v
python game.py --screenshot docs/solar-grove.png --frames 22
```

Tests cover the Gymnasium contract, reproducible connected resets, material conservation, collision rules, inventory capacity, rewards, time limits, fixed observations, editor controls and offscreen rendering. The screenshot command uses SDL's offscreen driver and exits automatically.

## Repository layout

```text
solar_env.py              Environment, production, shared rewards and observations
agents.py                 Random and greedy planning baselines
game.py                   Indie-style Pygame viewer and scenario editor
train.py                  PPO learning, evaluation callback and checkpoint saving
evaluate.py               Paired held-out evaluations and CSV/JSON export
tests/                    Environment and interface regression tests
docs/solar-grove.png      Actual game screenshot using the greedy controller
requirements-lock.txt     Validated dependency versions
```

## Scope and next steps

This is a grid-based construction benchmark. It does not model robot dynamics, sunlight, shading, battery discharge, real watts, weather or hardware control. Adjacency bonuses are an explicit game mechanic. Useful next experiments include convolutional observation encoders, potential-based reward shaping, collision-fairness alternatives, and curriculum training. MAPPO would be a separate extension for decentralized policies.

References: [PPO paper](https://arxiv.org/abs/1707.06347), [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/).

## Recorded validation run

A local run with training seed **42** completed **501,760 timesteps** (the requested 500,000 rounded to complete rollouts). Deterministic final-checkpoint inference was evaluated on **30 paired maps, seeds 20000–20029**:

| Controller | Mean cumulative energy ± std | Mean panels | Mean episode return |
| --- | ---: | ---: | ---: |
| Random | 3196.8 ± 749.7 | 10.37 | 304.28 |
| Greedy planner | 6648.4 ± 270.5 | 12.00 | 663.85 |
| PPO, final checkpoint | 842.5 ± 778.7 | 2.13 | 60.59 |

**PPO underperformed both baselines in this experiment.** This validates training, saving, loading and evaluation, but does not demonstrate a successful learned controller. Frequent invalid actions and poor transfer to new maps are unresolved limitations. The greedy controller remains the default for a useful interactive demo. The periodic best checkpoint was not evaluated in this table. Values are game units, and this is one training seed, not a statistically robust algorithm comparison.

Raw records: [episode CSV](docs/ppo-evaluation.csv), [summary JSON](docs/ppo-evaluation.json). The trained checkpoint is a local artifact at `models/solar_ppo.zip` and is not included in the repository. Re-run training with the locked dependencies to conduct your own experiment; exact numerical reproduction can vary by platform.
