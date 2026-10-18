# Rewards, observations, and run records

The physical simulator returns voltages in p.u., tap positions, tap ratios, and convergence.
`OneShotEnv` and `DayEnv` advance the physical experiment and return `PhysicalState` and `HourOutcome`.
They do not choose a learning reward.

## Change a learning reward

Edit the complete named formula in `src/common/rewards.py`.
Each trainer's `REWARDS` table holds the selected function and its complete effective parameter defaults.
The trainer copies those parameters, applies CLI overrides, and passes the same dictionary to training and `run.json`.
Give each new formula variant a new reward name and table entry.
The one-shot runner selects a reward through `--reward voltage_band` and accepts `--mad-weight`.
The sequential runner selects `--reward voltage_band_switching` and accepts `--beta`.
The selected function receives the executed outcome; it does not change the physical solve.

```python
from common.rewards import voltage_band_reward

reward = voltage_band_reward(outcome.result.voltages, mad_weight=0.0)
```

Training multiplies cost by 1000 in the reward formula.
The default mean absolute deviation weight is zero.
Sequential switching reward also charges `beta` for executed tap steps.
Use a distinct run name for a changed reward configuration.

## Change an observation

Edit `voltage_observation` or `day_observation` in `src/common/observations.py`.
Size networks with `voltage_observation_size(simulator)` or `day_observation_size(simulator)` for the encoding in use.
When an encoding changes, update its size function in `common.observations`.
The physical environments contain no observation-size logic.
The training runner calls the selected function after each environment reset and step.

```python
from common.observations import voltage_observation

state = env.reset(scenario)
observation = voltage_observation(state)
```

Physical voltages are p.u.; voltage network features are `(V - 1) / 0.1` in float32.
Day features append normalized held tap positions, then sine and cosine of the current hour.
The action proposal is a physical tap ratio in p.u.
Execution projects it to an integer tap position; replay stores the executed ratio.

## Compare runs

`src/common/metrics.py` computes fixed operational voltage cost, violations, and feasibility.
`src/common/evaluation.py` uses these metrics for learned and reference controllers.
The daily benchmark adds `beta` times executed tap steps to the voltage cost.
These comparisons do not depend on the selected training reward.
Sequential training accepts `--benchmark-beta`; it defaults to the reward beta and records the effective value.
Choose an explicit `--run` when the benchmark beta differs from the reward beta.
Sequential testing uses the recorded benchmark beta by default and writes `<split>.csv` and `<split>_hourly.csv`.
Its `--benchmark-beta` override writes `<split>_b<beta>.csv` and `<split>_b<beta>_hourly.csv`.
The `<beta>` text follows Python's `str(beta)` spelling.

New `run.json` files have `task` and `agent` groups.
`task` records feeder, dataset, algorithm, observation, action representation, reward, and evaluation settings.
`agent.layer_sizes` records the widths needed to reconstruct a checkpoint.
`agent.settings` is optional for algorithms whose records predate complete recipes.
Sequential TD3 requires it and records random streams, optimizer rates, replay behavior, update schedule, noise in normalized and physical units, action mapping, initialization, training budget, and inference-only checkpoint support.
`src/ddpg/checkpoints.py` validates the record and tensor shapes before evaluation writes output.
`src/td3/checkpoints.py` applies the corresponding checks to `seq_td3` records and twin-critic checkpoints.
The loader also accepts exact historical records with the verified historical widths and reward definitions.
`common.runs.resolve_run_settings` is the shared source for current and historical saved settings.
It resolves task metadata and network widths without loading checkpoints or importing PyTorch.
