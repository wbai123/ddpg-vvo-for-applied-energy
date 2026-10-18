# Reproducibility

This file defines the environment, commands, inputs, and output directories.

## Environment

Use Linux, macOS, or WSL2 with Python 3.14.
`uv.lock` pins every package and the DSS-Extensions engine.

```bash
uv sync
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

The fixed thread count preserves seeded reduction order.
`PYTHONUNBUFFERED=1` displays progress immediately.

## Commands

```bash
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward voltage_band --mad-weight 0
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split validation
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split test
uv run src/reference/test.py --dataset ieee13-random --controller neutral --split test
uv run src/reference/test.py --dataset ieee13-random --controller regcontrol --split test
uv run src/seq_reference/test.py --dataset ieee13-random --controller neutral regcontrol --beta 1e-5 --split test
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5 --reward voltage_band_switching
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split test
uv run src/seq_td3/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5 --reward voltage_band_switching
uv run src/seq_td3/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
```

The default seed set is 0, 1, and 2.
Run each training and learned test command once per seed.

## Inputs

The three feeder inputs live under `data/<feeder>/` beside `feeder.toml`.
Named inputs live under `data/scenarios/<dataset>/`.
Each dataset uses one uniform multiplier per hourly row, repeated across its loads.
Each CSV header starts with `day_id,hour,` and continues with the compiled load order.
`generation.json` records the feeder and generation settings.
Each split uses `np.random.default_rng([seed, k])`, with a distinct `k`.
Training order uses `np.random.default_rng([seed, 1])`.
Each shuffled pass visits every available item once.
The requested training budget may stop inside a shuffled pass.
Regenerated inputs require researchers to retrain controllers and revalidate outputs.

`loads.py` converts multiplier rows to absolute kW and kvar.
`FeederSimulator` receives absolute load arrays in its frozen `load_names` order.

## Outputs

All outputs live under `results/<dataset>/<algorithm>/<run>/`.
Trainers write `train.csv`, `run.json`, and `checkpoints/`.
Sequential TD3 also writes `counts.json` with environment interactions and critic, actor, and target update counts.
Its inference-only checkpoints are `checkpoints/Actor_td3` and `checkpoints/Critic_td3`.
New `run.json` files record task and agent settings, including reward, observation, action units, and network widths.
Historical runs use shared settings resolution in `common.runs`; see `docs/rewards_and_observations.md`.
Static evaluators write `validation.csv` or `test.csv`.
Sequential evaluators also write `validation_hourly.csv` or `test_hourly.csv`.
Sequential evaluate and view modes reject a requested beta that differs from a saved benchmark beta.

Static evaluation columns start with `scenario,k1..kn` and end with physical metrics.
Day evaluation columns are `day,beta,voltage_cost,tap_steps,daily_cost,feasible_hours`.
Hourly evaluation columns are:

`day,hour,beta,k1_before..kn_before,k1..kn,tap_steps,voltage_cost,switching_cost,hourly_cost,vmin,vmax,viol_cnt,feasible,converged`

Do not overwrite existing output silently.
Trainers refuse existing runs unless `--overwrite` is passed.
Trainers do not resume runs.

## History-free solves

Each fixed-tap solve writes every load and tap before solving.
Snapshot mode removes dependence on earlier solution paths.
Native static and native sequential operations retain their distinct initialization conventions.
