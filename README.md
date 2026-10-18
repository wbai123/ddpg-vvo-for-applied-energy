# Learned regulator control on three feeders

This repository studies `ieee13`, `ieee37std`, and `ieee37mod`.
It contains one-shot DDPG, sequential DDPG, sequential TD3, and two reference controllers.

## Setup

```bash
uv sync
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

`uv.lock` pins the packages and DSS-Extensions engine.
Run every command from the repository root.

## Structure

| Location | Contents |
|---|---|
| `src/datasets/` | Deterministic named dataset generator. |
| `src/feeder/` | Named simulator, physical rules, load readers, and environments. |
| `src/common/` | Rewards, observations, metrics, shared notebook workflow and reports, evaluation, and run records. |
| `src/ddpg/` | One-shot DDPG experiment. |
| `src/reference/` | Neutral and native RegControl static tests. |
| `src/seq_ddpg/` | Sequential DDPG pilot on complete days. |
| `src/td3/` | Canonical continuous-action TD3 learner and checkpoint loader. |
| `src/seq_td3/` | Sequential TD3 training and evaluation on complete days. |
| `src/seq_reference/` | Neutral and native RegControl day tests. |
| `data/<feeder>/` | Three feeder cases with `feeder.toml` settings. |
| `data/scenarios/` | Named datasets with three splits and generation settings. |
| `experiments/<feeder>/` | One-shot and sequential notebooks. |
| `docs/` | Reproducibility instructions. |
| `papers/` | Research sources managed separately. |
| `tmp/` | Ignored scratch files. |

`FeederSimulator(feeder)` owns one DSS-Extensions context.
`solve` takes absolute loads and one position per controlled transformer.
`solve_native` takes absolute loads and the previous positions.
`solve_native_from_case` takes absolute loads only and lets the compiled case's controls choose.
All three operations return an independent `SolveResult` copy.

`OneShotEnv` makes one decision after a neutral reset.
`DayEnv` makes 24 decisions while loads advance and taps persist.
Both return physical states and outcomes; training applies rewards and observation functions from `src/common/`.
See [rewards and observations](docs/rewards_and_observations.md) for formulas, units, and run records.

## Dataset

Each named random dataset uses one multiplier per hourly row, repeated across its feeder's loads.
The generator creates deterministic train, validation, and test splits.

```bash
uv run src/datasets/generate.py --feeder ieee13 --dataset ieee13-random
uv run src/datasets/generate.py --feeder ieee37std --dataset ieee37std-random
uv run src/datasets/generate.py --feeder ieee37mod --dataset ieee37mod-random
```

Regenerating inputs requires researchers to retrain controllers and revalidate outputs.

## Commands

```bash
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000 --reward voltage_band --mad-weight 0
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split validation
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split test
uv run src/reference/test.py --dataset ieee13-random --controller neutral --split test
uv run src/reference/test.py --dataset ieee13-random --controller regcontrol --split test
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5 --reward voltage_band_switching
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split test
uv run src/seq_td3/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5 --reward voltage_band_switching
uv run src/seq_td3/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
uv run src/seq_reference/test.py --dataset ieee13-random --controller neutral regcontrol --beta 1e-5 --split test
```

Learned tests require compatible checkpoints.
The default seed set is 0, 1, and 2; run each training command once per seed.

Run outputs live under `results/<dataset>/<algorithm>/<run>/`.
Do not overwrite existing output silently.
The sequential default run name is `b<beta>_seed<seed>`.
Trainers refuse existing output unless `--overwrite` is passed.
