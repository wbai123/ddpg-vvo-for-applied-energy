# Sequential DDPG

This pilot applies corrected DDPG over complete 24-hour days.

Each observation contains 41 normalized voltages, three scaled tap positions, and two clock features.
Tap positions persist between hours and start neutral.
The reward is negative 1,000 times voltage cost plus beta-weighted tap steps.

Training order uses `np.random.default_rng([seed, 1])`.
Each shuffled pass visits every training day once.
The day budget may stop inside a shuffled pass.

```bash
uv run src/seq_ddpg/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
uv run src/seq_ddpg/test.py --dataset ieee13-random --run b1e-05_seed0 --split test
```

Training accepts `--dataset`, `--seed`, `--days`, `--beta`, `--run`, and `--overwrite`.
The default seed set is 0, 1, and 2, one run per seed and beta.
Testing requires `--dataset` and `--run` and reads beta from `train.csv`.
Training refuses existing output unless `--overwrite` is passed.
Training does not resume runs.

Training starts with 20 warmup days of uniform actions.
After warmup, uniform actions retain a 0.1 share.
The discount is 1.0.
Training saves checkpoints every 200 days and at the end.
The default run name is `b<beta>_seed<seed>`.
The recipe constants live at the top of `src/seq_ddpg/train.py`.
The `train` function owns the reusable day loop, while `main` handles command-line setup.

Runs live under `results/<dataset>/seq_ddpg/<run>/`.
Each run contains training progress, checkpoints, and optional daily and hourly evaluation files.
Evaluation writes `<split>.csv` and `<split>_hourly.csv`.
Hourly columns are:

`day,hour,beta,k1_before,k2_before,k3_before,k1,k2,k3,tap_steps,voltage_cost,switching_cost,hourly_cost,vmin,vmax,viol_cnt,feasible,converged`

The actor sees no future loads.
Sequential DDPG and sequential TD3 own separate learner, training, and checkpoint code.
The two implementations evolve independently while using the same physical environment, observations, rewards, evaluation metrics, and notebook workflow.
