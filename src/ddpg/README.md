# Corrected one-shot DDPG

This experiment makes one tap decision after a neutral static reset.

The actor observes 41 normalized live-node voltages.
It proposes three ratios on `[0.90, 1.10]`.
The runner projects them onto the 33-position grid.
Replay stores the executed grid ratios.

Training reads flattened hours from the named training split.
Training order uses `np.random.default_rng([seed, 1])`.
Each shuffled pass visits every training hour once.
The episode budget may stop inside a shuffled pass.

```bash
uv run src/ddpg/train.py --dataset ieee13-random --seed 0 --episodes 20000
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split validation
uv run src/ddpg/test.py --dataset ieee13-random --run seed0 --split test
```

Training accepts `--dataset`, `--seed`, `--episodes`, `--run`, and `--overwrite`.
The default seed set is 0, 1, and 2, one run per seed.
Training refuses existing output unless `--overwrite` is passed.
Training does not resume runs.

Testing requires `results/<dataset>/ddpg/<run>/checkpoints/Actor_ddpg`.
The checkpoint directory also contains `Critic_ddpg`.
It writes `results/<dataset>/ddpg/<run>/<split>.csv`.

The controller has no persistent device state.
Its one-step discount is zero.
Evaluation is deterministic.
