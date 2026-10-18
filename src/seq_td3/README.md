# Sequential TD3

This directory trains and evaluates TD3 over complete days with persistent taps.

The actor proposes continuous physical ratios in the 0.90–1.10 p.u. range.
The feeder projects each proposal to its discrete tap grid.
Replay stores the executed ratios returned by the physical solve, not the proposals.

Training uses 480 uniform-action interactions, then adds 0.01 p.u. Gaussian exploration noise to actor proposals.
Learning starts at interaction 481 with one twin-critic update per interaction and one actor and target update every two critic updates.
The actor and both critic branches use the current paper-aligned 400/300 hidden-width recipe.
The previous baseline used the author-code 256/256 recipe.
Gamma is 1.0, batch size is 256, replay capacity is 1,000,000, and tau is 0.005.
Target smoothing uses sigma 0.02 p.u. and a 0.05 p.u. clip.

Reward beta charges executed switching during training.
Benchmark beta prices switching in evaluation reports and may differ from reward beta.
Both effective values are recorded.
Use an explicit `--run` when they differ or when selecting a non-default reward.

```bash
uv run src/seq_td3/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5
uv run src/seq_td3/test.py --dataset ieee13-random --run b1e-05_seed0 --split validation
uv run src/seq_td3/train.py --dataset ieee13-random --seed 0 --days 1000 --beta 1e-5 --benchmark-beta 2e-5 --run reward1e-05_benchmark2e-05_seed0
```

Runs live under `results/<dataset>/seq_td3/<run>/`.
Each run writes `run.json`, `train.csv`, `counts.json`, `checkpoints/Actor_td3`, and `checkpoints/Critic_td3`.
`counts.json` records completed days, interactions, critic iterations, actor updates, and target updates.
Sequential evaluation writes `<split>.csv` and `<split>_hourly.csv` as a protected pair.
Training refuses an existing run unless `--overwrite` is explicit, and evaluation protects both output files in the same way.

The checkpoint files contain inference networks only.
They omit optimizers and replay and do not support exact training resumption.
