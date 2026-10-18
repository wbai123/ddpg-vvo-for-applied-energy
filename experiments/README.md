# Experiment notebooks

Start each kernel in its notebook's own directory.
Use the project environment or an existing kernel with the pinned dependencies.
The notebooks import project modules from `src/`; they install no packages.

`ieee13/`, `ieee37std/`, and `ieee37mod/` each contain one-shot DDPG, sequential DDPG, and sequential TD3 notebooks.
The sequential TD3 notebook is `seq_td3.ipynb` in each feeder directory.
Each notebook names its feeder and dataset in its parameter cell.

Edit the parameter cell, restart the kernel, and run all cells.
`MODE = "train"` trains and evaluates fresh runs.
`MODE = "evaluate"` evaluates existing checkpoints, while `MODE = "view"` reads saved results without launching subprocesses.
`OVERWRITE` permits writing over selected outputs in `train` and `evaluate` modes.
The former `TRAIN = True` maps to `MODE = "train"`.
The former `TRAIN = False` with evaluation maps to `MODE = "evaluate"`; use `view` to display saved results.
The default seed set is `[0, 1, 2]`, and `SPLIT` defaults to `validation`.
`EVALUATION_DAYS = None` evaluates the full split.
One-shot notebooks expose `REWARD` and `MAD_WEIGHT` in the parameter cell.
Sequential notebooks expose `REWARD` and `BETA`, with 1,000 days and beta `1e-5` by default.
The workflow selects sequential DDPG with `"algorithm": "ddpg"` and sequential TD3 with `"algorithm": "td3"`.
TD3 is unavailable for the one-shot protocol.
The selected reward applies during training; evaluation uses fixed operational metrics.
`SHOW_HOURLY_DETAILS` controls the optional selected-day table.

Sequential workflow preflight validates each reference output pair before launching a command.
It reuses every complete compatible Neutral or RegControl pair, even when `OVERWRITE = True` replaces learned outputs.
It evaluates only missing reference controllers.
Run the sequential reference evaluator directly with `--overwrite` to deliberately recompute compatible references.

Make sure existing outputs are protected after a run by setting `OVERWRITE = False`.

## Writing a new notebook

- Use notebooks as research control tools and feedback loops. Keep stable implementation in source modules.
- Reuse existing simulation, algorithm, reward, observation, execution, and reporting code.
- Keep substantial experiment-specific code in one notebook when only that notebook uses it. Do not extract code for length alone.
- Move behavior into a descriptive source function when two notebooks need it. Update both callers instead of copying cells.
- Parameters, short explanations, and short calls may repeat. Refactor behavior that would otherwise need synchronized fixes.
- Separate execution from presentation. Plot edits and example selection must not retrain or reevaluate.
- Select feeder, algorithm, protocol, and settings through the workflow. Keep algorithm training in its algorithm directory.
- Keep one-off scientific choices readable and editable. Avoid generalizing them before another use appears.
- Follow the established layout, kernel setup, naming, and concise style. Do not install packages inside notebooks.
- Keep reproducible outputs embedded in executed notebooks. Store durable CSV and checkpoint outputs in the results layout, without HTML exports.
- Split a report function with a large one-shot block and a large sequential block into `_oneshot` and `_seq` implementations behind a short dispatcher.
- Keep small protocol differences followed by shared work in one function.

A one-shot notebook can call the shared workflow directly:

```python
from common.workflow import run_experiment

results = run_experiment({
    "protocol": "one_shot",
    "algorithm": "ddpg",
    "feeder": "ieee13",
    "dataset": "ieee13-random",
    "seeds": [0, 1, 2],
    "mode": "view",
    "overwrite": False,
    "episodes": 20000,
    "split": "validation",
    "evaluation_days": None,
    "reward": "voltage_band",
    "mad_weight": 0.0,
    "run_prefix": "seed",
    "root": ROOT,
})
```
