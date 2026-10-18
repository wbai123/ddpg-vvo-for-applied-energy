# Static reference controllers

This directory evaluates neutral taps and native RegControl.
Both controllers use flattened hours from one named dataset split.

```bash
uv run src/reference/test.py --dataset ieee13-random --controller neutral --split test
uv run src/reference/test.py --dataset ieee13-random --controller regcontrol --split test
```

Neutral holds all taps at position 16.
RegControl selects taps through `solve_native_from_case`.
The protocol then scores those taps with a fixed-tap solve.

The command accepts `--dataset`, `--split`, `--days`, and `--overwrite`.
Each controller writes `results/<dataset>/reference/<controller>/<split>.csv`.
The file uses the shared static columns.
