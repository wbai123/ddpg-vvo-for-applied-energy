# Sequential reference controllers

These controllers use the shared 24-hour environment.
Days start from neutral taps.

```bash
uv run src/seq_reference/test.py \
  --dataset ieee13-random --controller neutral regcontrol \
  --beta 1e-6 1e-5 1e-4 --split test
```

Neutral holds all taps at position 16.
RegControl acts from the positions held at each hour.
The day protocol records the native solve directly.

The command accepts `--dataset`, `--days`, `--split`, and `--overwrite`.
The `--controller` and `--beta` flags each accept several values.
It writes `results/<dataset>/seq_reference/<controller>_b<beta>/<split>.csv`.
Each row records day, beta, voltage cost, tap steps, daily cost, and feasible hours.
It also writes `<split>_hourly.csv` beside the daily file.
Hourly columns are:

`day,hour,beta,k1_before,k2_before,k3_before,k1,k2,k3,tap_steps,voltage_cost,switching_cost,hourly_cost,vmin,vmax,viol_cnt,feasible,converged`
