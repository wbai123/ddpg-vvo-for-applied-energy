# Named dataset generator

The default pattern draws one uniform multiplier per hourly row, repeated across all 15 loads.
Each CSV begins with `day_id,hour,` followed by the fifteen load names.

```bash
uv run src/datasets/generate.py --dataset ieee13-random
```

Each split uses `np.random.default_rng([seed, k])` with its assigned `k`.
Changing one split's day count leaves the other split bytes unchanged.
The default seed is `12345`.
The default range is 0.5 through 1.5.

Use `--overwrite` only when replacing generated inputs intentionally.
Regenerated inputs require researchers to retrain controllers and revalidate outputs.
The generator preserves unrelated files and every other dataset.
