# Shared feeder package

`FeederSimulator(feeder)` selects `ieee13`, `ieee37std`, or `ieee37mod`.
Each simulator owns one DSS-Extensions context and reads its case and `feeder.toml`.
The compiled case supplies load powers, load order, transformer tap metadata, and native targets.

`_dss.py` isolates engine calls. `config.py` reads the small feeder record.
`simulator.py` freezes measurement identities and validates each solve.
`taps.py` and `bands.py` define the common physical action and voltage metrics.
`loads.py` reads named datasets and checks their feeder identity.
`envs.py`, `evaluation.py`, and `day_evaluation.py` implement the shared protocols.
`runs.py` records the feeder and network dimensions for learned checkpoints.

IEEE-13 uses its initial live-node line-ground measurements in the original order.
IEEE-37 uses AB, BC, and CA complex line-line differences at every three-phase bus except `sourcebus`.
Each pair uses its bus's line-line base, including the 0.48 kV bus.

One-shot observations contain only voltage features.
Day observations add the previous tap positions and two hour features.
The action count follows the controlled transformer list in each `feeder.toml`.

Run package checks with:

```bash
uv run pytest tests/test_simulator.py tests/test_environments.py -q
```
