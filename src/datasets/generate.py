"""Generate deterministic named daily multiplier datasets.

The command initializes the feeder to obtain its load order, then writes train,
validation, and test CSV files plus generation metadata. It does not solve the
generated scenarios or create absolute load values.
"""

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

from feeder.loads import dataset_dir
from feeder.simulator import FeederSimulator

MAX_SEED = 4294967295
SPLIT_STREAMS = (("train", 0), ("validation", 1), ("test", 2))


def random_multipliers(rng, hour=None, *, low, high, count):
    """Draw one uniform multiplier and repeat it in compiled load order."""
    del hour
    value = float(rng.uniform(low, high))
    return [value] * count


PATTERNS = {"random": random_multipliers}


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if converted < minimum or maximum is not None and converted > maximum:
        domain = f"{minimum}..{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be {domain}")
    return converted


def _bounds(low, high):
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError) as error:
        raise ValueError("low and high must be finite numbers with 0 < low < high") from error
    if (
        not math.isfinite(low_value)
        or not math.isfinite(high_value)
        or not 0 < low_value < high_value
    ):
        raise ValueError("low and high must be finite numbers with 0 < low < high")
    return low_value, high_value


def _validated(pattern, seed, train_days, validation_days, test_days, low, high):
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}")
    return (
        _integer(seed, "seed", 0, MAX_SEED),
        _integer(train_days, "train", 1),
        _integer(validation_days, "validation", 1),
        _integer(test_days, "test", 1),
        *_bounds(low, high),
    )


def _write_split(path, split, days, pattern, seed, low, high, load_names):
    """Write complete days from the split's independent random stream."""
    rng = np.random.default_rng([seed, dict(SPLIT_STREAMS)[split]])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("day_id", "hour", *load_names))
        for day_index in range(days):
            day_id = f"{split}-{day_index:04d}"
            for hour in range(24):
                factors = PATTERNS[pattern](
                    rng, hour=hour, low=low, high=high, count=len(load_names)
                )
                writer.writerow((day_id, hour, *(f"{factor:.6f}" for factor in factors)))


def generate_dataset(
    directory,
    *,
    feeder="ieee13",
    pattern,
    seed,
    train_days,
    validation_days,
    test_days,
    low,
    high,
):
    """Write three deterministic day splits and their generation metadata.

    Each CSV begins with day and hour, followed by multipliers in the compiled
    feeder load order. Split-specific random streams keep validation and test
    draws unchanged when another split's size changes.
    """
    seed, train_days, validation_days, test_days, low, high = _validated(
        pattern, seed, train_days, validation_days, test_days, low, high
    )
    load_names = FeederSimulator(feeder).load_names
    output = pathlib.Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train": train_days, "validation": validation_days, "test": test_days}
    for split, _stream in SPLIT_STREAMS:
        _write_split(
            output / f"{split}.csv", split, counts[split], pattern, seed, low, high, load_names
        )
    metadata = {
        "feeder": feeder,
        "pattern": pattern,
        "seed": seed,
        "train_days": train_days,
        "validation_days": validation_days,
        "test_days": test_days,
        "low": low,
        "high": high,
        "decimals": 6,
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args():
    """Parse dataset identity, generation bounds, split sizes, and overwrite policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feeder", choices=("ieee13", "ieee37std", "ieee37mod"), default="ieee13")
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--seed", default="12345")
    parser.add_argument("--train", default="1000")
    parser.add_argument("--validation", default="200")
    parser.add_argument("--test", default="200")
    parser.add_argument("--low", default="0.5")
    parser.add_argument("--high", default="1.5")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Validate all settings before writing or replacing a named dataset."""
    args = parse_args()
    display = pathlib.Path("data/scenarios") / args.dataset
    try:
        output = dataset_dir(args.dataset)
        values = _validated(
            args.pattern,
            args.seed,
            args.train,
            args.validation,
            args.test,
            args.low,
            args.high,
        )
        if output.exists() and not args.overwrite:
            raise ValueError(f"{display} exists; use --overwrite to replace its generated files")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    generate_dataset(
        output,
        feeder=args.feeder,
        pattern=args.pattern,
        seed=values[0],
        train_days=values[1],
        validation_days=values[2],
        test_days=values[3],
        low=values[4],
        high=values[5],
    )
    if args.overwrite:
        print("previous inputs were replaced; retrain and revalidate affected results")
    print(f"wrote {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
