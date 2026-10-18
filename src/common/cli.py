"""Validate shared command-line identities, numeric domains, and result paths."""

import argparse
import math
import pathlib
import re
import sys

from feeder.loads import dataset_dir

MAX_SEED = 4294967295
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class ArgumentParser(argparse.ArgumentParser):
    """Print concise command-line errors to stderr and exit with status two."""

    def error(self, message):
        print(message, file=sys.stderr)
        raise SystemExit(2)


def dataset_name(value):
    """Validate and return one dataset name."""
    try:
        dataset_dir(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def run_name(value):
    """Validate and return one run name."""
    if RUN_NAME.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"invalid run name: {value!r}")
    return value


def seed_value(value):
    """Validate and return one random seed."""
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be an integer in 0..{MAX_SEED}")
    return seed


def positive_integer(value):
    """Validate and return one positive integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer of at least 1") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be an integer of at least 1")
    return number


def beta_value(value):
    """Validate and return one finite non-negative beta."""
    try:
        beta = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("beta must be a finite non-negative float") from error
    if not math.isfinite(beta) or beta < 0:
        raise argparse.ArgumentTypeError("beta must be a finite non-negative float")
    return beta


def result_directory(root, dataset, algorithm, name=None):
    """Resolve a result directory without allowing a run name to escape its parent."""
    base = (pathlib.Path(root) / "results" / dataset / algorithm).resolve()
    if name is None:
        return base
    checked = run_name(name)
    target = (base / checked).resolve()
    if target.parent != base:
        raise ValueError(f"unsafe result directory: {target}")
    return target


def display_path(path, root):
    """Return a repository-relative path for command output."""
    return pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
