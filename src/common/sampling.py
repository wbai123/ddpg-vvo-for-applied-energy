"""Generate reproducible training order independently of replay sampling."""

import numpy as np


def sampling_rng(seed):
    """Return the training-order stream derived from ``[seed, 1]``."""
    return np.random.default_rng([seed, 1])


def shuffled_passes(items, rng):
    """Yield every item once per shuffled pass until the caller's budget ends."""
    if not items:
        raise ValueError("cannot shuffle an empty sequence")
    while True:
        for index in rng.permutation(len(items)):
            yield items[int(index)]
