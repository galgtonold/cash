"""Functions the speed set decorates (see ``_speed_scenarios.py``).

A real module rather than functions built inside the scenarios: cash reads a
decorated function's source and folds the globals it reads, so the targets
have to look like the functions users write, in a file.

Each body is called plain as well, for the ratio's denominator.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

RATE = 0.5
COLUMNS = ["id", "user", "amount"]


def small_dict(n):
    """A config/summary-sized result: ten keys, flat values."""
    return {f"key{i}": i * RATE + n for i in range(10)}


def nested_records(n):
    """Records as JSON or an API gives them: a list of dicts holding lists and dicts."""
    return [{"id": i + n, "user": f"u{i % 50}", "tags": [i, i + 1], "meta": {"ok": i % 2 == 0}} for i in range(2000)]


def numpy_array(n):
    """A 1M-element float array (8 MB)."""
    return np.arange(1_000_000, dtype=np.float64) * RATE + n


def pandas_frame(n):
    """A 50k-row frame with numeric and string columns."""
    k = 50_000
    return pd.DataFrame(
        {
            "id": np.arange(k) + n,
            "user": [f"u{i % 100}" for i in range(k)],
            "amount": np.arange(k, dtype=np.float64) * RATE,
        }
    )


@dataclasses.dataclass
class Point:
    x: float
    y: float
    label: str


def dataclass_list(n):
    """A list of 2000 dataclass instances."""
    return [Point(i * RATE, i + n, f"p{i % 7}") for i in range(2000)]


def keyed_args(arr, frame, config, names):
    """Typical arguments, light body: the key work dominates a hit."""
    return float(arr[:1000].sum()) + len(frame) * config["lr"] + len(names)
