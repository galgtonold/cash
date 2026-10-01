"""User code held in a numpy object array, a pandas object column or a
polars Object column is keyed by its code.

The code search walked lists, dicts and attributes, but an array or a frame
keeps its objects in its buffer or blocks, which nothing walked, and the
content hash pickles such an object by reference. Editing the class of a
model stored in ``np.array([Model(1)], dtype=object)`` served the old result.

Each case runs in fresh processes on one cache: first run, an unedited run
that must hit, then an edit that must recompute.
"""

from __future__ import annotations

import os
import textwrap
import time

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

pytest.importorskip("pandas")


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(proj):
    p = run_python("job.py", cwd=proj)
    return p.stdout.split(), p.stderr.count("[RUN]")


HELPER = """
    class Model:
        def predict(self, x):
            return x * 1

    def double(x):
        return x * 1
"""

JOB = """
    import sys
    import numpy as np
    import pandas as pd
    import cash
    from helper import Model, double

    GLOBAL = np.array([Model()], dtype=object)

    @cash.cache
    def use(holder):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        first = holder["m"].iloc[0] if isinstance(holder, pd.DataFrame) else holder[0]
        return first.predict(10) if hasattr(first, "predict") else first(10)

    @cash.cache
    def use_global(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return GLOBAL[0].predict(x)

    values = [
        np.array([Model()], dtype=object),
        pd.Series([Model()]),
        pd.DataFrame({"m": [Model()], "name": ["a"]}),
        np.array([double], dtype=object),
        pd.DataFrame({"m": [double]}),
    ]
    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None:
        values.append(pl.Series([Model()], dtype=pl.Object))
    print(*[use(v) for v in values], use_global(10))
"""


def test_editing_code_in_an_object_array_recomputes(tmp_path):
    _write(tmp_path / "helper.py", HELPER)
    _write(tmp_path / "job.py", JOB)
    first, runs = _run(tmp_path)
    assert set(first) == {"10"} and runs == len(first)
    again, runs = _run(tmp_path)
    assert again == first and runs == 0
    _write(tmp_path / "helper.py", HELPER.replace("x * 1", "x * 2"))
    edited, runs = _run(tmp_path)
    assert set(edited) == {"20"}, edited
    assert runs == len(first)
