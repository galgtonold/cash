"""A helper made by a factory keys the array, frame or model it captured,
even when it calls a method on it or passes it to a function.

``score = make_scorer(np.array(...))`` whose body is ``weights.sum() * x``,
``predict = make_predictor(model)`` calling ``model.predict(x)``, or a helper
that passes its capture on (``np.sum(a)``): such a capture was left out of
the key, so a new array, frame or model -- in the next process, or rebound
or edited in place in this one -- was served the old result. The cached
function's own closure keyed the same captures all along, and so does a
helper's now, by the same rule: keyed, and watched by the drift guard, which
stops keying a capture once a call is seen to change it.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytest.importorskip("numpy")
pytest.importorskip("pandas")

FACTORIES = textwrap.dedent("""
    import numpy as np, pandas as pd

    def make_scorer(weights):
        def score(x):
            return float(weights.sum()) * x
        return score

    def make_summer(a):
        def score(x):
            return float(np.sum(a)) * x
        return score

    def make_lookup(table):
        def lookup(x):
            return float(table["a"].mean()) + x
        return lookup

    class Model:
        def __init__(self, k):
            self.k = k
        def predict(self, x):
            return x * self.k

    def make_predictor(model):
        def predict(x):
            return model.predict(x)
        return predict
""")

CROSS_PROCESS = FACTORIES + textwrap.dedent("""
    import sys
    import cash

    W = float(sys.argv[1])
    score = make_scorer(np.array([W, W]))
    summed = make_summer(np.array([W, W]))
    lookup = make_lookup(pd.DataFrame({"a": [W, W]}))
    predict = make_predictor(Model(W))

    @cash.cache
    def by_method(x):
        return score(x)

    @cash.cache
    def by_call(x):
        return summed(x)

    @cash.cache
    def by_frame(x):
        return lookup(x)

    @cash.cache
    def by_model(x):
        return predict(x)

    print(by_method(10), by_call(10), by_frame(10), by_model(10))
""")


def test_a_new_capture_in_the_next_process_recomputes(tmp_path):
    (tmp_path / "job.py").write_text(CROSS_PROCESS, encoding="utf-8")
    first = run_python("job.py", "1", cwd=tmp_path).stdout.strip()
    second = run_python("job.py", "2", cwd=tmp_path).stdout.strip()
    assert first == "20.0 20.0 11.0 10.0"
    assert second == "40.0 40.0 12.0 20.0"


IN_PROCESS = FACTORIES + textwrap.dedent("""
    import cash

    a = np.array([1.0, 1.0])
    score = make_scorer(a)
    predict = make_predictor(Model(1))

    @cash.cache
    def by_method(x):
        return score(x)

    @cash.cache
    def by_model(x):
        return predict(x)

    out = [by_method(10), by_model(10)]
    a[0] = 3.0                                   # edited in place
    out.append(by_method(10))
    score = make_scorer(np.zeros(2))             # rebound to a new helper
    out.append(by_method(10))
    predict = make_predictor(Model(5))
    out.append(by_model(10))
    out.append(by_method(10))                    # unchanged: a hit
    print(out, by_method.cache_info()["hits"])
""")


def test_a_capture_changed_in_this_process_recomputes(tmp_path):
    (tmp_path / "job.py").write_text(IN_PROCESS, encoding="utf-8")
    out = run_python("job.py", cwd=tmp_path).stdout.strip()
    assert out == "[20.0, 10, 40.0, 0.0, 50, 0.0] 1"


CHANGED_BY_THE_HELPER = textwrap.dedent("""
    import warnings
    import cash

    class Tally:
        def __init__(self):
            self.seen = []
        def note(self, x):
            self.seen.append(x)
            return x

    def make_counter(tally):
        def count(x):
            return tally.note(x) * 2
        return count

    count = make_counter(Tally())

    @cash.cache
    def f(x):
        return count(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = [f(3) for _ in range(4)]
    codes = sorted({getattr(w.message, "code", "") for w in caught} - {""})
    print(out, f.cache_info()["hits"], codes)
""")


def test_a_capture_the_helper_changes_is_dropped_after_one_call(tmp_path):
    # The helper changes what it captured through a method no rule names:
    # the drift guard sees it after the first call and stops keying it, so
    # later calls hit instead of missing for ever, and it says so.
    (tmp_path / "job.py").write_text(CHANGED_BY_THE_HELPER, encoding="utf-8")
    out = run_python("job.py", cwd=tmp_path).stdout.strip()
    assert out.startswith("[6, 6, 6, 6] ")
    # One extra miss: the entry the first call wrote is keyed on the state
    # before the call; the second call keys without the capture.
    assert int(out.split()[4]) == 2, out
    assert "IMPURE-SCOPE-MUTATION" in out
