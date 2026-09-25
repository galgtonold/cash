"""User code held inside an object is keyed by its code, as a global and as an argument.

Only the argument path walked an object thoroughly for the code it holds, and
it stopped at a library object. So these were served the old result after an
edit, with no warning:

* a function an instance global holds (``RUNNER = Runner(scale)``) and an
  instance inside a tuple inside a list global (``STEPS = [("m", Model())]``);
* a user transformer or function inside an sklearn pipeline, passed as an
  argument or read as a global.

Each case runs in fresh processes on one cache: first run, an unedited run
that must hit, then an edit that must recompute.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    # Saved a while before the run, like an ordinary edit.
    past = time.time() - 30
    os.utime(path, (past, past))


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.split(), p.stderr.count("[RUN]")


def _edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    _write(path, text.replace(old, new))


APP = """
    from feats import scale

    class Runner:
        def __init__(self, fn):
            self.fn = fn
        def run(self, x):
            return self.fn(x)

    class Model:
        def predict(self, x):
            return x * 10

    RUNNER = Runner(scale)
    STEPS = [("m", Model())]
"""

JOB = """
    import sys
    import cash
    from app import RUNNER, STEPS

    @cash.cache
    def a(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return RUNNER.run(x)

    @cash.cache
    def b(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return STEPS[0][1].predict(x)

    print(a(2), b(2))
"""


def test_code_inside_a_global_object_reaches_the_key(tmp_path):
    _write(tmp_path / "feats.py", "def scale(x):\n    return x * 10\n")
    _write(tmp_path / "app.py", APP)
    _write(tmp_path / "job.py", JOB)
    assert _run(tmp_path) == (["20", "20"], 2)
    assert _run(tmp_path) == (["20", "20"], 0), "the unedited run did not hit"
    _edit(tmp_path / "feats.py", "x * 10", "x * 11")
    assert _run(tmp_path) == (["22", "20"], 1), "an edit to the function an instance holds was served stale"
    _edit(tmp_path / "app.py", "return x * 10", "return x * 11")
    assert _run(tmp_path) == (["22", "22"], 1), "an edit to a class nested in a list global was served stale"


FEATS = """
    from sklearn.base import BaseEstimator, TransformerMixin

    class Scale(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.fitted_ = True
            return self
        def transform(self, X):
            return X * 10

    def double(X):
        return X * 2
"""

PIPE = """
    import sys
    import numpy as np
    import cash
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    from feats import Scale, double

    PIPE = make_pipeline(Scale(), FunctionTransformer(double)).fit([[0.0]])

    @cash.cache
    def as_argument(pipe, x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return float(pipe.transform(np.array([[x]]))[0, 0])

    @cash.cache
    def as_global(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return float(PIPE.transform(np.array([[x]]))[0, 0])

    print(as_argument(PIPE, 2.0), as_global(2.0))
"""


def test_user_code_inside_a_library_pipeline_reaches_the_key(tmp_path):
    pytest.importorskip("sklearn")
    _write(tmp_path / "feats.py", FEATS)
    _write(tmp_path / "job.py", PIPE)
    assert _run(tmp_path) == (["40.0", "40.0"], 2)
    assert _run(tmp_path) == (["40.0", "40.0"], 0), "the unedited run did not hit"
    _edit(tmp_path / "feats.py", "X * 10", "X * 11")
    assert _run(tmp_path) == (["44.0", "44.0"], 2), "an edit to a transformer inside the pipeline was served stale"
    _edit(tmp_path / "feats.py", "X * 2", "X * 3")
    assert _run(tmp_path) == (["66.0", "66.0"], 2), "an edit to a function inside the pipeline was served stale"
