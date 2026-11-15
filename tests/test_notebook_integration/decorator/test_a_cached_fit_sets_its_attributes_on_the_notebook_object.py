"""A cached fit that sets attributes on an object a notebook cell made.

``s = Scaler()`` in a cell of its own is tagged by the notebook, and
``@cash.cache`` keys ``s`` by that tag. The tag does not move while the
call runs, so the check after the body hashed ``s`` alike before and after
``s.mean_ = ...``: the call was stored as if it changed nothing, and a hit
(re-run, or Run All again) handed back the stored copy, leaving ``s``
without ``mean_`` and ``r is s`` False. Plain Python gives ``s.mean_`` and
``r is s``: a hit sets the attributes on the caller's object and hands that
object back. These tests hold the notebook to the plain run.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_DEFS = (
    "import time, numpy as np, cash\n"
    "class Scaler:\n"
    "    def __init__(self, k=2):\n"
    "        self.k = k\n"
    "    def fit(self, X):\n"
    "        time.sleep(0.05)\n"
    "        self.mean_ = float(np.mean(X)) * self.k\n"
    "        return self\n"
    "\n"
    "class Model:\n"
    "    def __init__(self):\n"
    "        self.history = []\n"
    "    def fit(self, X):\n"
    "        self.history.append(float(X.sum()))\n"
    "        self.coef_ = 1.0\n"
    "        return self\n"
    "\n"
    "X = np.arange(6.0)\n"
)
_FIT_SELF = "@cash.cache\ndef fit(s, X):\n    return s.fit(X)\n"
_FIT_SELF_EDITED = "@cash.cache\ndef fit(s, X):\n    s.fit(X)\n    s.mean_ += 100\n    return s\n"
_FIT_MEAN = "@cash.cache\ndef fit_mean(s, X):\n    s.fit(X)\n    return s.mean_\n"
_TRAIN = "@cash.cache\ndef train(m, X):\n    return m.fit(X)\n"


def _state(runner) -> tuple[str, str, str]:
    return runner.peek("vars(s)"), runner.peek("r is s"), runner.peek("fit.cache_info()['hits']")


def test_a_rerun_sets_the_attributes_on_the_object_and_hands_it_back(nb_runner):
    nb_runner.create_notebook([_DEFS, _FIT_SELF, "s = Scaler()", "r = fit(s, X)", "centred = X - s.mean_"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _state(nb_runner) == ("{'k': 2, 'mean_': 5.0}", "True", "0")

    nb_runner.run_all()
    assert _state(nb_runner) == ("{'k': 2, 'mean_': 5.0}", "True", "1"), "the second run is a hit, as in plain Python"
    # A cell further down reads what the hit set.
    assert nb_runner.peek("centred.tolist()") == "[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]"


def test_the_object_made_in_the_same_cell_gets_the_attributes(nb_runner):
    nb_runner.create_notebook([_DEFS, _FIT_SELF, "s = Scaler()\nr = fit(s, X)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    vars_s, same, _hits = _state(nb_runner)
    assert (vars_s, same) == ("{'k': 2, 'mean_': 5.0}", "True")


def test_a_fit_that_returns_something_else_still_sets_the_attributes(nb_runner):
    nb_runner.create_notebook([_DEFS, _FIT_MEAN, "s = Scaler()", "m = fit_mean(s, X)", "later = s.mean_ * 2"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("fit_mean.cache_info()['hits']") == "1"
    assert nb_runner.peek("(vars(s), m, later)") == "({'k': 2, 'mean_': 5.0}, 5.0, 10.0)"


def test_after_a_kernel_restart_run_all_matches_plain_python(nb_runner):
    nb_runner.create_notebook([_DEFS, _FIT_SELF, "s = Scaler()", "r = fit(s, X)", "centred = X - s.mean_"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_all()
    vars_s, same, _hits = _state(nb_runner)
    assert (vars_s, same) == ("{'k': 2, 'mean_': 5.0}", "True")
    assert nb_runner.peek("centred.tolist()") == "[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]"


def test_an_edit_to_the_cached_function_is_not_served_the_old_attributes(nb_runner):
    nb_runner.create_notebook([_DEFS, _FIT_SELF, "s = Scaler()", "r = fit(s, X)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, _FIT_SELF_EDITED)  # 1-based
    nb_runner.run_all()
    assert nb_runner.peek("(vars(s), r is s)") == "({'k': 2, 'mean_': 105.0}, True)"


def test_a_change_a_hit_cannot_replay_runs_every_time(nb_runner):
    # ``m.history.append`` changes a list ``m`` already held: plain Python
    # runs it every time, and so must the notebook.
    nb_runner.create_notebook([_DEFS, _TRAIN, "m = Model()", "r = train(m, X)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("(vars(m), r is m)") == "({'history': [15.0], 'coef_': 1.0}, True)"
    assert nb_runner.peek("train.cache_info()['hits']") == "0"


def test_an_object_that_cannot_be_hashed_by_its_content_still_ends_up_fitted(nb_runner):
    # A lock cannot be pickled, so the check after the body cannot hash the
    # object by what it holds; the recorded attribute change still shows.
    nb_runner.create_notebook(
        [
            _DEFS,
            _FIT_SELF,
            "import threading\nclass Locked(Scaler):\n    def __init__(self):\n        super().__init__()\n"
            "        self.lock = threading.Lock()",
            "s = Locked()",
            "r = fit(s, X)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("(s.mean_, r is s)") == "(5.0, True)"
