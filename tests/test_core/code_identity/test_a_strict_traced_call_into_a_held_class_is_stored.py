"""A call that needs its trace and reaches a held class is stored and served.

A cached function that picks a module by a run-time name (lmfit's
``jsonutils`` calls ``__import__(name)``) needs a strict trace: the code it
ran that no key of the call covers is judged, and a call that ran code with
a side effect is not stored. Once module data's classes were keyed by their
surface (a held class, `HELD_BY_NAME`), the key no longer covered their
methods, so the strict trace judged all of the library's construction and
fit code -- ``kwargs.update`` on a constructor's own ``**kwargs``, an
``__init__`` writing ``self`` -- and refused every call: nothing was ever
stored. The strict trace accepts what a key of the call built unnarrowed
covers; the narrowed key and the trace still check each method that ran.

Each run is a fresh interpreter over one cache folder; every result is
compared with the same program run without cash.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

pytestmark = [
    pytest.mark.core,
]

FITLIB = """
import math


def codec(name):
    # Picks a module by a run-time name, as lmfit.jsonutils does.
    return __import__(name)


class Parameters(dict):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.update(kwargs.pop("extra", {}))
        kwargs.setdefault("b", 0)
        self.update(kwargs)


class Minimizer:
    def __init__(self, model, params, **kws):
        kws["scale"] = 2
        self.model = model
        self.params = params
        self.kws = kws

    def prepare_fit(self):
        self.nfev = 0
        return dict(self.params)

    def run(self, x):
        values = self.prepare_fit()
        return self.model.func(x, **values) * self.kws["scale"]

    def shgo(self, x):
        return x


class Model:
    def __init__(self, func, **kwargs):
        kwargs.update({"prefix": ""})
        kwargs["nan_policy"] = "raise"
        self.func = func
        self.opts = kwargs

    def make_params(self, **kw):
        return Parameters(**kw)

    def fit(self, x, params):
        return Minimizer(self, params).run(x)


def line(x, a, b):
    return a * x + b


MODELS = {"line": Model}
"""

MAIN = """
import cash
import fitlib


@cash.cache
def fit(x):
    model = fitlib.MODELS["line"](fitlib.line, independent_vars=["x"])
    params = model.make_params(a=3, extra={"b": 1})
    return model.fit(x, params) + fitlib.codec("math").floor(0.5)


first = fit(2)
print(first, fit(2), fit.cache_info()["hits"])
"""

#: Edits to code the call ran: recomputed.
RAN = {
    "constructor": ("fitlib.py", 'kwargs["nan_policy"] = "raise"', 'kwargs["nan_policy"] = "omit"'),
    "constructor's kwargs": ("fitlib.py", 'kws["scale"] = 2', 'kws["scale"] = 3'),
    "parameters": ("fitlib.py", 'kwargs.setdefault("b", 0)', 'kwargs.setdefault("b", 4)'),
    "fit method": ("fitlib.py", 'self.kws["scale"]\n', 'self.kws["scale"] + 1\n'),
    "helper it ran": ("fitlib.py", "return a * x + b\n", "return a * x + b + 5\n"),
}


def _parts(out: str) -> tuple[str, str, int]:
    first, second, hits = out.split()
    return first, second, int(hits)


def test_an_unchanged_call_is_stored_and_served_in_a_new_process(tmp_path):
    first, after, plain = edited_runs(tmp_path, {"fitlib.py": FITLIB, "main.py": MAIN})
    value = _parts(plain)[0]
    assert value == "14"
    assert _parts(first) == (value, value, 1), "the call was not stored"
    assert _parts(after) == (value, value, 2), "a new process was not served the stored result"


@pytest.mark.parametrize("edit", list(RAN))
def test_an_edit_to_code_the_call_ran_recomputes(tmp_path, edit):
    first, after, plain = edited_runs(tmp_path, {"fitlib.py": FITLIB, "main.py": MAIN}, [RAN[edit]])
    assert _parts(first) == ("14", "14", 1)
    value = _parts(plain)[0]
    assert _parts(after) == (value, value, 1), f"{edit}: served the result from before"
