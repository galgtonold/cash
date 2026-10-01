"""A callable read as a global is keyed by what it carries, wherever it is held.

A callable in a global was replaced by its code identity before hashing, so
what it was built with fell away: ``SCALE = Scaler(10)`` with ``__call__``
kept its key after ``Scaler(11)``, and so did ``{"scale": partial(mul, k=10)}``
after ``k=11`` and ``{"a": Cfg(10).mul}`` after ``Cfg(11)``. A bare partial or
bound method was keyed; the same object in a dict or list was not.

Each case rebinds the global to an object with other data and expects the
next call to recompute.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash

MODEL = """
import functools

class Scaler:
    def __init__(self, k):
        self.k = k
    def __call__(self, x):
        return x * self.k
    def method(self, x):
        return x * self.k

class Cfg:
    def __init__(self, k):
        self.k = k
    def mul(self, x):
        return x * self.k

def mul(x, k):
    return x * k

def helper(x):
    return SCALE(x)

SCALE = Scaler(10)
D = {"a": Scaler(10)}
L = [Scaler(10)]
H = {"scale": functools.partial(mul, k=10)}
BM = {"a": Cfg(10).mul}
"""

CASES = {
    "called": ("SCALE", "Scaler(11)", "return SCALE(x)"),
    "method": ("SCALE", "Scaler(11)", "return SCALE.method(x)"),
    "attribute": ("SCALE", "Scaler(11)", "return x * SCALE.k"),
    "in_a_dict": ("D", "{'a': Scaler(11)}", "return D['a'](x)"),
    "in_a_list": ("L", "[Scaler(11)]", "return L[0](x)"),
    "partial_in_a_dict": ("H", "{'scale': functools.partial(mul, k=11)}", "return H['scale'](x)"),
    "bound_method_in_a_dict": ("BM", "{'a': Cfg(11).mul}", "return BM['a'](x)"),
    "through_a_helper": ("SCALE", "Scaler(11)", "return helper(x)"),
}


def _modules(tmp_path, monkeypatch, body: str):
    tag = f"_{time.monotonic_ns()}"
    (tmp_path / f"model{tag}.py").write_text(textwrap.dedent(MODEL), encoding="utf-8")
    (tmp_path / f"reader{tag}.py").write_text(
        f"from model{tag} import *\n\ndef f(x):\n    {body}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    model = importlib.import_module(f"model{tag}")
    reader = importlib.import_module(f"reader{tag}")
    for m in (model, reader):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    return model, reader


@pytest.mark.parametrize("case", sorted(CASES))
def test_rebinding_what_the_callable_carries_recomputes(case, tmp_path, monkeypatch):
    name, new, body = CASES[case]
    model, reader = _modules(tmp_path, monkeypatch, body)
    f = Cash(cache_dir=str(tmp_path / "cache")).cache(reader.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert f(2) == 20
        value = eval(new, vars(model))
        setattr(model, name, value)
        if name in vars(reader):
            setattr(reader, name, value)
        assert f(2) == 22, f"{case}: served the result of the old {name}"


def test_a_callable_that_memoises_into_itself_settles(tmp_path, monkeypatch):
    """Folding its state must not key each call on the last call's memo."""
    tag = f"_{time.monotonic_ns()}"
    (tmp_path / f"memo{tag}.py").write_text(
        textwrap.dedent("""
        class Memo:
            def __init__(self, k):
                self.k = k
                self.seen = {}
            def __call__(self, x):
                self.seen[x] = x * self.k
                return self.seen[x]

        SCALE = Memo(10)

        def f(x):
            return SCALE(x)
    """),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module(f"memo{tag}")
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    f = Cash(cache_dir=str(tmp_path / "cache")).cache(mod.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in (1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3):
            assert f(x) == x * 10
    assert f.cache_info()["misses"] <= 5, f.cache_info()
