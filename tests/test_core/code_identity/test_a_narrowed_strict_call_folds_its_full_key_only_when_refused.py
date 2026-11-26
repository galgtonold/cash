"""A narrowed strict call builds its full key's code only when it needs it.

A call that needs a strict trace (it picks a module by a run-time name) and
whose key holds the classes of module data by their surface accepts the
code a key of it built unnarrowed folds (`KeyBuilder.full_key_folded`). That
fold keys every held class by all its code: hundreds of milliseconds for a
library's classes, built on every miss although the narrowed key's code was
enough for nearly every call. It is built after the body now, only for a
trace that the narrowed key's code refuses, and only while nothing it reads
moved in the body: no function's code set, no module but cash's own
imported, the arguments the same. Else the call is refused and the
function's later misses build it before the body, as before.

Each run is a fresh interpreter; the script counts the full folds.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
]

COUNTING = """
import cash
from cash import code_swaps
from cash.decorator.runtime import KeyBuilder

FOLDS = [0]
_full = KeyBuilder.full_key_folded


def counted(self, *args, **kwargs):
    FOLDS[0] += 1
    return _full(self, *args, **kwargs)


KeyBuilder.full_key_folded = counted
WATCHED = getattr(code_swaps, "function_sets", lambda: 0)() is not None
"""

PURE_LIB = """
def codec(name):
    return __import__(name)


class Scaler:
    factor = 2

    def apply(self, v):
        return v * self.factor


SCALER = Scaler()
"""

PURE_MAIN = (
    COUNTING
    + """
import plainlib


@cash.cache
def scaled(x):
    return plainlib.SCALER.apply(x) + plainlib.codec("math").floor(0.5)


print(scaled(3), scaled(3), scaled.cache_info()["hits"], FOLDS[0], WATCHED)
"""
)

SWAP_LIB = """
def codec(name):
    return __import__(name)


def helper(v):
    return v


def doubled(v):
    return v * 2


class Model:
    def __init__(self, **kwargs):
        kwargs.update(scale=2)
        # Sets code that the fold of every held class's code reads.
        helper.__code__ = doubled.__code__
        self.scale = kwargs["scale"]

    def fit(self, x):
        return helper(x) * self.scale


MODELS = {"line": Model}
"""

SWAP_MAIN = (
    COUNTING
    + """
import swaplib


@cash.cache
def fit(x):
    return swaplib.MODELS["line"]().fit(x) + swaplib.codec("math").floor(0.5)


print(fit(2), fit(2), fit(2), fit.cache_info()["hits"], FOLDS[0], WATCHED)
"""
)


def _run(tmp_path, files: dict[str, str]) -> list[str]:
    project = tmp_path / "proj"
    project.mkdir()
    for name, text in files.items():
        (project / name).write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    out = run_python("main.py", cwd=project, cache_dir=tmp_path / "cache", timeout=60)
    return out.stdout.split()


def test_a_call_the_narrowed_key_accepts_builds_no_full_fold(tmp_path):
    first, second, hits, folds, watched = _run(tmp_path, {"plainlib.py": PURE_LIB, "main.py": PURE_MAIN})
    assert (first, second, hits) == ("6", "6", "1"), "the call was not stored"
    # Without the extension's function watcher it is built before the body.
    assert folds == ("0" if watched == "True" else "1")


def test_a_body_that_sets_code_the_fold_reads_is_folded_before_the_next_body(tmp_path):
    out = _run(tmp_path, {"swaplib.py": SWAP_LIB, "main.py": SWAP_MAIN})
    *values, hits, folds, watched = out
    assert values == ["8", "8", "8"]
    if watched == "True":
        # The first call's body set a function's code: a fold built after it
        # is not what the key before it would have folded, so that call is
        # refused, and the second call's fold is built before its body.
        assert (hits, folds) == ("1", "1")
    else:
        assert (hits, folds) == ("2", "1")
