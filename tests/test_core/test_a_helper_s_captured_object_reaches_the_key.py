"""A helper built by a factory keys the object its closure captured.

``score = make_scorer(Config(weight=10))`` builds a helper whose closure holds
the config. Only primitives and plain containers among a helper's captures
were keyed, so a dataclass, an ``argparse.Namespace`` or any instance fell
away and ``Config(weight=11)`` was served the old result. The cached
function's own closure keyed the same object all along.

An object the helper may change (it calls a method on it, or writes to it)
is still left out: keying it would key each call on the last one.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash

SCORING = """
import argparse
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    weight: int

class Plain:
    def __init__(self, weight):
        self.weight = weight

def make_scorer(cfg):
    def score(x):
        return x * cfg.weight
    return score

def make_counting(cfg):
    def score(x):
        cfg.calls.append(x)
        return x * cfg.weight
    return score

score = make_scorer(Config(10))
"""

BUILDS = {
    "frozen_dataclass": "Config({w})",
    "namespace": "argparse.Namespace(weight={w})",
    "plain_instance": "Plain({w})",
}


@pytest.fixture()
def scoring(tmp_path, monkeypatch):
    tag = f"_{time.monotonic_ns()}"
    (tmp_path / f"scoring{tag}.py").write_text(textwrap.dedent(SCORING), encoding="utf-8")
    (tmp_path / f"reader{tag}.py").write_text(
        f"import scoring{tag} as scoring\n\ndef f(x):\n    return scoring.score(x)\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mods = [importlib.import_module(f"scoring{tag}"), importlib.import_module(f"reader{tag}")]
    for m in mods:
        monkeypatch.setitem(sys.modules, m.__name__, m)
    return mods[0], mods[1], Cash(cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize("build", sorted(BUILDS))
def test_a_captured_object_reaches_the_key(scoring, build):
    mod, reader, c = scoring
    mod.score = mod.make_scorer(eval(BUILDS[build].format(w=10), vars(mod)))
    f = c.cache(reader.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert f(2) == 20
        mod.score = mod.make_scorer(eval(BUILDS[build].format(w=11), vars(mod)))
        assert f(2) == 22, f"{build}: the helper's captured object did not reach the key"


def test_an_object_the_helper_changes_is_not_keyed(scoring):
    mod, reader, c = scoring
    cfg = mod.Plain(10)
    cfg.calls = []
    mod.score = mod.make_counting(cfg)
    f = c.cache(reader.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            assert f(2) == 20
    assert f.cache_info()["hits"] == 2, "keyed on an object the helper appends to"
