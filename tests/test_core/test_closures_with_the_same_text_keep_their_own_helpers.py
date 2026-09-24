"""Two closures with the same text in one module are keyed by their own helpers.

A closure's names resolve in its cells as well as its module. Two factories
whose inner functions read the same can capture different modules, and each
cached closure must follow the helper it calls: editing that helper
recomputes it, and editing the other one does not.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from cash import Cash
from cash.purity_analyzer import PurityAnalyzer

pytestmark = [pytest.mark.core]

HELPER = "def helper(x):\n    return x + {step}\n"

CALLER = """\
import {a}
import {b}


def make_a():
    cap = {a}

    def total(x):
        return cap.helper(x)

    return total


def make_b():
    cap = {b}

    def total(x):
        return cap.helper(x)

    return total
"""


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """(helper module a, helper module b, caller module), all fresh."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    tag = tmp_path.name
    a, b, caller = f"_same_text_a_{tag}", f"_same_text_b_{tag}", f"_same_text_caller_{tag}"
    for name in (a, b, caller):
        monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{a}.py").write_text(HELPER.format(step=1), encoding="utf-8")
    (tmp_path / f"{b}.py").write_text(HELPER.format(step=2), encoding="utf-8")
    (tmp_path / f"{caller}.py").write_text(CALLER.format(a=a, b=b), encoding="utf-8")
    return importlib.import_module(a), importlib.import_module(b), importlib.import_module(caller)


def test_the_analyzer_follows_each_closures_own_capture(modules):
    a, b, caller = modules
    analyzer = PurityAnalyzer()
    first = analyzer.analyze(caller.make_a())
    second = analyzer.analyze(caller.make_b())
    assert {q for q in first.helper_source_hashes if q.endswith(".helper")} == {f"{a.__name__}.helper"}
    assert {q for q in second.helper_source_hashes if q.endswith(".helper")} == {f"{b.__name__}.helper"}


def test_editing_the_helper_a_closure_calls_recomputes_it(tmp_path, modules):
    a, b, caller = modules
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    total_a = c.cache(caller.make_a())
    total_b = c.cache(caller.make_b())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert total_a(1) == 2
        assert total_b(1) == 3
        assert total_b(1) == 3
        assert total_b.cache_info()["hits"] == 1  # the entry is really served

        (tmp_path / f"{b.__name__}.py").write_text(HELPER.format(step=200), encoding="utf-8")
        importlib.reload(b)

        assert total_b(1) == 201
        assert total_a(1) == 2
