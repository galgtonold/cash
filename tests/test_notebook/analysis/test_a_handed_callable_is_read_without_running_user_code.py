"""Whether a function handed to ``df['o'].apply(...)`` changes its parameter
is read from its code object alone. A callable object it calls, with a
``__getattr__`` or a ``__class__`` property of its own, is never asked
anything, nor a ``__wrapped__`` set on the function."""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap

import pytest

from cash.analysis import mutation_effects
from cash.analysis.mutations import top_level_calls

MODULE = textwrap.dedent(
    """
    CALLS = []

    class Spy:
        def __getattr__(self, name):
            CALLS.append(("getattr", name))
            raise AttributeError(name)

        def __call__(self, *a):
            return None

    class SpyClass(Spy):
        @property
        def __class__(self):
            CALLS.append(("__class__",))
            return Spy

    helper = Spy()
    helper2 = SpyClass()

    def upd(d):
        helper(d)
        if d["i"] == 3:
            d["z"] = 1

    def upd2(d):
        helper2(d)
        d["z"] = 1

    def upd3(d):
        d["z"] = 1

    upd3.__wrapped__ = Spy()

    def upd4(d):
        upd(d)
    """
)


@pytest.fixture
def spies(tmp_path, monkeypatch):
    (tmp_path / "handed_spies.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("handed_spies")
    yield module
    sys.modules.pop("handed_spies", None)


@pytest.mark.parametrize(
    "statement",
    [
        "_ = df['o'].apply(upd)",
        "_ = df['o'].apply(upd2)",
        "_ = df['o'].apply(upd3)",
        "_ = df['o'].apply(upd4)",
        "_ = df['o'].map(lambda d: upd(d))",
        "_ = df['o'].map(lambda d: upd3(d))",
        "_ = list(map(upd4, df['o']))",
    ],
)
def test_the_frame_changes_and_no_code_of_the_users_runs(spies, statement):
    namespace = {name: getattr(spies, name) for name in ("upd", "upd2", "upd3", "upd4", "helper", "helper2")}
    namespace["df"] = None
    tree = ast.parse(statement)
    changed, _watched = mutation_effects._changed_through_lambda_parameters(tree, top_level_calls(tree), namespace)
    assert changed == {"df"}
    assert spies.CALLS == []
