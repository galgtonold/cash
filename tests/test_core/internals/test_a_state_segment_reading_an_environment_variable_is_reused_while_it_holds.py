"""A state segment that reads an environment variable by name is reused
while that variable holds.

The key folds each variable a function reads by name (``os.environ.get``,
``os.getenv``, ``os.environ[...]``, ``... in os.environ``), never the rest
of the environment. The remembered state segment (`cash.decorator.state_memo`)
keeps what each such read answered and is served only while every one
answers the same: a write to another variable -- a library setting its own
at import or call time -- leaves it standing, a write to one it read builds
it again, in this process and in the next.

The environment read as a whole is judged as before: through ``os`` it is
warned about and keyed not at all, and a module global bound to
``os.environ`` is folded as a value, so any variable moves the key.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import state_memo
from cash.exceptions import CashImpurityWarning

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

READ = "CASH_TEST_STATE_ENV_READ"
OTHER = "CASH_TEST_STATE_ENV_OTHER"

SOURCE = textwrap.dedent(f"""
    import os

    def by_name(x):
        return (x, os.environ.get("{READ}"), os.getenv("{READ}_B"), "{READ}_C" in os.environ)

    def as_whole(x):
        return (x, len(dict(os.environ)) > 0)
""")

GLOBAL_SOURCE = textwrap.dedent("""
    import os

    ENV = os.environ

    def through_global(x):
        return (x, ENV.get("UNUSED_BY_NAME"))
""")


@pytest.fixture
def mods(tmp_path, monkeypatch):
    stem = tmp_path.name.replace("-", "_")
    names = (f"envreads_{stem}", f"envglobal_{stem}")
    (tmp_path / f"{names[0]}.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / f"{names[1]}.py").write_text(GLOBAL_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for var in (READ, f"{READ}_B", f"{READ}_C", OTHER):
        monkeypatch.delenv(var, raising=False)
    importlib.invalidate_caches()
    yield tuple(importlib.import_module(n) for n in names)
    for n in names:
        sys.modules.pop(n, None)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)


def _spec(c, f):
    return c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = _spec(c, f)
    return c._keys._code_state(spec, [], note=True), c._keys._code_state_now(spec, [], note=True)


def _reused(c, f) -> str:
    for _ in range(6):
        before = state_memo.STATS["reused"]
        got, fresh = _state(c, f)
        assert got == fresh
        if state_memo.STATS["reused"] > before:
            return got
    raise AssertionError("the segment is not being reused: the test proves nothing")


def test_a_write_to_another_variable_keeps_the_segment(c, mods, monkeypatch):
    f = c.cache(mods[0].by_name)
    f(1)
    old = _reused(c, f)
    monkeypatch.setenv(OTHER, "1")
    assert _reused(c, f) == old
    misses = f.cache_info()["misses"]
    assert f(1) == (1, None, None, False)
    assert f.cache_info()["misses"] == misses  # a hit


@pytest.mark.parametrize("var", [READ, f"{READ}_B", f"{READ}_C"])
@pytest.mark.parametrize("memo_kind", ["honest", "sabotaged"])
def test_a_write_to_a_variable_read_moves_the_key(c, mods, monkeypatch, var, memo_kind):
    if memo_kind == "sabotaged":
        # Takes every variable for unchanged: what the check guards against.
        monkeypatch.setattr(state_memo, "VERIFY", False)
        monkeypatch.setattr(state_memo, "_env_holds", lambda env: True)
    f = c.cache(mods[0].by_name)
    f(1)
    old = _reused(c, f)
    monkeypatch.setenv(var, "on")
    got, fresh = _state(c, f)
    assert fresh != old, "the write did not move the segment: the scenario proves nothing"
    if memo_kind == "sabotaged":
        assert got == old
        return
    assert got == fresh
    misses = f.cache_info()["misses"]
    result = f(1)
    assert f.cache_info()["misses"] == misses + 1  # computed again
    assert result == (1, "on" if var == READ else None, "on" if var.endswith("_B") else None, var.endswith("_C"))
    # The new answer is what is remembered now.
    assert _reused(c, f) == fresh


CHILD = textwrap.dedent("""
    import sys, warnings
    warnings.simplefilter("ignore")
    sys.path.insert(0, {path!r})
    from cash import Cash
    import {mod} as m
    f = Cash(cache_dir={cache!r}, register_magic=False).cache(m.by_name)
    for _ in range(3):
        value = f(1)
    print(f.cache_info()["misses"], value[1])
""")


def test_a_variable_read_moves_the_key_in_another_process(c, mods, tmp_path):
    f = c.cache(mods[0].by_name)
    f(1)
    _reused(c, f)
    script = CHILD.format(path=str(tmp_path), mod=mods[0].__name__, cache=str(tmp_path / "cache"))

    def child(**env):
        base = {k: v for k, v in os.environ.items() if not k.startswith("CASH_TEST_STATE_ENV")}
        out = subprocess.run(
            [sys.executable, "-c", script], env={**base, **env}, capture_output=True, text=True, check=True
        )
        return out.stdout.split()

    assert child() == ["0", "None"]  # the entry this process stored
    assert child(**{OTHER: "x"}) == ["0", "None"]  # another variable: still a hit
    assert child(**{READ: "x"}) == ["1", "x"]  # the variable read: computed
    assert child(**{READ: "x", OTHER: "y"}) == ["0", "x"]  # and stored under its own key


def test_the_environment_read_as_a_whole_is_judged_as_before(c, mods, monkeypatch):
    with pytest.warns(CashImpurityWarning):
        f = c.cache(mods[0].as_whole)
        f(1)
    got, fresh = _state(c, f)
    assert got == fresh
    monkeypatch.setenv(OTHER, "1")
    got2, fresh2 = _state(c, f)
    # Not keyed through `os`, as before: the warning says so.
    assert got2 == fresh2 == fresh

    g = c.cache(mods[1].through_global)
    g(1)
    for _ in range(3):
        _state(c, g)
    before, _ = _state(c, g)
    monkeypatch.setenv(OTHER, "2")
    got, fresh = _state(c, g)
    # A global bound to the mapping is folded as a value: any variable moves it.
    assert fresh != before
    assert got == fresh
