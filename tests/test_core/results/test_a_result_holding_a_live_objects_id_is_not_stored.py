"""A ``@cash.cache`` result that holds the ``id()`` of a live object is not
stored (``STORE-IDENTITY-HELD``).

An address is only good in the process that took it. Before the fix, a
function returning ``f"run-{id(KEEP)}"`` was served in a second process
over the same cache folder with the first process's address, where plain
Python gives the current one:

    tag   WRONG got='run-140460520282096-1' want='run-140435745462256-1'
    rep   WRONG got=['<__main__.Q object at 0x7fbf836a97f0>', 1] want=[... 0x7fb9beb8d7f0>', 1]

The notebook path already refused such a value (``NOTEBOOK-IDENTITY-HELD``,
``identity_holders``); the decorator now asks the same detector. Its
positive controls, that ordinary results are still stored, are in
``test_an_ordinary_result_with_numbers_is_still_stored.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash.exceptions import CashCacheStoreFailedWarning

np = pytest.importorskip("numpy")

_SCRIPT = textwrap.dedent(
    """
    import json
    import warnings
    from cash import Cash

    warnings.simplefilter("ignore")
    c = Cash(cache_dir=".c", register_magic=False)
    runs = []

    class Q:
        pass

    KEEP = Q()

    @c.cache
    def tag(n):
        runs.append("tag")
        return f"run-{id(KEEP)}-{n}"

    @c.cache
    def bare(n):
        runs.append("bare")
        return [id(KEEP), n]

    @c.cache
    def rep(n):
        runs.append("rep")
        return [repr(KEEP), n]

    @c.cache
    def hexed(n):
        runs.append("hexed")
        return {"k": "obj" + hex(id(KEEP)), "n": n}

    got = {"tag": tag(1), "bare": bare(1), "rep": rep(1), "hexed": hexed(1)}
    want = {
        "tag": f"run-{id(KEEP)}-1",
        "bare": [id(KEEP), 1],
        "rep": [repr(KEEP), 1],
        "hexed": {"k": "obj" + hex(id(KEEP)), "n": 1},
    }
    print(json.dumps({"wrong": sorted(k for k in got if got[k] != want[k]), "runs": sorted(runs)}))
    """
)


def _run(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    p = subprocess.run(
        [sys.executable, "-c", _SCRIPT], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_a_second_process_gets_the_current_id_not_the_stored_one(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert first["wrong"] == []
    assert second["wrong"] == [], "a second process was served the first process's address"
    assert second["runs"] == ["bare", "hexed", "rep", "tag"]


class _Thing:
    pass


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def test_an_id_of_an_argument_in_the_result_is_not_stored_and_says_so(disk_cash):
    runs = []

    @disk_cash.cache
    def key_for(obj, n):
        runs.append(n)
        return {"key": f"k{id(obj)}", "n": n}

    thing = _Thing()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert key_for(thing, 1) == {"key": f"k{id(thing)}", "n": 1}
        assert key_for(thing, 1) == {"key": f"k{id(thing)}", "n": 1}
    assert runs == [1, 1], "a result holding an argument's id() was served from the cache"
    assert "STORE-IDENTITY-HELD" in _codes(caught)
    assert any(issubclass(w.category, CashCacheStoreFailedWarning) for w in caught)


def test_an_id_of_an_object_the_result_holds_is_not_stored(disk_cash):
    runs = []

    @disk_cash.cache
    def build(n):
        runs.append(n)
        thing = _Thing()
        return {"thing": thing, "who": id(thing), "n": n}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(1)
        build(1)
    assert runs == [1, 1]


def test_a_category_column_spelling_an_arguments_id_is_not_stored(disk_cash):
    """A pandas category column is read by its categories, the very
    values it holds (`_pandas_column`)."""
    pd = pytest.importorskip("pandas")
    runs = []

    @disk_cash.cache
    def label(obj, n):
        runs.append(n)
        return pd.DataFrame({"who": pd.Categorical([repr(obj), "other"] * n), "n": range(2 * n)})

    thing = _Thing()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label(thing, 3)
        label(thing, 3)
    assert runs == [3, 3], "a category column holding an argument's repr was served from the cache"


def test_a_big_int_array_holding_an_arguments_id_is_not_stored(disk_cash):
    """Past `_BOUNDS_FIRST` numbers an array is first told by its lowest and
    highest; one id among small numbers still spans a mapping."""
    runs = []

    @disk_cash.cache
    def table(obj, n):
        runs.append(n)
        out = np.arange(n, dtype=np.int64)
        out[n // 2] = id(obj)
        return out

    thing = _Thing()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table(thing, 300_000)
        table(thing, 300_000)
    assert runs == [300_000, 300_000]
