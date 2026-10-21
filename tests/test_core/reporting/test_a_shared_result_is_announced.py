"""A result that shares state with something the caller holds says so.

Found while stress-testing the decorator: on the computing run these
behave like plain Python, and on a HIT they quietly stop:

* ``return base[lo:hi]`` -- the numpy view stops sharing memory with ``base``,
* ``return Wrapper(rows)`` -- the result no longer holds the caller's list,
* ``return CONFIG`` -- the caller's write no longer reaches the module global.

A hit hands back a value rebuilt from the stored bytes, so this is what caching
such a result means. Cash cannot tell whether the caller relies on the sharing,
so it says what will differ and lets the user waive it (``assume_safe=True``).
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")


CONFIG: dict = {}


def _warnings_of(fn, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(*args)
    return [str(w.message) for w in seen if "CACHE-RESULT-SHARED" in str(w.message)]


def test_a_view_of_an_argument_is_announced(cash_instance):
    @cash_instance.cache
    def window(base, lo, hi):
        return base[lo:hi]

    said = _warnings_of(window, np.arange(5), 1, 4)
    assert said, "no warning for a result sharing memory with an argument"
    assert "base" in said[0], said[0]


def test_a_result_holding_an_argument_is_announced(cash_instance):
    @cash_instance.cache
    def wrap(rows, tag):
        return {"data": rows, "tag": tag}

    assert _warnings_of(wrap, [1, 2, 3], "t")


def test_a_returned_module_global_is_announced(cash_instance):
    @cash_instance.cache
    def config(n):
        return CONFIG

    assert _warnings_of(config, 1)


def test_assume_safe_waives_it(cash_instance):
    @cash_instance.cache(assume_safe=True)
    def window(base, lo, hi):
        return base[lo:hi]

    assert not _warnings_of(window, np.arange(5), 1, 4)


def test_an_independent_result_says_nothing(cash_instance):
    @cash_instance.cache
    def copied(base, lo, hi):
        return base[lo:hi].copy()

    @cash_instance.cache
    def rebuilt(rows, tag):
        return {"data": list(rows), "tag": tag}

    @cash_instance.cache
    def snapshot(n):
        return dict(CONFIG)

    assert not _warnings_of(copied, np.arange(5), 1, 4)
    assert not _warnings_of(rebuilt, [1, 2, 3], "t")
    assert not _warnings_of(snapshot, 1)


ROWS: list = [1, 2]


def test_a_result_holding_a_module_global_is_announced(cash_instance):
    # `{'rows': ROWS}`: plain Python hands back the caller's own list, and an
    # edit to ROWS shows through the result; a hit's copy does not.
    @cash_instance.cache
    def wrap(n):
        return {"rows": ROWS}

    said = _warnings_of(wrap, 1)
    assert said, "no warning for a result holding a module global"
    assert "ROWS" in said[0], said[0]


def test_a_result_holding_a_module_global_in_an_attribute_is_announced(cash_instance):
    import argparse

    @cash_instance.cache
    def wrap(n):
        return argparse.Namespace(rows=ROWS)

    assert _warnings_of(wrap, 1)


def test_a_closure_variable_returned_or_held_is_announced(cash_instance):
    box = [1]

    @cash_instance.cache
    def held(n):
        return {"rows": box}

    @cash_instance.cache
    def returned(n):
        return box

    said = _warnings_of(held, 1)
    assert said and "box" in said[0], said
    assert _warnings_of(returned, 1)


def test_a_held_sentinel_global_says_nothing(cash_instance):
    # Handed back as itself on a hit (`identity_refs`): nothing differs.
    @cash_instance.cache
    def lookup(n):
        return {"rows": _MARK}

    assert not _warnings_of(lookup, 1)


_MARK = object()


CONFIG_TREE: dict = {"sub": {"a": 1}}
NESTED_T = ([1, 2],)


class _Holder:
    ROWS: list = [1, 2]


@pytest.mark.parametrize(
    ("label", "body", "named"),
    [
        ("an item of a global dict", "return CONFIG_TREE['sub']", "CONFIG_TREE['sub']"),
        ("a list inside a global tuple", "return NESTED_T", "NESTED_T[0]"),
        ("a class attribute", "return {'r': _Holder.ROWS}", "_Holder.ROWS"),
        ("a global in a deque", "return collections.deque([ROWS])", "ROWS"),
        ("a global in an OrderedDict", "return collections.OrderedDict(a=ROWS)", "ROWS"),
    ],
)
def test_an_object_the_function_reaches_by_name_is_announced(cash_instance, label, body, named):
    ns = {**globals(), "collections": __import__("collections")}
    exec(f"def reach(n):\n    {body}\n", ns)
    said = _warnings_of(cash_instance.cache(ns["reach"]), 1)
    assert said, f"no warning for {label}"
    assert named in said[0] and "which the caller" not in said[0], said[0]


def test_a_default_and_a_function_attribute_are_announced(cash_instance):
    def defaulted(n, d=[]):
        return d

    def attributed(n):
        return attributed.store

    attributed.store = [1]
    said = _warnings_of(cash_instance.cache(defaulted), 1)
    assert said and "default argument 'd'" in said[0], said
    assert _warnings_of(cash_instance.cache(attributed), 1)


def test_a_view_of_a_global_array_is_announced(cash_instance):
    np = pytest.importorskip("numpy")
    ns = {"ARR": np.arange(10.0)}
    exec("def window(n):\n    return ARR[1:4]\n", ns)
    said = _warnings_of(cash_instance.cache(ns["window"]), 1)
    assert said and "shares memory with the module global 'ARR'" in said[0], said


def test_an_enum_member_and_a_frozen_dataclass_say_nothing(cash_instance):
    import dataclasses
    import enum

    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass(frozen=True)
    class Frozen:
        x: int = 1

    ns = {"RED": Color.RED, "CFG": Frozen()}
    exec("def colored(n):\n    return {'c': RED}\ndef configured(n):\n    return {'cfg': CFG}\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["colored"]), 1)
    assert not _warnings_of(cash_instance.cache(ns["configured"]), 1)


def test_the_result_is_walked_once_whatever_the_function_names(cash_instance, monkeypatch):
    from cash.decorator import purity_checks

    walks = []
    real = purity_checks._found_in
    monkeypatch.setattr(
        purity_checks, "_found_in", lambda v, t, d: walks.append(len(t)) or real(v, t, d) if d == 2 else real(v, t, d)
    )
    ns = {f"G{i}": {"k": i} for i in range(5)}
    exec("def many(n):\n    return [G0, G1, G2, G3, G4] and [{'n': n}]\n", ns)
    _warnings_of(cash_instance.cache(ns["many"]), 1)
    assert len(walks) == 1, walks


@pytest.mark.parametrize("cow", [True, False], ids=["copy-on-write", "pandas-2-views"])
def test_a_column_of_a_global_frame_is_announced_without_copy_on_write(cash_instance, monkeypatch, cow):
    # Without copy-on-write, `DF["a"]` is a view of the frame: a write
    # through the result reaches DF in plain Python, and not on a hit.
    pd = pytest.importorskip("pandas")
    from cash.decorator import purity_checks

    monkeypatch.setattr(purity_checks, "is_cow_pandas", lambda value: cow)
    ns = {"DF": pd.DataFrame({"a": [1.0, 2.0, 3.0]})}
    exec("def column(n):\n    return DF['a']\n", ns)
    said = _warnings_of(cash_instance.cache(ns["column"]), 1)
    assert bool(said) is (not cow), said
