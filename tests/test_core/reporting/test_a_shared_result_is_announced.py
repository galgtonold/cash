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
    CFG: dict = {"sub": {"a": 1}}


@pytest.mark.parametrize(
    ("label", "body", "named"),
    [
        ("an item of a global dict", "return CONFIG_TREE['sub']", "CONFIG_TREE['sub']"),
        ("a list inside a global tuple", "return NESTED_T", "NESTED_T[0]"),
        ("a class attribute", "return {'r': _Holder.ROWS}", "_Holder.ROWS"),
        ("an item of a class attribute", "return _Holder.CFG['sub']", "_Holder.CFG['sub']"),
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


@pytest.mark.parametrize(
    ("label", "frame", "body"),
    [
        ("a column past the 64th", "pd.DataFrame({f'c{i}': [1.0, 2.0] for i in range(70)})", "DF['c69']"),
        ("an Int64 column", "pd.DataFrame({'a': pd.array([1, 2], dtype='Int64'), 'f': [1.0, 2.0]})", "DF['a']"),
        ("a row slice", "pd.DataFrame({'a': [1.0, 2.0, 3.0]})", "DF.iloc[:2]"),
    ],
)
def test_every_block_of_a_global_frame_is_looked_at(cash_instance, monkeypatch, label, frame, body):
    pd = pytest.importorskip("pandas")
    from cash.decorator import purity_checks

    monkeypatch.setattr(purity_checks, "is_cow_pandas", lambda value: False)
    ns = {"pd": pd}
    exec(f"DF = {frame}\ndef part(n):\n    return {body}\n", ns)
    if not purity_checks.shares_memory(ns["part"](1), ns["DF"]):
        pytest.skip(f"this pandas copies {label}")
    assert _warnings_of(cash_instance.cache(ns["part"]), 1), f"no warning for {label}"


def test_a_view_of_a_read_only_global_array_says_nothing(cash_instance):
    np = pytest.importorskip("numpy")
    frozen = np.arange(10.0)
    frozen.flags.writeable = False
    ns = {"RO": frozen, "RW": np.arange(10.0)}
    exec("def ro(n):\n    return RO[1:4]\ndef rw(n):\n    return RW[1:4]\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["ro"]), 1)
    assert _warnings_of(cash_instance.cache(ns["rw"]), 1)


def test_an_enum_whose_member_holds_a_list_says_nothing(cash_instance):
    import enum

    class Box(enum.Enum):
        A = [1]

    ns = {"Box": Box}
    exec("def boxed(n):\n    return [Box.A]\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["boxed"]), 1)


def test_the_result_is_looked_into_as_far_as_its_first_items(cash_instance):
    from cash.decorator.purity_checks import SHARED_ITEMS

    ns = {"ROWS": [1, 2], "SHARED_ITEMS": SHARED_ITEMS}
    exec(
        "def early(n):\n    return [ROWS] + [None] * SHARED_ITEMS\n"
        "def late(n):\n    return [None] * SHARED_ITEMS + [ROWS]\n",
        ns,
    )
    assert _warnings_of(cash_instance.cache(ns["early"]), 1)
    assert not _warnings_of(cash_instance.cache(ns["late"]), 1)


def test_every_store_is_looked_at_and_warns_once(cash_instance, monkeypatch):
    from cash.decorator.purity_checks import PurityChecks

    looks = []
    real = PurityChecks._shared_with
    monkeypatch.setattr(PurityChecks, "_shared_with", lambda *a: looks.append(1) or real(*a))
    ns = {"ROWS": [1, 2]}
    exec("def rows(n):\n    return ROWS\n", ns)
    rows = cash_instance.cache(ns["rows"])
    said = [_warnings_of(rows, n) for n in range(4)]
    assert len(looks) == 4, looks
    assert said[0] and not any(said[1:]), said


def test_a_hit_in_a_process_that_never_ran_the_call_says_it_too(tmp_path):
    # After a restart the hit is the first value the caller gets, already a
    # separate copy: a process that only ever hits is told as the miss was.
    from cash import Cash

    ns = {"ROWS": [1, 2]}
    exec("def wrap(n):\n    return {'rows': ROWS}\n", ns)
    first = Cash(cache_dir=str(tmp_path), register_magic=False)
    assert _warnings_of(first.cache(ns["wrap"]), 1)
    from cash.backends._writes import all_pending_writes

    for queue in all_pending_writes():
        queue.wait_all()
    later = Cash(cache_dir=str(tmp_path), register_magic=False)
    wrapped = later.cache(ns["wrap"])
    said = _warnings_of(wrapped, 1)
    assert wrapped.cache_info()["hits"] == 1, wrapped.cache_info()
    assert said and "the module global 'ROWS'" in said[0], said


def _store_then_hit(tmp_path, ns, name, stored, hit):
    """Store *name* for each of *stored* in one process; in another, the
    warnings of a hit of *hit* alone."""
    from cash import Cash
    from cash.backends._writes import all_pending_writes

    first = Cash(cache_dir=str(tmp_path), register_magic=False).cache(ns[name])
    for args in stored:
        _warnings_of(first, *args)
    for queue in all_pending_writes():
        queue.wait_all()
    later = Cash(cache_dir=str(tmp_path), register_magic=False).cache(ns[name])
    said = _warnings_of(later, *hit)
    assert later.cache_info()["hits"] == 1, later.cache_info()
    return said


def test_a_later_entry_sharing_another_object_says_so_on_its_own_hit(tmp_path):
    # get("a") warns first; get("b"), stored after, shares CONFIGS["b"]: a
    # process that only hits get("b") is told so, naming it.
    ns = {"CONFIGS": {"a": [1], "b": [2]}}
    exec("def get(name):\n    return CONFIGS[name]\n", ns)
    said = _store_then_hit(tmp_path, ns, "get", [("a",), ("b",)], ("b",))
    assert said and "CONFIGS['b']" in said[0], said


def test_a_later_entry_of_its_own_says_nothing_on_its_hit(tmp_path):
    # pick(True) is ROWS and warns; pick(False), stored after, is a fresh
    # list: a process that only hits it says nothing.
    ns = {"ROWS": [1, 2]}
    exec("def pick(flag):\n    return ROWS if flag else [0]\n", ns)
    assert not _store_then_hit(tmp_path, ns, "pick", [(True,), (False,)], (False,))


@pytest.mark.parametrize(
    ("label", "column"),
    [
        ("interval", "pd.arrays.IntervalArray.from_breaks(range(4))"),
        ("int64[pyarrow]", "pd.array([1, 2, 3], dtype='int64[pyarrow]')"),
        ("string[pyarrow]", "pd.array(list('abc'), dtype='string[pyarrow]')"),
    ],
)
def test_an_extension_column_of_a_global_frame_is_announced(cash_instance, monkeypatch, label, column):
    # The column holds the very extension array the frame's block does: a
    # write through it reaches the frame without copy-on-write.
    pd = pytest.importorskip("pandas")
    if "pyarrow" in column:
        pytest.importorskip("pyarrow")
    from cash.decorator import purity_checks

    monkeypatch.setattr(purity_checks, "is_cow_pandas", lambda value: False)
    ns = {"pd": pd}
    exec(f"DF = pd.DataFrame({{'c': {column}, 'x': [1.0, 2.0, 3.0]}})\ndef col(n):\n    return DF['c']\n", ns)
    if not any(
        v is w for v in purity_checks._extension_values(ns["col"](1)) for w in purity_checks._extension_values(ns["DF"])
    ):
        pytest.skip(f"this pandas copies {label}")
    assert _warnings_of(cash_instance.cache(ns["col"]), 1), f"no warning for {label}"
    exec("def fresh(n):\n    return DF['c'].copy()\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["fresh"]), 1)


@pytest.mark.parametrize(
    "made",
    [
        "pd.Timestamp('2024-01-01')",
        "pd.Timedelta('1D')",
        "pd.Period('2024-01')",
        "pd.NaT",
        "logging.getLogger('app.shared-result')",
    ],
)
def test_a_global_a_hit_gives_back_as_itself_says_nothing(cash_instance, made):
    pd = pytest.importorskip("pandas")
    import logging

    ns = {"pd": pd, "logging": logging}
    exec(f"HELD = {made}\ndef wrap(n):\n    return {{'h': HELD, 'n': n}}\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["wrap"]), 1)
