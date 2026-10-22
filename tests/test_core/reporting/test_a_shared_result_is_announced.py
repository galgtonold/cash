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

import functools
import sys
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


def test_a_singleton_whose_reduce_carries_state_is_announced(cash_instance, monkeypatch):
    # Loading its state on a hit rewrites the live object's attributes: it
    # is not handed back as it was, and the check never applies the state.
    ns: dict = {"__name__": __name__}
    exec(
        "class Settings:\n"
        "    _inst = None\n"
        "    def __new__(cls):\n"
        "        if cls._inst is None:\n"
        "            cls._inst = super().__new__(cls)\n"
        "        return cls._inst\n"
        "    def __reduce__(self):\n"
        "        return (Settings, (), dict(self.__dict__))\n"
        "    def __setstate__(self, state):\n"
        "        self.__dict__.update(state)\n"
        "    def __eq__(self, other):\n"
        "        return isinstance(other, Settings) and self.__dict__ == other.__dict__\n"
        "    __hash__ = None\n"
        "SETTINGS = Settings()\n"
        "SETTINGS.tags = ['a']\n"
        "def current(n):\n"
        "    return {'settings': SETTINGS, 'n': n}\n",
        ns,
    )
    # Found by its name, as a class of a module is: pickle rebuilds it.
    monkeypatch.setattr(sys.modules[__name__], "Settings", ns["Settings"], raising=False)
    assert _warnings_of(cash_instance.cache(ns["current"]), 1)


def test_an_object_rebuilt_by_name_says_nothing_nor_does_what_it_holds(cash_instance, monkeypatch):
    ns: dict = {"__name__": __name__}
    exec(
        "def get_registry():\n"
        "    return REGISTRY\n"
        "class Registry:\n"
        "    def __init__(self):\n"
        "        self.entries = {'a': [1]}\n"
        "    def __reduce__(self):\n"
        "        return (get_registry, ())\n"
        "    def __eq__(self, other):\n"
        "        return isinstance(other, Registry) and other.entries == self.entries\n"
        "    __hash__ = None\n"
        "REGISTRY = Registry()\n"
        "def whole(n):\n"
        "    return {'reg': REGISTRY, 'n': n}\n"
        "def entries(n):\n"
        "    return REGISTRY.entries\n",
        ns,
    )
    monkeypatch.setattr(sys.modules[__name__], "get_registry", ns["get_registry"], raising=False)
    assert not _warnings_of(cash_instance.cache(ns["whole"]), 1)
    said = _warnings_of(cash_instance.cache(ns["entries"]), 1)
    assert said and "REGISTRY.entries" in said[0], said


def test_a_class_rebuilt_through_a_partial_is_announced_and_never_built_by_the_check(cash_instance, monkeypatch):
    # `functools.partial` reports `functools` as its module whatever it
    # wraps: the check called it, and `Client.__init__` ran on every store.
    ns: dict = {"__name__": __name__, "functools": functools, "BUILT": []}
    exec(
        "class Client:\n"
        "    def __init__(self, url, timeout=1):\n"
        "        BUILT.append(url)\n"
        "        self.url, self.timeout = url, timeout\n"
        "    def __reduce__(self):\n"
        "        return (functools.partial(Client, timeout=self.timeout), (self.url,))\n"
        "    def __eq__(self, other):\n"
        "        return isinstance(other, Client) and other.url == self.url\n"
        "    __hash__ = None\n"
        "CLIENT = Client('db://x', timeout=9)\n"
        "def connect():\n"
        "    BUILT.append('connect')\n"
        "    return CLIENT\n"
        "class Conn:\n"
        "    def __reduce__(self):\n"
        "        return (functools.partial(connect), ())\n"
        "CONN = Conn()\n",
        ns,
    )
    monkeypatch.setattr(sys.modules[__name__], "Client", ns["Client"], raising=False)
    from cash.decorator import purity_checks

    ns["BUILT"].clear()
    assert not purity_checks._comes_back_as_itself(ns["CLIENT"])
    assert not purity_checks._comes_back_as_itself(ns["CONN"])
    assert ns["BUILT"] == [], "the check ran the user's code"


def test_the_check_never_calls_a_library_function_a_reduce_names(tmp_path):
    from cash.decorator import purity_checks

    path = tmp_path / "sink.txt"
    path.write_text("kept", encoding="utf-8")

    class Sink:
        def __reduce__(self):
            return (open, (str(path), "w"))

    assert not purity_checks._comes_back_as_itself(Sink())
    assert path.read_text(encoding="utf-8") == "kept", "the check opened the file for writing"


def test_a_logger_is_looked_up_and_none_is_made():
    import logging

    from cash.decorator import purity_checks

    made = logging.getLogger("app.shared-result.made")
    assert purity_checks._comes_back_as_itself(made)
    assert purity_checks._comes_back_as_itself(logging.getLogger())
    loose = logging.Logger("app.shared-result.never-registered")
    assert not purity_checks._comes_back_as_itself(loose)
    assert "app.shared-result.never-registered" not in logging.Logger.manager.loggerDict


def test_a_global_not_in_the_result_is_never_asked_its_reduce(cash_instance, monkeypatch):
    from cash.decorator import purity_checks

    asked = []
    real = purity_checks._comes_back_as_itself
    monkeypatch.setattr(purity_checks, "_comes_back_as_itself", lambda v: asked.append(v) or real(v))
    import logging

    ns = {"logging": logging, "VOCAB": {f"w{i}" for i in range(1000)}}
    exec(
        "A = logging.getLogger('a.shared'); B = logging.getLogger('b.shared')\n"
        "C = logging.getLogger('c.shared'); D = logging.getLogger('d.shared')\n"
        "def window(n):\n"
        "    _ = len(VOCAB)\n"
        "    return {'a': A, 'b': B, 'c': C, 'd': D, 'n': n}\n",
        ns,
    )
    assert not _warnings_of(cash_instance.cache(ns["window"]), 1)
    assert all(v is not ns["VOCAB"] for v in asked)


@pytest.mark.parametrize(
    "body",
    ["return {'train': X[:8], 'test': X[8:]}", "return [X[1:4]]", "return {'v': {'w': X[1:4]}}"],
)
def test_a_view_of_a_global_array_inside_the_result_is_announced(cash_instance, body):
    ns = {"X": np.arange(10.0)}
    exec(f"def split(n):\n    {body}\n", ns)
    said = _warnings_of(cash_instance.cache(ns["split"]), 1)
    assert said and "X" in said[0], said


@pytest.mark.parametrize("column", ["pd.array([1, 2, 3], dtype='Int64')", "pd.date_range('2024', periods=3, tz='UTC')"])
@pytest.mark.parametrize("attr", ["array", "values"])
def test_the_array_of_a_global_frames_extension_column_is_announced(cash_instance, column, attr):
    # The frame's own array: a write through it reaches the frame, under
    # copy-on-write too.
    pd = pytest.importorskip("pandas")
    ns = {"pd": pd}
    exec(f"DF = pd.DataFrame({{'c': {column}}})\ndef col(n):\n    return DF['c'].{attr}\n", ns)
    got = ns["col"](1)
    if type(got).__name__ == "ndarray":
        pytest.skip("this pandas hands back a numpy array")
    assert _warnings_of(cash_instance.cache(ns["col"]), 1)
    exec(f"def fresh(n):\n    return DF['c'].{attr}.copy()\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["fresh"]), 1)


def test_an_attribute_of_an_argument_is_announced(cash_instance, monkeypatch):
    ns: dict = {"__name__": __name__}
    exec(
        "class Table:\n"
        "    def __init__(self):\n"
        "        self.rows = [1, 2]\n"
        "    def __eq__(self, other):\n"
        "        return isinstance(other, Table) and other.rows == self.rows\n"
        "    __hash__ = None\n"
        "def rows_of(t):\n"
        "    return {'rows': t.rows}\n",
        ns,
    )
    monkeypatch.setattr(sys.modules[__name__], "Table", ns["Table"], raising=False)
    table = ns["Table"]()
    said = _warnings_of(cash_instance.cache(ns["rows_of"]), table)
    assert said and "t.rows" in said[0], said


_INITS: list = []


class _Client:
    """Rebuilt from its config: its ``__init__`` opens what it opens."""

    def __init__(self, url):
        _INITS.append(url)
        self.url = url

    def __reduce__(self):
        return (_Client, (self.url,))


class _Registry:
    """A singleton whose ``__init__`` re-initialises it: a hit's unpickle
    calls ``_Registry()`` and wipes the live one."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __init__(self):
        self.entries = {}

    def __reduce__(self):
        return (_Registry, ())

    # Compared by value: one compared by identity is a sentinel, handed back
    # as itself.
    def __eq__(self, other):
        return isinstance(other, _Registry) and other.entries == self.entries

    __hash__ = None


def test_the_check_never_calls_a_class_its_reduce_names(cash_instance):
    from cash.decorator import purity_checks

    client = _Client("db://x")
    del _INITS[:]
    assert not purity_checks._comes_back_as_itself(client)
    assert _INITS == []


def test_a_singleton_a_hit_re_initialises_is_announced(cash_instance):
    reg = _Registry()
    reg.entries["model"] = "trained"
    ns = {"REG": reg, "__name__": __name__}
    exec("def with_reg(n):\n    return {'reg': REG, 'n': n}\n", ns)
    assert _warnings_of(cash_instance.cache(ns["with_reg"]), 1)


@pytest.mark.parametrize(
    "body",
    [
        "return DF['a'].array",
        "return {'v': DF['a'].array}",
        "return DF['o'].array",
        "return df['a'].array",
        "return df['i'].array",
        "return df['i'].values",
        "return df['k'].values",
        "return {'v': df['i'].array}",
    ],
)
def test_an_array_a_write_through_reaches_the_frame_is_announced(cash_instance, body):
    # A numpy column's ``.array`` is over the block's own writable memory,
    # copy-on-write or not; and an argument frame is looked at as a global.
    pd = pytest.importorskip("pandas")

    def frame():
        return pd.DataFrame(
            {
                "a": np.arange(6.0),
                "o": pd.Series(list("abcdef"), dtype=object),
                "i": pd.array(range(6), dtype="Int64"),
                "k": pd.Categorical(list("ababab")),
            }
        )

    ns = {"DF": frame()}
    exec(f"def f(df, n):\n    {body}\n", ns)
    said = _warnings_of(cash_instance.cache(ns["f"]), frame(), 1)
    assert said, body


@pytest.mark.parametrize("dtype", ["pd.CategoricalDtype(['a', 'b'])", "pd.Int64Dtype()"])
def test_a_pandas_dtype_held_in_a_global_says_nothing(cash_instance, dtype):
    pd = pytest.importorskip("pandas")
    ns = {"pd": pd}
    exec(f"G = {dtype}\ndef f(n):\n    return {{'v': G, 'n': n}}\n", ns)
    assert not _warnings_of(cash_instance.cache(ns["f"]), 1)
