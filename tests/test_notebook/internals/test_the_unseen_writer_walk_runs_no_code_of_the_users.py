"""The walk that looks for a writing special method without a record of
what ran (`writers_without_a_trace`) reads objects by type and Python's own
slots: no property (``dtype``, ``columns``) or method (``to_numpy``) of the
user's runs, and its steps are bounded however often a value repeats. Two
lambdas on one line are told apart by where each starts."""

from __future__ import annotations

import sys

import pytest

from cash.notebook.statement.unseen_writes import SET_LATER, own_writes, writers_without_a_trace

RAN: list[str] = []


class Lazy:
    """A dataset wrapper whose attributes load on read."""

    @property
    def dtype(self):
        RAN.append("dtype")
        return None

    @property
    def dtypes(self):
        RAN.append("dtypes")
        return []

    @property
    def columns(self):
        RAN.append("columns")
        return []


class ArrayLike:
    dtype = None

    def to_numpy(self):
        RAN.append("to_numpy")
        return []

    def ravel(self):
        RAN.append("ravel")
        return []


def test_no_property_or_method_of_the_users_runs():
    import numpy as np

    ArrayLike.dtype = np.dtype(object)
    RAN.clear()
    ns = {"x": Lazy(), "lst": [Lazy(), ArrayLike()], "a": ArrayLike()}
    # Its properties write (``RAN.append``): found, read from their text.
    writers_without_a_trace("w = (type(x).__name__, lst, a)", ns)
    assert RAN == []


def _counted(monkeypatch):
    """Count what the walk opens: each value it is handed, and the items
    of the containers and arrays among them."""
    from cash.analysis import handed_callables as hc
    from cash.notebook.statement import unseen_writes

    work = {"values": 0, "items": 0}
    held, count = unseen_writes._held, unseen_writes._item_count

    def counted_held(value, kind, hc_):
        work["values"] += 1
        return held(value, kind, hc_)

    def counted_items(value, kind, hc_):
        n = count(value, kind, hc_)
        work["items"] += n
        return n

    monkeypatch.setattr(unseen_writes, "_held", counted_held)
    monkeypatch.setattr(unseen_writes, "_item_count", counted_items)
    return work, unseen_writes._OPEN_AT + 1, unseen_writes._SCAN_AT + hc.ITEM_TYPES_AT


def test_a_grid_of_repeated_values_is_bounded(monkeypatch):
    """3.3 s when every item of a 3000 x 3000 grid was opened. Its work is
    bounded, and the grid is taken as holding anything: a class of the
    user's with a writing special method counts (this module's `Lazy`)."""
    work, values, items = _counted(monkeypatch)
    found = writers_without_a_trace("w = g[0][0]", {"g": [[0] * 3000 for _ in range(3000)]})
    assert work["values"] <= values and work["items"] <= items, work
    assert any(fn.startswith("Lazy.") for fn, _ in found), found


class Plain:
    pass


def test_the_same_object_many_times_is_looked_at_once(monkeypatch):
    """The same object a million times: a million steps, then past the
    limit. Looked at once per list, nothing is found: it holds nothing."""
    one = Plain()
    work, values, items = _counted(monkeypatch)
    found = writers_without_a_trace("w = g[0][0]", {"g": [[one] * 1000 for _ in range(1000)]})
    assert work["values"] <= 2 * 1000 + 1 and work["items"] <= items, work
    assert found == []


def test_the_items_of_arrays_count_toward_the_limit(monkeypatch):
    """A list of 1000 arrays of 99,000 objects each was 2 s a miss: the
    arrays' items did not count. They count before any is read."""
    np = pytest.importorskip("numpy")
    column = np.array(["x"] * 99_000, dtype=object)
    work, values, items = _counted(monkeypatch)
    found = writers_without_a_trace("w = g[0]", {"g": [column[:] for _ in range(1000)]})
    assert work["items"] <= items, work
    assert any(fn.startswith("Lazy.") for fn, _ in found), found


ARRAY_CALLS: list = []


class W:
    def __add__(self, o):
        ARRAY_CALLS.append(1)
        return 0


def _holders():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    class SeriesOfMine(pd.Series):
        @property
        def _constructor(self):
            return SeriesOfMine

    class ArrayOfMine(np.ndarray):
        pass

    def objects():
        return np.array([W(), W()], dtype=object)

    return {
        "a Series": pd.Series([W(), W()], dtype=object),
        "a Series subclass": SeriesOfMine([W(), W()], dtype=object),
        "an Index": pd.Index([W(), W()], dtype=object),
        "an ndarray subclass": objects().view(ArrayOfMine),
        "a masked array": np.ma.masked_array(objects()),
        "a Categorical": pd.Categorical([W(), W()]),
        "a DataFrame": pd.DataFrame({"a": pd.Series([W(), W()], dtype=object)}),
    }


@pytest.mark.parametrize(
    "holder",
    [
        "a Series",
        "a Series subclass",
        "an Index",
        "an ndarray subclass",
        "a masked array",
        "a Categorical",
        "a DataFrame",
    ],
)
def test_objects_in_an_array_of_any_kind_are_found(holder):
    """Read by module and class name, a Series or ndarray subclass, a
    masked array and a Categorical were not opened, and ``h + 1`` was
    served after a restart with its ``__add__`` not run."""
    found = writers_without_a_trace("w = h + 1", {"h": _holders()[holder]})
    assert [fn for fn, _ in found] == ["W.__add__"], found


def test_a_property_of_an_array_subclass_does_not_run():
    np = pytest.importorskip("numpy")

    class Watched(np.ndarray):
        @property
        def size(self):
            RAN.append("size")
            return 0

    RAN.clear()
    held = np.array([W()], dtype=object).view(Watched)
    found = writers_without_a_trace("w = h + 1", {"h": held})
    # Its property is read from its text (it writes), never run.
    assert RAN == [] and "W.__add__" in [fn for fn, _ in found]


def test_strings_are_not_read_as_objects():
    """A pandas 3 ``str`` column holds no object of the user's: its items
    are not turned into Python strings on every miss."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": pd.Series(["x"] * 10, dtype="string"), "b": [1] * 10})
    from cash.notebook.statement import unseen_writes

    assert unseen_writes._array_items(frame["a"], "series", None) == []


calls: list = []
H = {"g": lambda o: 1, "f": lambda o: calls.append(1) or 1}
G = {"f": lambda o: calls.append(1) or 1, "g": lambda o: 1}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="positions to tell them apart are there from 3.11")
def test_two_lambdas_on_one_line_are_read_each_as_itself():
    assert own_writes(H["f"].__code__) == {"calls"}
    assert own_writes(H["g"].__code__) == set()
    assert own_writes(G["f"].__code__) == {"calls"}
    assert own_writes(G["g"].__code__) == set()


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="3.10 has no positions to tell them apart")
def test_two_lambdas_on_one_line_are_unreadable_before_3_11():
    assert own_writes(H["g"].__code__) == {"?"}


def test_a_method_set_on_a_class_later_is_said_to_be_unseeable():
    class Vec:
        pass

    Vec.__add__ = lambda self, o: 10
    found = writers_without_a_trace("w = v + 1", {"v": Vec()})
    assert [targets for _, targets in found] == [(SET_LATER,)]


def test_a_method_set_later_is_named_as_the_class_holds_it():
    """``def _add(...)`` set as ``Vec.__add__`` was said as ``Vec._add()``."""

    class Vec:
        pass

    def _add(self, o):
        return 10

    Vec.__add__ = _add
    found = writers_without_a_trace("w = v + 1", {"v": Vec()})
    assert [fn for fn, _ in found] == ["test_a_method_set_later_is_named_as_the_class_holds_it.<locals>.Vec.__add__"]


class PassesFor:
    """A proxy: its ``__class__`` says what it wraps."""

    @property
    def __class__(self):
        RAN.append("__class__")
        return dict


class Watching(type):
    def __getattribute__(cls, name):
        RAN.append(name)
        return super().__getattribute__(name)


class Kept(metaclass=Watching):
    def __add__(self, o):
        calls.append(1)
        return 0


def test_a_class_property_or_a_metaclass_hook_does_not_run():
    RAN.clear()
    found = writers_without_a_trace("w = (p, k + 1, K)", {"p": PassesFor(), "k": Kept(), "K": Kept})
    assert RAN == [] and "Kept.__add__" in [fn for fn, _ in found], (RAN, found)
