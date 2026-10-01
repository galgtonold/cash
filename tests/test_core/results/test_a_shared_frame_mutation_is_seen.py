"""Every way to change a frame is seen, now that the memo also covers frames
that share their data (``pd.DataFrame(ndarray)``, ``read_csv``, a groupby
result, or a frame another pandas object shares blocks with).

Each case first checks that the memo holds the frame -- otherwise the test
would pass without testing it -- then changes the frame and asks again.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash, content_hashers
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


def _numbers():
    return pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), columns=["a", "b", "c"])


def _from_csv(tmp_path):
    path = tmp_path / "x.csv"
    _numbers().to_csv(path, index=False)
    return pd.read_csv(path, dtype=float), None


def _grouped(tmp_path):
    frame = _numbers().assign(k=[0, 0, 1, 2])
    return frame.groupby("k")[["a", "b", "c"]].sum(), None


def _sharing(share):
    def make(tmp_path):
        frame = _numbers()
        return frame, share(frame)

    return make


#: ``make(tmp_path) -> (frame, what else the test keeps alive)``
SOURCES = {
    "from numpy": lambda tmp_path: (_numbers(), None),
    "read_csv": _from_csv,
    "groupby": _grouped,
    "while d.assign(z=1) lives": _sharing(lambda d: d.assign(z=1)),
    "while d[cols] lives": _sharing(lambda d: d[["a", "b"]]),
    "while d['a'] lives": _sharing(lambda d: d["a"]),
    "while d.reset_index() lives": _sharing(lambda d: d.reset_index()),
}


def _first(frame):
    return frame.columns[0]


MUTATIONS = {
    "iloc": lambda d, other: d.iloc.__setitem__((0, 0), 100.0),
    "iat": lambda d, other: d.iat.__setitem__((0, 1), 100.0),
    "loc": lambda d, other: d.loc.__setitem__((d.index[0], _first(d)), 100.0),
    "at": lambda d, other: d.at.__setitem__((d.index[0], _first(d)), 100.0),
    "column assign": lambda d, other: d.__setitem__(_first(d), d[_first(d)] * 2),
    "new column": lambda d, other: d.__setitem__("new", 1.0),
    "fillna inplace": lambda d, other: (d.iloc.__setitem__((0, 0), np.nan), d.fillna(-1.0, inplace=True)),
    "replace inplace": lambda d, other: d.replace(d.iloc[0, 0], -1.0, inplace=True),
    "clip inplace": lambda d, other: d.clip(upper=2.0, inplace=True),
    "where inplace": lambda d, other: d.where(d > d.iloc[0, 0], -1.0, inplace=True),
    "mask inplace": lambda d, other: d.mask(d >= d.iloc[0, 0], -1.0, inplace=True),
    "sort_values inplace": lambda d, other: d.sort_values(_first(d), ascending=False, inplace=True),
    "+=": lambda d, other: d.__iadd__(1.0),
    "update": lambda d, other: d.update(pd.DataFrame({_first(d): [99.0]}, index=d.index[:1])),
    "drop inplace": lambda d, other: d.drop(columns=_first(d), inplace=True),
    "index renamed": lambda d, other: setattr(d.index, "name", "row"),
    "column .array": lambda d, other: d[_first(d)].array.__setitem__(0, 100.0),
    "iloc column .array": lambda d, other: d.iloc[:, 1].array.__setitem__(0, 100.0),
}


@pytest.fixture
def seen(monkeypatch):
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def seen(frame):
        return frame.to_csv()

    hashed: list[str] = []
    real = content_hashers.hash_pandas
    monkeypatch.setattr(content_hashers, "hash_pandas", lambda value: hashed.append(1) or real(value))
    seen.hashed = hashed
    return seen


@pytest.mark.parametrize("mutation", list(MUTATIONS), ids=list(MUTATIONS))
@pytest.mark.parametrize("source", list(SOURCES), ids=list(SOURCES))
def test_a_change_is_seen(source, mutation, seen, tmp_path):
    frame, other = SOURCES[source](tmp_path)
    seen(frame)
    seen(frame)
    assert len(seen.hashed) == 1, "the memo does not hold this frame: the test would prove nothing"
    before = seen.__wrapped__(frame)
    MUTATIONS[mutation](frame, other)
    assert seen.__wrapped__(frame) != before  # the mutation did change the frame
    assert seen(frame) == seen.__wrapped__(frame)
    del other


@pytest.mark.parametrize("source", [s for s in SOURCES if s.startswith("while")])
def test_a_write_to_the_sharing_object_leaves_the_frame_as_it_was(source, seen, tmp_path):
    """Copy-on-write: the other object copies before it is written, so the
    frame is unchanged and its memoised hash still right."""
    frame, other = SOURCES[source](tmp_path)
    before = seen(frame)
    other.iloc[0] = 100.0
    assert seen.__wrapped__(frame) == before
    assert seen(frame) == before
    assert len(seen.hashed) == 1


@pytest.mark.parametrize("source", [s for s in SOURCES if s.startswith("while")])
def test_a_write_through_the_sharing_objects_array_is_seen(source, seen, tmp_path):
    """``.array`` of the other object writes the shared memory in place."""
    frame, other = SOURCES[source](tmp_path)
    seen(frame)
    seen(frame)
    column = other if other.ndim == 1 else other[other.columns[-1] if "z" not in other else "a"]
    before = seen.__wrapped__(frame)
    column.array[0] = 100.0
    assert seen.__wrapped__(frame) != before
    assert seen(frame) == seen.__wrapped__(frame)


READ_ONLY = {
    "values": lambda d: d.values,
    "to_numpy()": lambda d: d.to_numpy(),
    "to_numpy(copy=False)": lambda d: d.to_numpy(copy=False),
    "np.asarray": lambda d: np.asarray(d),
    "column values": lambda d: d[d.columns[0]].values,
    "column to_numpy(copy=False)": lambda d: d[d.columns[0]].to_numpy(copy=False),
}


@pytest.mark.parametrize("handle", list(READ_ONLY.values()), ids=list(READ_ONLY))
@pytest.mark.parametrize("source", list(SOURCES), ids=list(SOURCES))
def test_numpy_views_pandas_hands_out_do_not_write_the_frame(source, handle, seen, tmp_path):
    frame, other = SOURCES[source](tmp_path)
    before = seen(frame)
    view = handle(frame)
    if np.shares_memory(view, frame._mgr.blocks[0].values):
        with pytest.raises(ValueError, match="read-only"):
            view[(0,) * view.ndim] = 100.0
    else:  # a frame of several blocks hands out a copy
        view[(0,) * view.ndim] = 100.0
    del view
    assert seen(frame) == before == seen.__wrapped__(frame)
    del other


BUILT_OVER = {
    "DataFrame(arr, copy=False)": lambda arr: pd.DataFrame(arr, copy=False),
    "DataFrame(arr[:, :2], copy=False)": lambda arr: pd.DataFrame(arr[:, :2], copy=False),
    "DataFrame({'a': arr[:, 0]}, copy=False)": lambda arr: pd.DataFrame({"a": arr[:, 0]}, copy=False),
    "Series(arr[:, 0], copy=False)": lambda arr: pd.Series(arr[:, 0], copy=False),
}


@pytest.mark.parametrize("dropped", [False, True], ids=["kept", "dropped before the call"])
@pytest.mark.parametrize("build", list(BUILT_OVER.values()), ids=list(BUILT_OVER))
def test_a_write_through_the_array_the_frame_was_built_over(build, dropped, tmp_path):
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def seen(frame):
        return frame.to_csv()

    arr = np.arange(12, dtype=float).reshape(4, 3)
    frame = build(arr)
    seen(frame)
    seen(frame)
    arr[0, 0] = 100.0
    if dropped:
        del arr
    assert "100.0" in seen.__wrapped__(frame)
    assert seen(frame) == seen.__wrapped__(frame)


def test_a_write_through_a_view_the_caller_kept(tmp_path):
    """The caller keeps a view, not the array that owns the memory: the
    view's reference to that array is the one outside pandas."""
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def seen(frame):
        return frame.to_csv()

    view = np.arange(12, dtype=float).reshape(4, 3)[:, :2]
    frame = pd.DataFrame(view.T.T, copy=False)
    seen(frame)
    seen(frame)
    view[0, 0] = 100.0
    assert seen(frame) == seen.__wrapped__(frame)


def test_an_index_pandas_looked_labels_up_in_is_hashed_once(seen):
    """``.loc`` builds a lookup table that holds the index's array; that is
    pandas' own reference, not a writer's."""
    frame = pd.DataFrame({"a": np.arange(50, dtype=float)}, index=np.linspace(0.0, 1.0, 50))
    frame.loc[frame.index[3]]
    for _ in range(4):
        seen(frame)
    assert len(seen.hashed) == 1, seen.hashed
