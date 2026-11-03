"""A frame with categorical columns keeps its digest until it is written.

The content-hash memo vouched for no categorical, since ``.values`` hands
out the column's own ``Categorical``, writable: a pricing script passing
one 450k-row frame of four categorical columns to four cached functions
read it in full every call, about 0.15 s each, on runs where every call was
a hit. A categorical keeps its values as codes in an ndarray and its labels
in an Index, which the memo checks as it checks a numpy block and an axis;
every handle pandas gives out to them is recorded.

Each way a categorical column changes is pinned below: its codes written in
place through each handle (and through one kept across calls), its
categories renamed, added, removed, reordered or replaced, a pandas write
to it, the column reassigned, and the categories' own labels written.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash import Cash
from cash.backends import InMemoryBackend
from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

N = 5_000


def _frame(categories=("x", "y", "z")) -> pd.DataFrame:
    labels = [categories[i % 2] for i in range(N)]
    return pd.DataFrame(
        {
            "c": pd.Categorical(labels, categories=list(categories)),
            "v": np.arange(N, dtype=float),
        }
    )


def _reads(monkeypatch) -> list:
    reads: list = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(obj.shape)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    return reads


def _flip_and_write(view, value):
    """What people do about numpy's "assignment destination is read-only"."""
    view.flags.writeable = True
    view[0] = value


def _write_lands(write, target) -> bool:
    """Run *write*; False when pandas refuses it as a write to a read-only
    array. pandas 3.1 hands out an Index's arrays read-only, so a write
    through an index over a column no longer reaches the column there."""
    try:
        write(target)
    except ValueError as exc:
        if "read-only" not in str(exc):
            raise
        return False
    return True


def test_an_unchanged_frame_of_categoricals_is_read_once(monkeypatch):
    frame = _frame()
    reads = _reads(monkeypatch)
    first = cow_frame_digest(frame)
    assert all(cow_frame_digest(frame) == first for _ in range(3))
    assert len(reads) == 1
    assert first == builtin_hash(frame)


@pytest.mark.parametrize(
    "categories", [("x", "y", "z"), (1, 2, 3), (1.5, 2.5, 3.5)], ids=["strings", "integers", "floats"]
)
def test_categories_of_any_dtype_are_vouched_for(monkeypatch, categories):
    frame = _frame(categories)
    reads = _reads(monkeypatch)
    cow_frame_digest(frame)
    cow_frame_digest(frame)
    assert len(reads) == 1


def test_a_filtered_and_reindexed_split_is_read_once(monkeypatch):
    """What ``df[~is_test].reset_index(drop=True)`` hands on."""
    frame = _frame()
    train = frame[frame["v"] % 4 != 0].reset_index(drop=True)
    reads = _reads(monkeypatch)
    cow_frame_digest(train)
    cow_frame_digest(train)
    assert len(reads) == 1


def test_a_categorical_index_is_vouched_for(monkeypatch):
    frame = _frame().set_index("c")
    reads = _reads(monkeypatch)
    cow_frame_digest(frame)
    cow_frame_digest(frame)
    assert len(reads) == 1


CODE_WRITES = {
    "series values": lambda f: f["c"].values.__setitem__(0, "z"),
    "series array": lambda f: f["c"].array.__setitem__(0, "z"),
    "pd.array without a copy": lambda f: pd.array(f["c"], copy=False).__setitem__(0, "z"),
    "Categorical without a copy": lambda f: pd.Categorical(f["c"], copy=False).__setitem__(0, "z"),
    "Categorical without a copy, by keyword": lambda f: pd.Categorical(values=f["c"], copy=False).__setitem__(0, "z"),
    "codes made writable": lambda f: _flip_and_write(f["c"].array.codes, 2),
    "codes of values made writable": lambda f: _flip_and_write(f["c"].values.codes, 2),
    "codes of an index over the column made writable": lambda f: _flip_and_write(pd.CategoricalIndex(f["c"]).codes, 2),
    "a view of the categorical": lambda f: f["c"].values.view().__setitem__(0, "z"),
    "codes viewed as another dtype": lambda f: f["c"].values.view("i1").__setitem__(0, 2),
    "sorted in place": lambda f: f["c"].values.sort_values(inplace=True, ascending=False),
    "an index over the column, its values": lambda f: pd.CategoricalIndex(f["c"]).values.__setitem__(0, "z"),
    "an index over the column, its array": lambda f: pd.Index(f["c"]).array.__setitem__(0, "z"),
}


@pytest.mark.parametrize("write", list(CODE_WRITES.values()), ids=list(CODE_WRITES))
def test_codes_written_in_place_are_read_again(write):
    frame = _frame()
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    if not _write_lands(write, frame):  # refused: the frame is as it was
        assert builtin_hash(frame) == before == cow_frame_digest(frame)
        return
    assert builtin_hash(frame) != before  # the write reached the frame
    assert cow_frame_digest(frame) == builtin_hash(frame)


@pytest.mark.parametrize(
    "keep, write",
    [
        (lambda f: f["c"].values, lambda kept: kept.__setitem__(0, "z")),
        (lambda f: pd.Categorical(f["c"], copy=False), lambda kept: kept.__setitem__(0, "z")),
        (lambda f: f["c"].array.codes, lambda kept: _flip_and_write(kept, 2)),
        (lambda f: f["c"].values.view("i1"), lambda kept: kept.__setitem__(0, 2)),
    ],
    ids=["values", "Categorical without a copy", "codes", "codes viewed as another dtype"],
)
def test_a_handle_kept_across_calls_is_seen(keep, write):
    """A handle taken before the frame is first hashed, kept, and written
    after the memo has seen the frame twice."""
    frame = _frame()
    kept = keep(frame)
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    write(kept)
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


CATEGORY_CHANGES = {
    "renamed": lambda f: f.__setitem__("c", f["c"].cat.rename_categories(["p", "q", "r"])),
    "renamed by a mapping": lambda f: f.__setitem__("c", f["c"].cat.rename_categories({"x": "w"})),
    "added": lambda f: f.__setitem__("c", f["c"].cat.add_categories(["w"])),
    "removed": lambda f: f.__setitem__("c", f["c"].cat.remove_categories(["z"])),
    "unused removed": lambda f: f.__setitem__("c", f["c"].cat.remove_unused_categories()),
    "reordered": lambda f: f.__setitem__("c", f["c"].cat.reorder_categories(["z", "y", "x"])),
    "set": lambda f: f.__setitem__("c", f["c"].cat.set_categories(["y", "x"])),
    "set with rename": lambda f: f.__setitem__("c", f["c"].cat.set_categories(["a", "b", "c"], rename=True)),
    "ordered": lambda f: f.__setitem__("c", f["c"].cat.as_ordered()),
    "astype to a new dtype": lambda f: f.__setitem__(
        "c", f["c"].astype(pd.CategoricalDtype(["y", "x", "z"], ordered=True))
    ),
    "isetitem": lambda f: f.isetitem(0, f["c"].cat.rename_categories(["p", "q", "r"])),
}


@pytest.mark.parametrize("change", list(CATEGORY_CHANGES.values()), ids=list(CATEGORY_CHANGES))
def test_changed_categories_are_read_again(change):
    frame = _frame()
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    change(frame)
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


PANDAS_WRITES = {
    "loc": lambda f: f.loc.__setitem__((0, "c"), "z"),
    "iloc": lambda f: f.iloc.__setitem__((0, 0), "z"),
    "iat": lambda f: f.iat.__setitem__((0, 0), "z"),
    "replace in place": lambda f: f.replace({"c": {"x": "z"}}, inplace=True),
    "where in place": lambda f: f.where(
        f["v"] > 0, f.assign(c=pd.Categorical(["z"] * N, categories=["x", "y", "z"])), inplace=True
    ),
    "sort in place": lambda f: f.sort_values("c", ascending=False, inplace=True),
    "update": lambda f: f.update(pd.DataFrame({"c": ["z"]}, index=[0])),
    "column reassigned": lambda f: f.__setitem__("c", pd.Categorical(["z"] * N, categories=["x", "y", "z"])),
    "column made strings": lambda f: f.__setitem__("c", f["c"].astype(str)),
    "column popped and put back": lambda f: f.insert(0, "c", f.pop("c").cat.rename_categories(["p", "q", "r"])),
}


@pytest.mark.parametrize("write", list(PANDAS_WRITES.values()), ids=list(PANDAS_WRITES))
def test_a_pandas_write_to_the_column_is_read_again(write):
    frame = _frame()
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    write(frame)
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


CATEGORY_LABEL_WRITES = {
    "values made writable": lambda f: _flip_and_write(f["c"].cat.categories.values, 100),
    "array": lambda f: f["c"].cat.categories.array.__setitem__(0, 100),
    "to_numpy made writable": lambda f: _flip_and_write(f["c"].array.categories.to_numpy(), 100),
}


@pytest.mark.parametrize("write", list(CATEGORY_LABEL_WRITES.values()), ids=list(CATEGORY_LABEL_WRITES))
def test_labels_written_into_the_categories_are_read_again(write):
    """Integer categories are an ndarray behind an Index, written through
    the handles any Index gives out."""
    frame = _frame((1, 2, 3))
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    try:
        write(frame)
    except (ValueError, TypeError) as exc:  # pandas refused the write: nothing changed
        pytest.skip(f"pandas refuses this write: {exc}")
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


@pytest.mark.parametrize(
    "write",
    [lambda i: i.values.__setitem__(0, "z"), lambda i: _flip_and_write(i.codes, 2)],
    ids=["values", "codes made writable"],
)
def test_a_write_to_a_categorical_index_is_read_again(write):
    frame = _frame().set_index("c")
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    if not _write_lands(write, frame.index):  # refused: the frame is as it was
        assert builtin_hash(frame) == before == cow_frame_digest(frame)
        return
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


def _categorical_of(frame):
    return next(block.values for block in frame._mgr.blocks if isinstance(block.values, pd.Categorical))


def test_a_new_categorical_at_the_old_one_s_address_is_read_again():
    """The memo's shallow copy holds a categorical of its own over the same
    codes and dtype, not the frame's: once the column is replaced, a later
    categorical can be given the old one's address. The codes and the dtype
    it keeps are what tell them apart."""
    frame = _frame()
    before = cow_frame_digest(frame)
    address = id(_categorical_of(frame))
    for i in range(5_000):
        frame["c"] = frame["c"].cat.rename_categories([f"p{i}", f"q{i}", f"r{i}"])
        if id(_categorical_of(frame)) == address:
            break
    else:
        pytest.skip("no categorical was given the old address")
    assert builtin_hash(frame) != before
    assert cow_frame_digest(frame) == builtin_hash(frame)


def test_a_cached_function_sees_a_category_written_between_calls():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def counts(df):
        return df["c"].astype(str).value_counts().to_dict()

    frame = _frame()
    assert counts(frame) == {"x": N // 2, "y": N // 2}
    assert counts(frame) == {"x": N // 2, "y": N // 2}
    frame["c"].values[0] = "z"
    assert counts(frame) == {"x": N // 2 - 1, "y": N // 2, "z": 1}
    frame["c"] = frame["c"].cat.rename_categories(["p", "q", "r"])
    assert counts(frame) == {"p": N // 2 - 1, "q": N // 2, "r": 1}


def test_a_categorical_is_read_every_time_when_its_handles_are_not_recorded(monkeypatch):
    """Under a pandas whose ``codes`` or constructor cash cannot wrap, a
    categorical is not vouched for."""
    monkeypatch.setattr(arg_hashing, "_CATEGORICALS_WATCHED", [False])
    frame = _frame()
    reads = _reads(monkeypatch)
    cow_frame_digest(frame)
    cow_frame_digest(frame)
    assert len(reads) == 2
