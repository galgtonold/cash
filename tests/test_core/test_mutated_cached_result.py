"""A cached result mutated in place and passed on is keyed by what it holds NOW.

Rounds 17-18: in a script, the decorator tags what it returns with the
producing call's identity, and nothing ever moves that tag. The argument hasher
trusted it twice -- a memo that skipped re-hashing a tagged frame, and the tag
itself as the identity of an object with no content hasher -- so

    q = make_params(); q.F = 0.03; run(q)     # got F=0.02's result
    df = load(); df.loc[0, "a"] = 100; total(df)

were served the pre-mutation answer. The tag is trusted only where the
notebook's statement layer keeps it current. For pandas >= 3, a frame's content
hash is reused while an exact copy-on-write check says the frame is unchanged,
so the common case (the same frame passed along a pipeline) still pays for one
hash, not one per call.
"""
from __future__ import annotations

import dataclasses
import gc

import pytest

from cash import Cash
from cash.core import _is_cow_pandas

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core

needs_cow = pytest.mark.skipif(
    not _is_cow_pandas(pd.DataFrame()), reason="pandas copy-on-write is not active")


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@dataclasses.dataclass
class Params:
    F: float = 0.02
    k: float = 0.06


def test_a_mutated_params_object_from_a_cached_call(c):
    """r18s2's sweep idiom: one returned params object, mutated per iteration."""
    @c.cache
    def make_params():
        return Params()

    @c.cache
    def run(p):
        return round(p.F * 1000 + p.k * 10, 6)

    q = make_params()
    got = []
    for f in (0.02, 0.03, 0.04):
        q.F = f
        got.append(run(q))
    assert got == [run.__wrapped__(Params(F=f)) for f in (0.02, 0.03, 0.04)]


def test_a_mutated_frame_from_a_cached_call(c):
    """The known-limitations example, now fixed for scripts."""
    @c.cache
    def load():
        return pd.DataFrame({"a": [1, 2, 3]})

    @c.cache
    def total(df):
        return int(df["a"].sum())

    df = load()
    assert total(df) == 6
    df.loc[0, "a"] = 100
    assert total(df) == 105


def _frame():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6], "s": ["x", "y", "z"],
                         "c": pd.Categorical(["p", "q", "p"])})


MUTATIONS = {
    "loc": lambda d: d.loc.__setitem__((0, "a"), 100.0),
    "iloc": lambda d: d.iloc.__setitem__((0, 1), 9),
    "at": lambda d: d.at.__setitem__((1, "s"), "q"),
    "column assign": lambda d: d.__setitem__("a", d["a"] * 2),
    "new column": lambda d: d.__setitem__("new", 1),
    "categorical": lambda d: d.loc.__setitem__((0, "c"), "q"),
    "fillna inplace": lambda d: (d.loc.__setitem__((0, "a"), np.nan), d.fillna(0, inplace=True)),
    "sort_values inplace": lambda d: d.sort_values("a", ascending=False, inplace=True),
    "drop inplace": lambda d: d.drop(columns="b", inplace=True),
    "rename inplace": lambda d: d.rename(columns={"a": "z"}, inplace=True),
    "columns reassigned": lambda d: setattr(d, "columns", ["p", "q", "r", "t"]),
    "index reassigned": lambda d: setattr(d, "index", [7, 8, 9]),
    "index renamed": lambda d: setattr(d.index, "name", "row"),
    "update": lambda d: d.update(pd.DataFrame({"a": [9.0]})),
    "insert": lambda d: d.insert(0, "k", 1),
    "pop": lambda d: d.pop("b"),
    "replace inplace": lambda d: d.replace(1.0, -1.0, inplace=True),
}


@needs_cow
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_every_in_place_mutation_is_seen_by_the_copy_on_write_check(c, mutation):
    @c.cache
    def summary(df):
        return (list(df.columns), [str(v) for v in df.to_numpy().ravel()], df.index.name)

    df = _frame()
    summary(df)
    MUTATIONS[mutation](df)
    assert summary(df) == summary.__wrapped__(df)


@needs_cow
def test_an_unchanged_frame_is_not_re_hashed(c, monkeypatch):
    """The point of the memo: the same frame through a pipeline hashes once."""
    calls = []
    real = Cash._try_hash_pandas

    def counting(value, type_name):
        calls.append(type_name)
        return real(value, type_name)

    monkeypatch.setattr(Cash, "_try_hash_pandas", staticmethod(counting))

    @c.cache
    def mean_a(df):
        return float(df["a"].mean())

    df = _frame()
    for _ in range(5):
        mean_a(df)
    assert len(calls) == 1, calls
    df.loc[0, "a"] = 50.0
    assert mean_a(df) == mean_a.__wrapped__(df)
    assert len(calls) == 2


@needs_cow
def test_the_memo_lets_go_of_a_collected_frame(c):
    """It holds a shallow copy of each frame; that copy must not outlive the frame."""
    @c.cache
    def n(df):
        return len(df)

    df = _frame()
    n(df)
    assert id(df) in c._frame_memo
    key = id(df)
    del df
    gc.collect()
    assert key not in c._frame_memo


def test_a_statement_maintained_tag_is_still_trusted(c, monkeypatch):
    """The notebook path: the statement layer re-tags on every change, so its
    tag stands in for the content and a big frame is not re-hashed per call."""
    calls = []
    real = Cash._try_hash_pandas

    def counting(value, type_name):
        calls.append(type_name)
        return real(value, type_name)

    monkeypatch.setattr(Cash, "_try_hash_pandas", staticmethod(counting))
    monkeypatch.setattr("cash.core._COW_PANDAS", False)     # isolate from the CoW memo

    @c.cache
    def n(df):
        return len(df)

    df = _frame()
    df._cash_lineage_hash = "L1"
    df._cash_lineage_src = "statement"
    n(df)
    n(df)
    assert len(calls) == 1, "a statement-maintained tag should have skipped the re-hash"


def test_a_decorator_tag_alone_is_not_trusted(c, monkeypatch):
    """The control for the above: the same tag written by the decorator."""
    monkeypatch.setattr("cash.core._COW_PANDAS", False)

    @c.cache
    def n(df):
        return int(df["a"].sum())

    df = _frame()
    df._cash_lineage_hash = "L1"
    df._cash_lineage_src = "decorator"
    assert n(df) == 6
    df.loc[0, "a"] = 100.0     # the tag does not move
    assert n(df) == n.__wrapped__(df)
