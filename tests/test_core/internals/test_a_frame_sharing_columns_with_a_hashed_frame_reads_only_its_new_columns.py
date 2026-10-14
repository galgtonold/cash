"""A frame that shares columns with a frame the memo vouches for reads only its new columns.

p28-elo-ratings: ``for k ...: for hfa ...: h, _ = elo.run_elo(played, ...)``
returns ``played.assign(elo_home=..., elo_away=..., p_home=...)`` on every
pass. Each new ``h`` was read in full for its digest (the early cutoff and
the store take one), 0.4 s a pass, although seven of its ten columns are
``played``'s own memory, which the copy-on-write memo already vouched for.
Now a column whose memory a frame in the memo holds, while that frame is
unchanged, adds the bytes it added to that frame's digest
(`FrameDigests.hash_frame`); the digest is the one a full read gives.

Each way a frame changes is checked against a full read: a column the
memo cannot vouch for is read again.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.decorator import arg_hashing
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

ROWS = 300_000  # every column over one leaf (1 MiB): its part of a digest is a tree digest


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = np.array([f"T{i:03d}" for i in range(400)], dtype=object)
    return pd.DataFrame(
        {
            "season": rng.integers(2000, 2050, ROWS),
            "date": pd.date_range("2000-01-01", periods=ROWS, freq="min"),
            "home": pd.array(teams[rng.integers(0, 400, ROWS)], dtype="str"),
            "away": teams[rng.integers(0, 400, ROWS)],
            "goals": rng.random(ROWS),
        }
    )


def _derived(base: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    return base.assign(p_home=base["goals"] * k)


def _full(frame: pd.DataFrame) -> str:
    """The digest a full read gives: no memo in front."""
    return content_hashers.builtin_hash(frame)


def test_a_derived_frame_reads_only_its_new_column_and_digests_alike():
    base = _base()
    assert arg_hashing.cow_frame_digest(base) == _full(base)
    for k in (1.0, 2.0, 3.0):  # a loop's passes
        derived = _derived(base, k)
        with hashed_bytes() as reused:
            digest = arg_hashing.cow_frame_digest(derived)
        with hashed_bytes() as read:
            full = _full(derived)
        assert digest == full
        # Only p_home (2.4 MB) is read; the five shared columns (~10 MB) are not.
        assert reused.bytes < 0.3 * read.bytes, (reused.bytes, read.bytes)


def test_a_frame_unchanged_across_uses_is_read_once():
    base = _base()
    with hashed_bytes() as first:
        arg_hashing.cow_frame_digest(base)
    with hashed_bytes() as again:
        for _ in range(5):
            arg_hashing.cow_frame_digest(base)
    assert again.bytes < 0.01 * first.bytes


CHANGES = {
    "loc": lambda f: f.loc.__setitem__((0, "goals"), 99.0),
    "column assign": lambda f: f.__setitem__("season", f["season"] + 1),
    "inplace method": lambda f: f.sort_values("goals", inplace=True),
    "insert": lambda f: f.insert(0, "extra", np.arange(len(f))),
    "pop": lambda f: f.pop("away"),
    "string cell": lambda f: f.loc.__setitem__((0, "home"), "XXX"),
    "date cell": lambda f: f.loc.__setitem__((0, "date"), pd.Timestamp("1999-01-01")),
}


@pytest.mark.parametrize("change", list(CHANGES.values()), ids=list(CHANGES))
def test_a_changed_frame_is_read_again_and_its_derived_frames_digest_its_new_content(change):
    base = _base()
    before = arg_hashing.cow_frame_digest(base)
    stale = _derived(base)
    arg_hashing.cow_frame_digest(stale)
    change(base)
    after = arg_hashing.cow_frame_digest(base)
    assert after == _full(base)
    assert after != before
    derived = _derived(base)
    assert arg_hashing.cow_frame_digest(derived) == _full(derived)
    # The frame made before the change still holds the old values.
    assert arg_hashing.cow_frame_digest(stale) == _full(stale)


def _write_through_to_numpy(frame):
    view = frame["goals"].to_numpy()
    view.flags.writeable = True
    view[0] = 1234.5


def _write_through_values(frame):
    view = frame["season"].values
    view.flags.writeable = True
    view[0] = 1


def _write_through_array(frame):
    frame["goals"].array[1] = -1.0


HANDLES = {
    "to_numpy": _write_through_to_numpy,
    "values": _write_through_values,
    "array": _write_through_array,
    "taken, not written": lambda f: f["goals"].to_numpy(),
}


@pytest.mark.parametrize("on", ["base", "derived"])
@pytest.mark.parametrize("handle", list(HANDLES.values()), ids=list(HANDLES))
def test_a_write_through_a_raw_handle_is_never_served_from_a_kept_column(handle, on):
    base = _base()
    arg_hashing.cow_frame_digest(base)
    derived = _derived(base)
    arg_hashing.cow_frame_digest(derived)
    try:
        handle(base if on == "base" else derived)
    except ValueError:  # pragma: no cover - numpy refusing writeable=True is fine too
        pytest.skip("this numpy does not let the view be made writable")
    again = _derived(base, 5.0)
    assert arg_hashing.cow_frame_digest(again) == _full(again)
    assert arg_hashing.cow_frame_digest(base) == _full(base)
    assert arg_hashing.cow_frame_digest(derived) == _full(derived)


def test_a_frame_shared_by_two_names_and_held_in_a_tuple_is_seen_changing_through_either():
    base = _base()
    alias = base
    held = (base, 1)
    arg_hashing.cow_frame_digest(base)
    arg_hashing.cow_frame_digest(_derived(held[0]))
    alias.loc[0, "goals"] = -5.0
    for frame in (base, held[0], _derived(held[0], 7.0)):
        assert arg_hashing.cow_frame_digest(frame) == _full(frame)
    assert held[0].loc[0, "goals"] == -5.0


def test_a_frame_built_over_a_callers_array_is_read_in_full(monkeypatch):
    """Memory the frame borrows is never vouched for: nothing is kept."""
    raw = np.random.default_rng(1).random((ROWS, 2))
    base = pd.DataFrame(raw, columns=["a", "b"], copy=False)
    arg_hashing.cow_frame_digest(base)
    raw[0, 0] = 42.0  # straight past pandas
    derived = base.assign(c=1.0)
    assert arg_hashing.cow_frame_digest(derived) == _full(derived)


def test_pandas_without_copy_on_write_reads_every_frame_in_full(monkeypatch):
    """No memo without copy-on-write: each digest is a full read."""
    monkeypatch.setattr(arg_hashing, "_COW_PANDAS", False)
    base = _base()
    assert arg_hashing.cow_frame_digest(base) is None
    with hashed_bytes() as read:
        from cash.value_hash import compute_hash

        first = compute_hash(_derived(base))
    with hashed_bytes() as again:
        second = compute_hash(_derived(base))
    assert first == second
    assert again.bytes >= 0.9 * read.bytes
    assert len(arg_hashing.FRAME_DIGESTS) == 0
