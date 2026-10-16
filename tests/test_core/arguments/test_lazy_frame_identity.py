"""A polars LazyFrame must be identified by its data, not by its plan.

The polars hasher hashed ``LazyFrame.explain()`` -- the human-readable
query plan. Two frames over different in-memory data print identically::

    pl.DataFrame({"x": [1, 2, 3]}).lazy().explain()
    pl.DataFrame({"x": [10, 20, 30]}).lazy().explain()
    # both: 'DF ["x"]; PROJECT */1 COLUMNS'

so the second call was served the first's cached result. A wrong answer, not
a stale-after-edit one, and reachable in three lines.

Found while asking a different question -- whether SHA-256 collisions are
worth worrying about. They are not (2^-256). The risk was never the hash; it
was the two places where identity is not derived from content at all. dask
was the other suspect and turned out to be fine: ``__dask_keys__()`` carries
a data-derived token.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import cash
from cash.tracking import native_reads

pl = pytest.importorskip("polars")


@pytest.fixture
def cached(tmp_path):
    """A cached function plus the list of times its body actually ran."""
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    calls: list[int] = []

    @c.cache(assume_safe=True)
    def total(frame):
        calls.append(1)
        collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        return int(collected["x"].sum())

    return total, calls


def test_two_lazy_frames_over_different_data_do_not_collide(cached):
    """The reproduction. Both plans render identically; the data does not."""
    total, calls = cached
    assert total(pl.DataFrame({"x": [1, 2, 3]}).lazy()) == 6
    assert total(pl.DataFrame({"x": [10, 20, 30]}).lazy()) == 60, (
        "the second LazyFrame was served the first's cached result"
    )
    assert len(calls) == 2


def test_the_plans_really_are_identical(cached):
    """Pins WHY the bug existed, so the fix is not mistaken for a no-op."""
    a = pl.DataFrame({"x": [1, 2, 3]}).lazy()
    b = pl.DataFrame({"x": [10, 20, 30]}).lazy()
    assert a.explain() == b.explain(), (
        "the plans now differ, so this test no longer covers what it was "
        "written for -- re-check what identifies a LazyFrame"
    )
    assert a.serialize() != b.serialize(), "serialize() must see the data"


def test_the_same_lazy_frame_still_hits(cached):
    """The control: a fix that made everything unique would pass the arm above."""
    total, calls = cached
    assert total(pl.DataFrame({"x": [1, 2, 3]}).lazy()) == 6
    calls.clear()
    assert total(pl.DataFrame({"x": [1, 2, 3]}).lazy()) == 6
    assert calls == [], "an identical LazyFrame recomputed"


def test_identity_is_stable_across_processes(tmp_path):
    """A digest carrying a pointer would make every persisted entry miss.

    Run in a subprocess, because the failure mode this guards against only
    shows up across process boundaries.
    """
    import subprocess
    import sys

    script = (
        'import polars as pl, hashlib;print(hashlib.sha256(pl.DataFrame({"x":[1,2,3]}).lazy().serialize()).hexdigest())'
    )
    digests = {
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8").stdout.strip()
        for _ in range(2)
    }
    assert len(digests) == 1, f"LazyFrame identity is not process-stable: {digests}"


def test_eager_frames_were_never_affected(cached):
    """The control that scopes the bug: eager frames hash by content already."""
    total, calls = cached
    assert total(pl.DataFrame({"x": [1, 2, 3]})) == 6
    assert total(pl.DataFrame({"x": [10, 20, 30]})) == 60
    assert len(calls) == 2


@pytest.fixture(params=["hook on", "hook off"])
def hook(request):
    """Both ways: on macOS there is no native file hook to see polars open
    the file, so the scanned files must be recorded from the plan."""
    was = native_reads.active()
    if request.param == "hook off":
        native_reads.disable()
    elif not native_reads.enable():
        pytest.skip("the native file hook is not supported here")
    yield
    (native_reads.enable if was else native_reads.disable)()


def _settle(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def test_a_scan_backed_plan_notices_its_file_changing(tmp_path, hook):
    """A scan-backed plan serializes the PATH, not the file's contents, so the
    key alone does not move; the files it scans are the call's reads."""
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    csv = pathlib.Path(tmp_path) / "data.csv"
    csv.write_text("x\n1\n2\n3\n", encoding="utf-8")
    _settle(csv)
    runs = []

    @c.cache(assume_safe=True)
    def total(frame):
        runs.append(1)
        return int(frame.collect()["x"].sum())

    assert total(pl.scan_csv(csv)) == 6
    assert total(pl.scan_csv(csv)) == 6
    assert len(runs) == 1, "an unchanged file recomputed"
    csv.write_text("x\n10\n20\n30\n", encoding="utf-8")
    assert total(pl.scan_csv(csv)) == 60


def test_a_new_file_matching_a_scanned_glob_counts(tmp_path, hook):
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    folder = tmp_path / "parts"
    folder.mkdir()
    pl.DataFrame({"x": [1, 2]}).write_parquet(folder / "a.parquet")

    @c.cache(assume_safe=True)
    def total(frame):
        return int(frame.collect()["x"].sum())

    pattern = str(folder / "*.parquet")
    assert total(pl.scan_parquet(pattern)) == 3
    pl.DataFrame({"x": [10]}).write_parquet(folder / "b.parquet")
    assert total(pl.scan_parquet(pattern)) == 13


def test_the_sources_of_a_plan():
    from cash.content_hashers import lazy_frame_sources

    lf = pl.scan_csv("a.csv").join(pl.scan_parquet("data/*.parquet"), on="x").filter(pl.col("x") > 1)
    assert lazy_frame_sources(lf) == ["a.csv", "data/*.parquet"]
    assert lazy_frame_sources(pl.DataFrame({"x": [1]}).lazy()) == []


def test_no_plan_digest_when_serialize_refuses(monkeypatch):
    """A plan ``serialize()`` refuses gets no built-in hash at all. It used to
    fall back to the ``explain()`` text, which is the collision above."""
    from cash.content_hashers import builtin_hash

    def refuse(self, *args, **kwargs):
        raise RuntimeError("serialize() unavailable")

    monkeypatch.setattr(pl.LazyFrame, "serialize", refuse)
    assert builtin_hash(pl.DataFrame({"x": [1, 2, 3]}).lazy()) is None


def test_eager_frames_carry_their_schema():
    """Equal values under two dtypes, or two names, are different frames."""
    from cash.content_hashers import builtin_hash

    base = pl.DataFrame({"x": [1, 2, 3]})
    assert builtin_hash(base) != builtin_hash(base.cast({"x": pl.Int32}))
    assert builtin_hash(base) != builtin_hash(base.rename({"x": "y"}))
    assert builtin_hash(base["x"]) != builtin_hash(base["x"].cast(pl.Int32))
    assert builtin_hash(base) == builtin_hash(pl.DataFrame({"x": [1, 2, 3]}))
