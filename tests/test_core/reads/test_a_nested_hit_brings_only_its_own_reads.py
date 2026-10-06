"""A cached call served from cache inside another brings the outer call only its own entry's reads.

``fit(region)`` reads ``cases_<region>.csv``. After ``fit("r1")`` and
``fit("r2")`` ran, ``scen("r2")`` called ``fit("r2")``, which hit -- and the
``scen`` entry recorded ``cases_r1.csv`` too, because everything ``fit``'s code
had ever read in the process was credited to it as if a memo had handed it
over. An edit to ``cases_r1.csv`` then recomputed ``scen("r2")``. The entry
that was served carries exactly the files the outer call depends on.
"""

from __future__ import annotations

import functools
import os
import threading

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def _write(path, values):
    path.write_text("x\n" + "\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return sum(int(line) for line in list(fh)[1:] if line.strip())


def _note(root, line):
    # `os.write`, not `open`: a counter the body writes itself through `open`
    # would be a tracked file, and a dict it appends to is part of its key.
    fd = os.open(root / "runs.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, f"{line}\n".encode())
    finally:
        os.close(fd)


def _runs(root):
    try:
        fd = os.open(root / "runs.log", os.O_RDONLY)
    except FileNotFoundError:
        return {"fit": [], "scen": []}
    try:
        text = os.read(fd, 1 << 20).decode()
    finally:
        os.close(fd)
    out: dict[str, list[str]] = {"fit": [], "scen": []}
    for line in text.split():
        name, region = line.split(":")
        out[name].append(region)
    return out


def _pipeline(c, root, through):
    def load(region):
        return _read(root / f"cases_{region}.csv")

    memo_load = functools.lru_cache(maxsize=8)(load)

    @c.cache(assume_safe=True)
    def fit(region):
        _note(root, f"fit:{region}")
        if through == "helper":
            return load(region)
        if through == "memo":
            return memo_load(region)
        return _read(root / f"cases_{region}.csv")

    @c.cache(assume_safe=True)
    def scen(region, k):
        _note(root, f"scen:{region}")
        return fit(region) * k + _read(root / f"contacts_{region}.csv")

    return fit, scen, memo_load


@pytest.fixture
def data(tmp_path):
    for region in ("r1", "r2"):
        _write(tmp_path / f"cases_{region}.csv", [1, 2])
        _write(tmp_path / f"contacts_{region}.csv", [5])
    return tmp_path


@pytest.mark.parametrize("through", ["body", "helper", "memo"])
def test_another_entry_of_the_inner_function_is_not_a_dependency(data, through):
    fit, scen, _ = _pipeline(Cash(cache_dir=str(data / "cache")), data, through)
    fit("r1")
    fit("r2")
    assert scen("r2", 1) == 8
    assert _runs(data) == {"fit": ["r1", "r2"], "scen": ["r2"]}, "fit('r2') did not hit inside scen"

    _write(data / "cases_r1.csv", [100])
    assert scen("r2", 1) == 8
    assert _runs(data)["scen"] == ["r2"], "an edit to a file only fit('r1') read recomputed scen('r2')"


@pytest.mark.parametrize("through", ["body", "helper", "memo"])
def test_the_served_entry_s_own_reads_are_still_dependencies(data, through):
    """Control: the file of the entry that WAS served still invalidates the outer call."""
    fit, scen, memo_load = _pipeline(Cash(cache_dir=str(data / "cache")), data, through)
    fit("r1")
    fit("r2")
    assert scen("r2", 1) == 8

    _write(data / "cases_r2.csv", [10])
    memo_load.cache_clear()  # what a new process starts with
    assert scen("r2", 1) == 15
    assert _runs(data)["scen"] == ["r2", "r2"], "an edit to the served entry's file did not recompute scen"


# The two controls below build their functions separately, a reader of their
# own included: cash credits a memo's reads to its CODE, which every closure a
# builder makes -- and every caller of `_read` -- shares, so one test's files
# would otherwise become the other's remembered reads.


def _direct(c, root):
    """``fit`` parses through a memo; ``scen`` reaches the same memo for another region."""

    def read(path):  # code of its own (see above)
        with open(path, encoding="utf-8") as fh:
            return sum(int(line) for line in list(fh)[1:] if line.strip())

    load = functools.lru_cache(maxsize=8)(lambda region: read(root / f"cases_{region}.csv"))

    @c.cache(assume_safe=True)
    def fit(region):
        _note(root, f"fit:{region}")
        return load(region)

    @c.cache(assume_safe=True)
    def scen(region):
        _note(root, f"scen:{region}")
        return fit(region) + load("r1")

    return fit, scen, load


def _unkeyed(c, root):
    """``scen`` also calls ``fit`` with an argument that has no key, so that
    call runs ``fit``'s body here and gets ``r1`` from the memo."""

    def read(path):  # code of its own (see above)
        with open(path, encoding="utf-8") as fh:
            return sum(int(line) for line in list(fh)[1:] if line.strip())

    load = functools.lru_cache(maxsize=8)(lambda region: read(root / f"cases_{region}.csv"))

    @c.cache(assume_safe=True)
    def fit(region, extra=None):
        _note(root, f"fit:{region}")
        return load(region)

    @c.cache(assume_safe=True)
    def scen(region):
        _note(root, f"scen:{region}")
        # A lock cannot be hashed: this `fit` call has no key and runs its body.
        return fit(region) + fit("r1", extra=threading.Lock())

    return fit, scen, load


def _memo_reached_another_way(data, build, fit_runs):
    fit, scen, load = build(Cash(cache_dir=str(data / "cache")), data)
    fit("r1")
    fit("r2")
    assert scen("r2") == 6
    assert scen("r2") == 6
    assert _runs(data)["scen"] == ["r2"], "the outer call was not cached"
    assert _runs(data)["fit"] == fit_runs, "fit('r2') was not served from cache inside scen"

    _write(data / "cases_r1.csv", [100])
    load.cache_clear()  # what a new process starts with
    assert scen("r2") == 103, "a file the memo handed the outer call was not its dependency"
    assert _runs(data)["scen"] == ["r2", "r2"]


def test_a_memo_the_outer_call_reads_itself_is_still_credited(data):
    """Control: ``fit('r2')`` is served from cache and ``scen`` takes
    ``cases_r1.csv`` from the memo ``fit('r1')`` filled. Leaving ``fit`` out
    of the crediting must not lose a helper ``scen`` reaches itself."""
    _memo_reached_another_way(data, _direct, ["r1", "r2"])


def test_a_memo_reached_by_an_unkeyed_call_of_the_served_function_is_still_credited(data):
    """Control: one ``fit`` call is served from cache, another has no key and
    runs ``fit``'s body inside ``scen``, getting ``cases_r1.csv`` from the
    memo. ``fit`` is followed again, so that file is not lost."""
    _memo_reached_another_way(data, _unkeyed, ["r1", "r2", "r1"])
