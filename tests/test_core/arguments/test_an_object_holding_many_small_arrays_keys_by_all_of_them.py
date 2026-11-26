"""An estimator-like object holding thousands of small arrays (a meta-analysis
estimator's data bags: a list of dicts of short arrays, beside a frame) is
keyed by every one of them, and a cached call taking it behaves as plain
Python: a value changed deep inside misses, a body that writes into one of
them runs every time. Reading such an object is cheap per array: an array
needs no search for sets and no layout sort when it is C-contiguous, and the
checks after the body read each array once between them (`quiet_digests`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cash import Cash, canonical_form, content_hashers
from cash.canonical_form import contains_set
from cash.content_hashers import array_layout


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _layout_by_strides(value):
    """`array_layout` read from the strides alone, as it was before its
    contiguous fast path."""
    axes = [(axis, stride) for axis, (n, stride) in enumerate(zip(value.shape, value.strides)) if n > 1]
    if len(axes) <= 1:
        return "C"
    outer_first = sorted(axes, key=lambda a: (-abs(a[1]) if a[1] else float("-inf"), a[0]))
    perm = tuple(axis for axis, _ in outer_first)
    natural = tuple(axis for axis, _ in axes)
    if perm == natural:
        return "C"
    if perm == natural[::-1]:
        return "F" if value.flags.f_contiguous else "Fs"
    return "K" + ",".join(map(str, perm))


def _arrays():
    base = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    yield np.float64(1.5) * np.ones(())
    yield np.arange(5)
    yield np.arange(10)[::2]
    yield base
    yield base.T
    yield base.transpose(1, 0, 2)
    yield base[:, ::2]
    yield base[:, :1]
    yield np.asfortranarray(base)
    yield base.T[::2]
    yield np.broadcast_to(np.arange(3), (4, 3))
    yield np.zeros((3, 0, 5)).transpose(2, 1, 0)
    yield np.zeros((0, 4))
    yield np.zeros((4, 1, 3))
    yield np.zeros((3, 4), dtype="V0")
    yield np.ones((3, 4), dtype=[("a", "i4"), ("b", "f8")])


def test_the_contiguous_fast_path_reads_the_layout_the_strides_give():
    for value in _arrays():
        assert array_layout(value) == _layout_by_strides(value), (value.shape, value.strides)


def test_arrays_are_not_searched_for_sets_and_sets_beside_them_still_are():
    class Holder:
        pass

    held = Holder()
    held.bags = [{"values": np.arange(3.0), "mask": np.arange(2)} for _ in range(50)]
    assert contains_set(held) is False
    held.bags[7]["kinds"] = {"a", "b"}
    assert contains_set(held) is True


class Estimator:
    """Holds what a fitted meta-analysis estimator holds: many small
    arrays in a list of dicts, a frame and a scalar setting."""

    def __init__(self, n_bags=400, scale=1.0):
        rng = np.random.default_rng(0)
        self.scale = scale
        self.inputs_ = {
            "data_bags": [
                {"values": rng.random((5, 3)), "voxel_mask": np.arange(3), "study_mask": np.arange(5)}
                for _ in range(n_bags)
            ]
        }
        self.table = pd.DataFrame({"x": np.arange(10.0)})


def _total(est):
    bags = est.inputs_["data_bags"]
    return float(sum(b["values"].sum() for b in bags) * est.scale + est.table["x"].sum())


def test_a_change_deep_inside_misses_like_plain_python(c):
    runs = []

    @c.cache
    def total(est):
        runs.append(1)
        return _total(est)

    est = Estimator()
    assert total(est) == _total(est)
    assert total(est) == _total(est)
    assert len(runs) == 1

    est.inputs_["data_bags"][321]["values"][4, 2] += 1.0  # one value of one small array
    assert total(est) == _total(est)
    est.inputs_["data_bags"][5]["voxel_mask"][0] = 7
    assert total(est) == _total(est)
    est.table.loc[3, "x"] = -1.0
    assert total(est) == _total(est)
    est.scale = 2.0
    assert total(est) == _total(est)
    assert len(runs) == 5
    assert total(est) == _total(est)
    assert len(runs) == 5


def test_a_method_that_writes_into_one_array_runs_every_time(c):
    class Fitting(Estimator):
        @c.cache
        def fit(self):
            self.result_ = _total(self)  # replayable on its own
            self.inputs_["data_bags"][10]["values"][0, 0] += 1.0  # not without the arrays
            return self.result_

    def plain_fit(est):
        est.result_ = _total(est)
        est.inputs_["data_bags"][10]["values"][0, 0] += 1.0
        return est.result_

    plain, fitted = Estimator(), Fitting()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            assert fitted.fit() == plain_fit(plain)
    assert fitted.inputs_["data_bags"][10]["values"][0, 0] == plain.inputs_["data_bags"][10]["values"][0, 0]
    assert fitted.result_ == plain.result_


def test_the_checks_after_a_body_read_each_array_once(c, monkeypatch):
    """The body binds an attribute and changes another in place: the check
    with the attribute put back and the one of the arguments as the body
    left them both hash every array, and between them nothing runs that
    could write into one."""
    read = []
    hash_numpy = content_hashers.hash_numpy

    def counting(value):
        read.append(id(value))
        return hash_numpy(value)

    class Fitting(Estimator):
        @c.cache
        def fit(self):
            self.result_ = 1
            self.log.append(1)
            return 1

    est = Fitting(n_bags=100)
    est.log = []
    monkeypatch.setattr(content_hashers, "hash_numpy", counting)
    # Counted in the Python walk: the C one reads these arrays without
    # `hash_numpy` (its own window test is in
    # test_the_c_key_walk_forms_what_the_python_walk_forms.py).
    monkeypatch.setattr(canonical_form, "_PLAIN_WALK", None)
    arrays = 3 * 100
    reads = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            read.clear()
            est.fit()
            reads.append(len(read))
    assert est.log == [1, 1, 1]
    # The first call was checked after its body: every array was read for
    # the key before it and once after it, not once per check.
    assert reads[0] >= 2 * arrays, reads
    assert max(reads) <= 2 * arrays + 2, reads
