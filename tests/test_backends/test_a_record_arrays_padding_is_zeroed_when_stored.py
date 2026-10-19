"""A stored record array's padding is zeroed, so a restored value's bytes do
not change from process to process.

numpy copies a record dtype field by field into fresh memory, so the gaps
between fields held whatever that memory held. A restored scikit-learn
forest had garbage in seven bytes of every tree node: the same values, but
a pickle of the restored model differed in every process while the plain
model's did not.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]

#: A ``float64`` then a ``uint8``, padded to 16 bytes: seven bytes of gap.
RECORD = np.dtype({"names": ["value", "flag"], "formats": ["<f8", "u1"], "offsets": [0, 8], "itemsize": 16})


def _stored(value):
    b = InMemoryBackend()
    b.set("k", value, {"copy_required": True})
    return b, b.get("k")[1]


def _gaps(records):
    return np.frombuffer(records.tobytes(), np.uint8).reshape(-1, 16)[:, 9:]


def test_the_gaps_of_a_record_array_are_zero():
    records = np.frombuffer(bytes(range(1, 65)), np.uint8).copy().view(RECORD)  # gaps hold 10..16, ...
    b, first = _stored({"nodes": records})
    second = b.get("k")[1]
    for got in (first, second):
        assert not _gaps(got["nodes"]).any()
        assert got["nodes"].tolist() == records.tolist()


def test_a_nested_record_and_a_fortran_array_keep_their_values():
    nested = np.dtype([("inner", RECORD), ("n", "<i8")], align=True)
    value = np.zeros(3, nested)
    value["inner"]["value"] = [1.5, 2.5, 3.5]
    value["n"] = [7, 8, 9]
    plain = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    _, got = _stored([value, plain])
    assert got[0].tolist() == value.tolist()
    assert got[0].tobytes() == value.tobytes()
    assert got[1].flags.f_contiguous and got[1].tolist() == plain.tolist()


def test_a_restored_forest_pickles_the_same_each_time():
    ensemble = pytest.importorskip("sklearn.ensemble")
    datasets = pytest.importorskip("sklearn.datasets")
    X, y = datasets.make_classification(n_samples=200, random_state=0)
    forest = ensemble.RandomForestClassifier(n_estimators=4, random_state=0).fit(X, y)
    restored = [_stored(forest)[1] for _ in range(3)]
    for model in restored:
        for tree in model.estimators_:
            nodes = tree.tree_.__getstate__()["nodes"]
            gaps = np.frombuffer(nodes.tobytes(), np.uint8).reshape(-1, nodes.dtype.itemsize)[:, 57:]
            assert not gaps.any()
    assert len({pickle.dumps(m) for m in restored}) == 1
    assert (restored[0].predict(X) == forest.predict(X)).all()
