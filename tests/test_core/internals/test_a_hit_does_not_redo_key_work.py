"""A cache hit does the key work that can change, and nothing it did before.

Each test pins one piece of repeated work a hit used to redo, and next to it
that what CAN change still reaches the key:

* ``import annotationlib`` retried on every hit below Python 3.14 (~40us per
  class and per method of every user object in a key);
* a module constant re-hashed on every hit (~45-80us each), while a list or
  dict global edited in place must still be seen;
* a class's methods folded twice, once by the helper walk and once by the
  class, on a function that builds the class;
* an object holding a few small arrays (a fitted random forest) opened up and
  hashed array by array, 2.5x slower than its pickle;
* ``dataclasses.fields()`` read on every hit to decode the entry's metadata;
* primitive arguments sent through the general canonical walk;
* three failed ``inspect.getsource`` calls per dataclass per hit.
"""

from __future__ import annotations

import builtins
import dataclasses

import numpy as np
import pandas as pd
import pytest

from cash import _annotation_refs, canonical_form, content_hashers
from cash.decorator import arg_hashing, cache_metadata, function_identity
from cash.decorator.arg_hashing import ArgHasher
from cash.decorator.module_attrs import ModuleAttrFold
from tests.test_core.internals import _hit_work_fixture as fx

pytestmark = pytest.mark.core


class Annotated:
    rate: float

    def scale(self, x: int) -> float:
        return x * self.rate


def test_annotations_do_not_retry_the_annotationlib_import(monkeypatch):
    real = builtins.__import__
    tried = []

    def spy(name, *args, **kwargs):
        if name == "annotationlib":
            tried.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    for _ in range(3):
        _annotation_refs.annotation_referents(Annotated)
    assert tried == []


def _spy_hashed(monkeypatch):
    hashed = []
    real = ArgHasher.hash_payload

    def spy(self, args, kwargs):
        hashed.append(args)
        return real(self, args, kwargs)

    monkeypatch.setattr(ArgHasher, "hash_payload", spy)
    return hashed


def test_a_hit_does_not_rehash_an_unchanged_constant(disk_cash, monkeypatch):
    f = disk_cash.cache(fx.reads_constants)
    assert f(2) == 5.0
    hashed = _spy_hashed(monkeypatch)
    assert f(2) == 5.0
    assert (2.5,) not in hashed and ("exp1",) not in hashed and (fx.SHAPE,) not in hashed
    monkeypatch.setattr(fx, "SCALE", 3.0)
    assert f(2) == 6.0
    monkeypatch.setattr(fx, "SHAPE", ())
    assert f(2) == 0.0


def test_a_container_global_is_keyed_by_its_content_on_every_hit(disk_cash, monkeypatch):
    hasher = disk_cash._args
    for value in ({"lr": 0.1, "layers": [64, 32]}, [1, [2, 2], "x"], [[0] * 3] * 3, (1, ("a", 2.5))):
        assert hasher.plain_value_digest(value) == hasher.hash_payload((value,), {})
    monkeypatch.setattr(fx, "COLS", ["a", "b"])
    monkeypatch.setattr(fx, "CONFIG", {"lr": 0.1, "layers": [64, 32]})
    f = disk_cash.cache(fx.reads_containers)
    assert f(10) == pytest.approx(2.0)
    assert f(10) == pytest.approx(2.0)
    fx.COLS.append("c")
    assert f(10) == pytest.approx(3.0)
    fx.CONFIG["lr"] = 0.2
    assert f(10) == pytest.approx(6.0)
    assert f.cache_info()["hits"] == 1


def test_a_hit_folds_each_method_once(disk_cash, monkeypatch):
    f = disk_cash.cache(fx.builds_classes)
    assert f(10) == pytest.approx(0.3)
    folded = []
    real = ModuleAttrFold.module_attr_parts

    def spy(self, func, *args, **kwargs):
        folded.append(func.__qualname__)
        return real(self, func, *args, **kwargs)

    monkeypatch.setattr(ModuleAttrFold, "module_attr_parts", spy)
    assert f(10) == pytest.approx(0.3)
    assert "Scaler.apply" in folded
    assert sorted(folded) == sorted(set(folded))
    # What the method reads still reaches the key.
    monkeypatch.setattr(fx, "RATE", 0.2)
    assert f(10) == pytest.approx(0.6)


class Forest:
    def __init__(self, n):
        self.name = "rf"
        self.classes_ = np.arange(3)
        self.trees = [Leafy(i) for i in range(n)]


class Leafy:
    def __init__(self, i):
        self.classes_ = np.arange(3) + i
        self.depth = i


def _spy_content_reads(monkeypatch):
    reads = []
    real = arg_hashing.builtin_hash

    def spy(value):
        if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
            reads.append(type(value).__name__)
        return real(value)

    monkeypatch.setattr(arg_hashing, "builtin_hash", spy)
    return reads


def test_an_object_holding_small_arrays_is_pickled_whole(disk_cash, monkeypatch):
    @disk_cash.cache
    def total(forest):
        return int(sum(t.classes_.sum() for t in forest.trees))

    forest = Forest(5)
    assert total(forest) == 45
    reads = _spy_content_reads(monkeypatch)
    assert total(forest) == 45
    assert reads == []
    forest.trees[0].classes_[0] = 100
    assert total(forest) == 145


def test_an_object_holding_a_big_array_is_keyed_part_by_part(disk_cash, monkeypatch):
    class Holder:
        def __init__(self):
            self.big = np.zeros(canonical_form.OPEN_UP_BYTES // 8 + 1)

    @disk_cash.cache
    def first(h):
        return float(h.big[0])

    h = Holder()
    assert first(h) == 0.0
    reads = _spy_content_reads(monkeypatch)
    assert first(h) == 0.0
    assert reads == ["ndarray"]
    h.big[0] = 7.0
    assert first(h) == 7.0


def test_metadata_is_decoded_without_reading_the_fields_again(monkeypatch):
    raw = cache_metadata.CacheMetadata(key="k", func_name="f").to_dict()
    cache_metadata.CacheMetadata.from_dict({**raw, "unknown": 1})

    def refuse(_cls):
        raise AssertionError("dataclasses.fields() read again")

    monkeypatch.setattr(cache_metadata, "fields", refuse)
    decoded = cache_metadata.CacheMetadata.from_dict({**raw, "unknown": 1})
    assert decoded.key == "k" and decoded.func_name == "f"


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((3, 2.5, "hello"), {}),
        ((None, True, b"x", 1j, -0.0, 2**70), {"a": "s", "b": 0}),
        (("s", "s"), {"x": "s"}),
    ],
)
def test_primitive_arguments_key_as_the_walk_keys_them(disk_cash, monkeypatch, args, kwargs):
    fast = disk_cash._args.hash_payload(args, kwargs)
    monkeypatch.setattr(arg_hashing, "canonical_call_bytes", lambda *_: None)
    assert disk_cash._args.hash_payload(args, kwargs) == fast


def test_primitive_arguments_skip_the_canonical_walk(disk_cash, monkeypatch):
    walked = []
    real = arg_hashing.canonical_bytes

    def spy(*args, **kwargs):
        walked.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(arg_hashing, "canonical_bytes", spy)
    disk_cash._args.hash_payload((3, 2.5, "hello"), {"k": None})
    assert walked == []
    disk_cash._args.hash_payload(([3],), {})
    assert len(walked) == 1


def test_a_generated_method_s_identity_is_read_once(monkeypatch):
    @dataclasses.dataclass
    class Point:
        x: int = 0

    init = Point.__init__
    reads = []
    real = function_identity.source_digest

    def spy(fn):
        reads.append(fn)
        return real(fn)

    monkeypatch.setattr(function_identity, "source_digest", spy)
    first = function_identity.hash_callable_source(init)
    assert function_identity.hash_callable_source(init) == first
    assert len(reads) == 1


def test_what_is_worth_keying_on_its_own_is_decided_by_size():
    sparse = pytest.importorskip("scipy.sparse")

    def worth(value):
        return canonical_form._worth_opening(value, content_hashers.BUILTIN_CONTENT)

    assert not worth(np.zeros(8))
    assert worth(np.zeros(canonical_form.OPEN_UP_BYTES // 8))
    assert not worth(sparse.eye(10, format="csr"))
    assert worth(sparse.random(1000, 1000, density=0.2, format="csr", random_state=0))
    assert not worth(pd.DataFrame({"x": range(10)}))
    assert worth(pd.DataFrame({"x": np.zeros(canonical_form.OPEN_UP_BYTES // 8)}))
