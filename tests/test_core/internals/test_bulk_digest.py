"""`cash.bulk_digest`: the digest of array buffers, columns and file contents.

What it must hold: the digest depends only on the bytes -- not on how they
were handed over, nor on how many threads hashed them -- and a big buffer is
hashed on several threads at once. The second is how it is fast, so it is
pinned here, where a refactor expects failures.
"""

from __future__ import annotations

import hashlib
import os
import threading

import pytest

from cash import bulk_digest
from cash.bulk_digest import LEAF_BYTES, BulkHasher, fold_buffer


def _data(n: int) -> bytes:
    return os.urandom(n)


@pytest.mark.parametrize("n", [0, 1, LEAF_BYTES - 1, LEAF_BYTES, LEAF_BYTES + 1, 5 * LEAF_BYTES + 77])
def test_the_digest_does_not_depend_on_the_pieces(n):
    data = _data(n)
    whole = bulk_digest.bulk_digest(data)
    for step in (1000, LEAF_BYTES - 3, LEAF_BYTES, 2 * LEAF_BYTES + 5):
        h = BulkHasher()
        for i in range(0, n, step):
            h.update(data[i : i + step])
        assert h.digest() == whole, f"pieces of {step} bytes changed the digest"


def test_the_digest_does_not_depend_on_the_threads(monkeypatch):
    data = _data(9 * LEAF_BYTES + 5)
    monkeypatch.setattr(bulk_digest, "PARALLEL_MIN_BYTES", LEAF_BYTES)
    digests = set()
    for threads in (1, 2, 3, 8):
        monkeypatch.setattr(bulk_digest, "_THREADS", threads)
        digests.add(bulk_digest.bulk_digest(data))
    assert len(digests) == 1


def test_a_big_buffer_is_hashed_on_several_threads(monkeypatch):
    monkeypatch.setattr(bulk_digest, "_THREADS", 4)
    started: list[str] = []
    real_start = threading.Thread.start

    def start(self):
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    bulk_digest.bulk_digest(bytes(32 * LEAF_BYTES))
    assert started.count("cash-digest") == 3, started
    started.clear()
    bulk_digest.bulk_digest(bytes(LEAF_BYTES))
    assert not started, "a small buffer started threads"


def test_every_byte_counts():
    data = bytearray(_data(3 * LEAF_BYTES))
    first = bulk_digest.bulk_digest(data)
    for pos in (0, LEAF_BYTES - 1, LEAF_BYTES, len(data) - 1):
        changed = bytearray(data)
        changed[pos] ^= 1
        assert bulk_digest.bulk_digest(changed) != first, pos
    assert bulk_digest.bulk_digest(data + b"\0") != first
    assert bulk_digest.bulk_digest(data[:-1]) != first


def test_a_folded_buffer_cannot_read_as_another():
    """Small buffers go in whole and big ones as their tree digest, each
    behind a tag and the length: no two buffers fold alike."""

    def folded(*buffers):
        h = hashlib.sha256()
        for b in buffers:
            fold_buffer(h, b)
        return h.digest()

    big = _data(LEAF_BYTES + 1)
    assert folded(b"ab", b"c") != folded(b"a", b"bc")
    assert folded(big) != folded(big[:-1])
    tree = b"t" + len(big).to_bytes(8, "little") + bulk_digest.bulk_digest(big)
    assert folded(big) != folded(tree)


def test_a_strided_buffer_is_read_in_order():
    np = pytest.importorskip("numpy")
    a = np.arange(24, dtype=np.int64).reshape(4, 6)
    h1, h2 = hashlib.sha256(), hashlib.sha256()
    fold_buffer(h1, a[:, ::2])
    fold_buffer(h2, np.ascontiguousarray(a[:, ::2]))
    assert h1.digest() == h2.digest()
