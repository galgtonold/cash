"""Big buffers are hashed with BLAKE3 when ``blake3`` is installed
(``cash-lib[fast]``), with SHA-256 when it is not, and the two never meet.

What must hold:

- each algorithm gives one digest for the same bytes, the one its scheme
  names: SHA-256 leaves without ``blake3`` (the digests every earlier build
  took, so a plain install keeps its cache), BLAKE3 leaves with it;
- the algorithm is in the digest's tag, so the same bytes digest apart under
  the two, and a cache written by one install only misses under the other --
  never serves an entry for other data;
- a missing ``blake3`` (here hidden from a fresh interpreter) is the SHA-256
  path, not an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap

import pytest

from cash import bulk_digest
from cash.bulk_digest import LEAF_BYTES, BulkHasher, fold_buffer
from tests._scripts import CASH_SRC, run_python

pytestmark = pytest.mark.core


def _hide_blake3(tmp_path) -> dict[str, str]:
    """The env of a child in which ``import blake3`` fails, as on an
    install without the ``fast`` extra."""
    shadow = tmp_path / "no-blake3"
    (shadow / "blake3").mkdir(parents=True, exist_ok=True)
    (shadow / "blake3" / "__init__.py").write_text(
        'raise ModuleNotFoundError("No module named blake3 (hidden by the test)")\n'
    )
    path = [str(shadow), CASH_SRC, os.environ.get("PYTHONPATH", "")]
    return {"PYTHONPATH": os.pathsep.join(p for p in path if p)}


def _reference(data: bytes, leaf, tag: bytes) -> bytes:
    """The tree digest written out from its definition."""
    leaves = [leaf(data[i : i + LEAF_BYTES]).digest() for i in range(0, len(data), LEAF_BYTES)]
    return hashlib.sha256(tag + len(data).to_bytes(8, "little") + b"".join(leaves)).digest()


_SHA256_TAG = b"cash-bulk-sha256-tree-1\x00"

_PROBE = textwrap.dedent("""
    import json, sys
    from cash import bulk_digest
    data = bytes(range(256)) * (3 * 4096 + 7)
    print(json.dumps({
        "algorithm": bulk_digest.ALGORITHM,
        "tag": bulk_digest.TAG.decode("ascii"),
        "scheme": bulk_digest.SCHEME,
        "direct_max": bulk_digest.DIRECT_MAX_BYTES,
        "blake3_loaded": "blake3" in sys.modules and sys.modules["blake3"] is not None,
        "digest": bulk_digest.bulk_digest(data).hex(),
    }))
""")
_PROBE_DATA = bytes(range(256)) * (3 * 4096 + 7)


def _probe(tmp_path, env=None) -> dict:
    done = run_python("-c", _PROBE, cwd=tmp_path, env=env)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_without_blake3_the_leaves_are_sha256_as_before(tmp_path):
    """The fallback: no error, SHA-256 leaves under the SHA-256 tag, and
    digests bit for bit those of a build that had only SHA-256."""
    got = _probe(tmp_path, _hide_blake3(tmp_path))
    assert got["algorithm"] == "sha256"
    assert got["tag"] == _SHA256_TAG.decode("ascii")
    assert got["scheme"] == "cash-bulk-sha256-tree-1"
    assert got["direct_max"] == LEAF_BYTES
    assert bytes.fromhex(got["digest"]) == _reference(_PROBE_DATA, hashlib.sha256, _SHA256_TAG)


def test_the_digest_is_the_one_the_running_algorithm_names():
    data = os.urandom(3 * LEAF_BYTES + 12345)
    if bulk_digest.ALGORITHM == "blake3":
        from blake3 import blake3

        leaf = blake3
    else:
        leaf = hashlib.sha256
    assert bulk_digest.bulk_digest(data) == _reference(data, leaf, bulk_digest.TAG)
    assert bulk_digest.ALGORITHM.encode() in bulk_digest.TAG


@pytest.mark.parametrize("n", [0, 1, 100, 16 * 1024, 16 * 1024 + 1, LEAF_BYTES + 3, 9 * LEAF_BYTES + 1])
def test_the_same_bytes_give_the_same_digest_however_handed_over(n, monkeypatch):
    data = os.urandom(n)
    first = bulk_digest.bulk_digest(data)
    assert bulk_digest.bulk_digest(bytearray(data)) == first
    h = BulkHasher()
    for i in range(0, n, 70_001):
        h.update(memoryview(data)[i : i + 70_001])
    assert h.digest() == first
    monkeypatch.setattr(bulk_digest, "PARALLEL_MIN_BYTES", LEAF_BYTES)
    monkeypatch.setattr(bulk_digest, "_THREADS", 3)
    assert bulk_digest.bulk_digest(data) == first


def test_blake3_digests_apart_from_sha256_under_its_own_tag(tmp_path):
    pytest.importorskip("blake3")
    assert bulk_digest.ALGORITHM == "blake3"
    with_it = _probe(tmp_path)
    without = _probe(tmp_path, _hide_blake3(tmp_path))
    assert with_it["blake3_loaded"] and not without["blake3_loaded"]
    assert with_it["tag"] != without["tag"]
    assert "blake3" in with_it["tag"] and "sha256" in without["tag"]
    assert with_it["scheme"] != without["scheme"]
    assert with_it["digest"] != without["digest"]


def test_under_blake3_a_buffer_past_16_kib_folds_in_as_its_tree_digest():
    """BLAKE3 hashes 16 KiB in a twelfth of SHA-256's time, so past that a
    buffer goes into a running SHA-256 as its tree digest; the tag and the
    length still keep the two forms apart."""
    pytest.importorskip("blake3")
    small, big = os.urandom(16 * 1024), os.urandom(16 * 1024 + 1)
    for data, mark in ((small, b"r"), (big, b"t")):
        h, want = hashlib.sha256(), hashlib.sha256()
        fold_buffer(h, data)
        want.update(mark + len(data).to_bytes(8, "little"))
        want.update(data if mark == b"r" else bulk_digest.bulk_digest(data))
        assert h.digest() == want.digest()


def test_a_hasher_thread_never_starts_blake3_threads_of_its_own():
    """``max_threads`` stays at blake3's default (one): the threads a digest
    uses are `hashing_threads`, never that times blake3's pool."""
    blake3 = pytest.importorskip("blake3")
    assert bulk_digest.leaf_hasher is blake3.blake3
    assert 1 <= bulk_digest.hashing_threads() <= bulk_digest.BLAKE3_MAX_THREADS


_JOB = textwrap.dedent("""
    import json, sys
    import numpy as np
    import cash
    from cash import bulk_digest

    calls = []

    @cash.cache(assume_safe=True)
    def total(a):
        calls.append(1)
        return float(a.sum())

    big = np.arange(400_000, dtype=np.float64)  # 3.2 MB: hashed as a tree
    print(json.dumps({"algorithm": bulk_digest.ALGORITHM, "value": total(big), "ran": len(calls)}))
""")


def _job(tmp_path, env=None) -> dict:
    done = run_python("-c", _JOB, cwd=tmp_path, env=env)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_cache_written_without_blake3_misses_with_it_and_hits_again_without(tmp_path):
    pytest.importorskip("blake3")
    pytest.importorskip("numpy")
    hidden = _hide_blake3(tmp_path)
    first = _job(tmp_path, hidden)
    assert first["algorithm"] == "sha256" and first["ran"] == 1
    again = _job(tmp_path, hidden)
    assert again["ran"] == 0  # the cache works across processes at all
    fast = _job(tmp_path)
    assert fast["algorithm"] == "blake3"
    assert fast["ran"] == 1, "an entry keyed by SHA-256 leaves was served under BLAKE3"
    assert fast["value"] == first["value"]
    assert _job(tmp_path)["ran"] == 0
    assert _job(tmp_path, hidden)["ran"] == 0  # each install's entry stands


def test_a_remembered_file_digest_of_the_other_scheme_is_not_used():
    from cash.tracking import digest_table

    key = ["/data/x.csv", 1, 2, 3, 4, 5]
    ours = json.dumps([digest_table._SCHEME, *key, "a" * 64, 1.0])
    other_scheme = "cash-bulk-sha256-tree-1" if bulk_digest.ALGORITHM == "blake3" else "cash-bulk-blake3-tree-1"
    theirs = json.dumps([other_scheme, *key, "a" * 64, 1.0])
    assert digest_table._parse(ours) is not None
    assert digest_table._parse(theirs) is None


def test_a_row_fingerprint_of_the_other_scheme_compares_with_nothing():
    from cash.row_blocks import RowBlocks

    blocks = RowBlocks("ndarray", 100, 10, ("0" * 16,) * 10)
    assert RowBlocks.from_dict(blocks.to_dict()) == blocks
    other = dict(blocks.to_dict(), scheme="cash-bulk-other-tree-1")
    assert RowBlocks.from_dict(other) is None
    old = {k: v for k, v in blocks.to_dict().items() if k != "scheme"}
    expect_old = blocks if bulk_digest.ALGORITHM == "sha256" else None
    assert RowBlocks.from_dict(old) == expect_old


def test_digests_taken_on_several_threads_at_once_agree(monkeypatch):
    """Each leaf gets a hasher of its own, so digests taken at once on
    several threads -- each of them threaded inside -- agree."""
    import threading

    monkeypatch.setattr(bulk_digest, "PARALLEL_MIN_BYTES", LEAF_BYTES)
    monkeypatch.setattr(bulk_digest, "_THREADS", 3)
    data = os.urandom(6 * LEAF_BYTES + 99)
    want = _reference(data, bulk_digest.leaf_hasher, bulk_digest.TAG)
    got: list[bytes] = []

    def run():
        for _ in range(3):
            got.append(bulk_digest.bulk_digest(data))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == [want] * 12
