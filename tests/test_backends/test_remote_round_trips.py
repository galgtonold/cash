"""How many times does one cache operation cross the network?

For the local backends the interesting number is microseconds. For Redis and
S3 it is not: a loopback benchmark measures loopback, and a benchmark against
a real endpoint measures whoever's network ran it. What survives both is the
ROUND TRIP COUNT, and the bytes transferred -- both machine-independent, so
both assertable.

Two things this found:

* ``get_metadata`` had no override on any remote backend, so it inherited
  ``CacheBackend.get_metadata``, which performs a full ``get()`` and throws
  the value away. Asking "when was this last accessed?" downloaded the whole
  cached object. On S3 that is an extra request AND the entire payload over
  the wire; on Redis it is the entire payload.
* ``S3Backend.delete`` issued one request per object where S3 has had a batch
  delete for as long as it has had a delete.

The doubles live in ``remote_doubles``; they store real bytes, so a count
assertion here is also asserting the operation worked.
"""
from __future__ import annotations

import pytest

from .remote_doubles import FakeRedisClient, FakeS3Client

BIG = 4 * 1024 * 1024


@pytest.fixture
def s3():
    pytest.importorskip("boto3")
    from unittest.mock import patch

    from cash.backends.s3_backend import S3Backend
    with patch("boto3.client"):
        backend = S3Backend(bucket="b", prefix="p/")
    backend.s3 = FakeS3Client()
    backend.bucket = "b"
    return backend


@pytest.fixture
def redis():
    pytest.importorskip("redis")
    from unittest.mock import patch

    from cash.backends.redis_backend import RedisBackend
    with patch("redis.Redis"):
        backend = RedisBackend(prefix="p:")
    backend.client = FakeRedisClient()
    return backend


def _seed(backend, payload=b"v" * BIG):
    backend.set("k", payload, {"execution_time": 1.0})
    backend._writes.wait_all()
    backend.s3.reset() if hasattr(backend, "s3") else backend.client.reset()
    return payload


def _wire(backend):
    return backend.s3 if hasattr(backend, "s3") else backend.client


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def test_s3_reading_metadata_does_not_download_the_value(s3):
    """The hole: inspecting an entry pulled the whole object across the wire.

    One ranged GET now fetches the front of the object. The prefetch is
    deliberately generous (8KB against ~200 bytes of metadata) because on S3
    a request costs orders of magnitude more than the bytes: paying for 8KB
    beats risking a second round trip.
    """
    from cash.backends.s3_backend import S3Backend

    payload = _seed(s3)

    meta = s3.get_metadata("k")

    assert meta is not None and meta["execution_time"] == 1.0
    wire = _wire(s3)
    assert wire.round_trips == 1, (
        f"{wire.round_trips} requests to read metadata: {dict(wire.calls)}"
    )
    assert wire.bytes_out <= S3Backend.METADATA_PREFETCH_BYTES, (
        f"downloaded {wire.bytes_out:,} bytes, more than the prefetch"
    )
    assert wire.bytes_out < len(payload) / 100, (
        f"downloaded {wire.bytes_out:,} bytes of a {len(payload):,}-byte value"
    )


def test_s3_reading_the_value_still_downloads_it(s3):
    """The control. A get_metadata that returned nothing would pass the above."""
    payload = _seed(s3)

    metadata, value = s3.get("k")

    assert value == payload and metadata is not None
    assert _wire(s3).bytes_out >= len(payload)


def test_s3_metadata_larger_than_the_prefetch_costs_a_second_request(s3):
    """Correctness above speed: a fat entry must still read, not fail.

    Notebook entries can carry outputs, lineages and source code. If one ever
    outgrows the prefetch the backend asks again for exactly the span the
    header declares, rather than reporting a readable entry as absent.
    """
    from cash.backends.s3_backend import S3Backend

    fat = {"execution_time": 1.0, "code": "x" * (S3Backend.METADATA_PREFETCH_BYTES * 2)}
    s3.set("fat", b"v" * 1000, fat)
    s3._writes.wait_all()
    _wire(s3).reset()

    meta = s3.get_metadata("fat")

    assert meta is not None and len(meta["code"]) == S3Backend.METADATA_PREFETCH_BYTES * 2
    assert _wire(s3).calls["get_object"] == 2, (
        "the widened re-read did not happen"
    )


def test_s3_delete_is_one_request(s3):
    _seed(s3)

    s3.delete("k")

    wire = _wire(s3)
    assert wire.round_trips == 1, (
        f"{wire.round_trips} requests to delete one entry: {dict(wire.calls)}"
    )
    assert s3.get("k") == (None, None)


def test_s3_delete_raises_on_a_real_failure(s3):
    """A delete that genuinely failed must not read as success."""
    from cash.exceptions import CacheBackendError

    _seed(s3, payload=b"x" * 100)

    def denied(**_kw):
        raise _client_error_code("AccessDenied")

    s3.s3.delete_object = denied
    with pytest.raises(CacheBackendError):
        s3.delete("k")


def _client_error_code(code: str):
    import botocore.exceptions
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": code}}, "DeleteObject")


def test_s3_a_write_and_a_read_are_each_one_request(s3):
    """It was two of each: an entry was two objects.

    The write also had an ordering constraint -- data first, metadata second,
    so a reader could never find metadata pointing at a payload that was not
    there yet -- and a cleanup path for when the second PUT failed. One object
    removes the constraint along with the request.
    """
    wire = _wire(s3)
    s3.set("k2", b"x" * 100, {"execution_time": 1.0})
    s3._writes.wait_all()
    assert wire.calls["put_object"] == 1, dict(wire.calls)

    wire.reset()
    metadata, value = s3.get("k2")
    assert value == b"x" * 100
    assert wire.calls["get_object"] == 1, dict(wire.calls)


def test_s3_listing_does_not_download_the_values(s3):
    """`list_entries` reports every entry; it must not fetch every payload."""
    for i in range(5):
        s3.set(f"e{i}", b"v" * (512 * 1024), {"execution_time": 1.0})
    s3._writes.wait_all()
    _wire(s3).reset()

    entries = s3.list_entries()

    assert len(entries) == 5, entries
    assert _wire(s3).bytes_out < 5 * 512 * 1024 / 10, (
        f"listing 5 entries transferred {_wire(s3).bytes_out:,} bytes"
    )


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

def test_redis_reading_metadata_does_not_transfer_the_value(redis):
    payload = _seed(redis)

    meta = redis.get_metadata("k")

    assert meta is not None and meta["execution_time"] == 1.0
    wire = _wire(redis)
    assert wire.bytes_out < 4096, (
        f"transferred {wire.bytes_out:,} bytes of a {len(payload):,}-byte "
        f"value to read its metadata"
    )


def test_redis_get_and_set_are_one_round_trip_each(redis):
    """Already true, and worth pinning: both keys ride one pipeline.

    A change that split them would still issue the same two commands, so only
    a test that counts round trips rather than commands would notice.
    """
    wire = _wire(redis)
    redis.set("k2", b"x" * 100, {"execution_time": 1.0})
    redis._writes.wait_all()
    assert wire.round_trips == 1, dict(wire.calls)

    wire.reset()
    redis.get("k2")
    assert wire.round_trips == 1, dict(wire.calls)


def test_redis_delete_is_one_round_trip(redis):
    _seed(redis)
    redis.delete("k")
    assert _wire(redis).round_trips == 1, dict(_wire(redis).calls)


# ---------------------------------------------------------------------------
# SQLite -- not remote, same hole
# ---------------------------------------------------------------------------

def test_sqlite_reading_metadata_does_not_deserialize_the_value(tmp_path):
    """Local, but it inherited the same ``get_metadata`` and paid the same way.

    Measured rather than counted here because there is no wire: a 16MB entry
    cost 35.9ms to inspect against 0.226ms for the file backend, because the
    inherited implementation unpickled the whole value to reach the metadata
    attached to it.
    """
    from cash.backends.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "c.db"))
    try:
        marker = _UnpicklingCanary()
        backend.set("k", marker, {"execution_time": 1.0})
        backend._writes.wait_all()
        _UnpicklingCanary.loads = 0

        meta = backend.get_metadata("k")

        assert meta is not None and meta["execution_time"] == 1.0
        assert _UnpicklingCanary.loads == 0, (
            "the value was deserialized to answer a metadata question"
        )
    finally:
        backend.shutdown()


def test_sqlite_keeps_the_payload_column_last(tmp_path):
    """Column ORDER is a performance property here, so it is asserted.

    SQLite lays a row out in declaration order and spills the overflow onto a
    page chain, so reading a column means walking past everything declared
    before it. With ``data`` ahead of ``metadata``, ``SELECT metadata`` on a
    16MB entry walked 16MB: measured at 7.398ms against 0.003ms with the two
    swapped. That is 2845x for a change that moves no bytes, and nothing else
    in the suite would notice it being undone.
    """
    from cash.backends.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "c.db"))
    try:
        cols = [row[1] for row in
                backend._conn.execute("PRAGMA table_info(cache_entries)")]
        assert cols.index("metadata") < cols.index("data"), (
            f"the payload column is not last: {cols}"
        )
    finally:
        backend.shutdown()


def test_sqlite_rebuilds_a_table_written_with_the_old_column_order(tmp_path):
    import pickle
    import sqlite3

    from cash.backends.sqlite_backend import SQLiteBackend

    db = str(tmp_path / "c.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE cache_entries (key TEXT PRIMARY KEY, data BLOB NOT NULL, "
        "metadata BLOB NOT NULL, size_bytes INTEGER DEFAULT 0, "
        "created_at REAL NOT NULL, last_access REAL NOT NULL, "
        "access_count INTEGER DEFAULT 0, ttl INTEGER DEFAULT NULL, "
        "serializer_cls TEXT DEFAULT 'PickleSerializer')"
    )
    conn.execute(
        "INSERT INTO cache_entries VALUES (?,?,?,?,?,?,0,NULL,?)",
        ("old", pickle.dumps("v"), pickle.dumps({"key": "old"}), 1, 0.0, 0.0,
         "PickleSerializer"),
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(db)
    try:
        cols = [row[1] for row in
                backend._conn.execute("PRAGMA table_info(cache_entries)")]
        assert cols.index("metadata") < cols.index("data"), cols
        assert backend.get("old") == (None, None), (
            "the pre-migration entry survived; it cannot be read from the new "
            "schema and must not be reported as a hit"
        )
        backend.set("k", {"a": 1}, {"execution_time": 1.0})
        backend._writes.wait_all()
        assert backend.get("k")[1] == {"a": 1}
    finally:
        backend.shutdown()


def test_sqlite_does_not_rebuild_a_current_table(tmp_path):
    """The control. A migration that fired every time would wipe the cache on
    every open, and every arm above would still pass."""
    from cash.backends.sqlite_backend import SQLiteBackend

    db = str(tmp_path / "c.db")
    first = SQLiteBackend(db)
    first.set("k", {"a": 1}, {"execution_time": 1.0})
    first._writes.wait_all()
    first.shutdown()

    second = SQLiteBackend(db)
    try:
        assert second.get("k")[1] == {"a": 1}, "reopening the cache wiped it"
    finally:
        second.shutdown()


class _UnpicklingCanary:
    """Counts its own unpicklings, so a value read cannot go unnoticed."""

    loads = 0

    def __reduce__(self):
        return (_revive, ())


def _revive():
    _UnpicklingCanary.loads += 1
    return _UnpicklingCanary()
