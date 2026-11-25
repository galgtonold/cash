"""``clear()`` on S3 and Redis removes cache entries and nothing else.

Both cleared by listing the configured prefix and deleting whatever came
back. The prefix is a string, so ``team`` also names ``team-reports/q3.pdf``
and ``teamdata.csv``, and an empty prefix names the whole bucket or database:
clearing the cache deleted the user's other objects.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cash.backends.cache_dir import CACHE_FORMAT_VERSION, format_tag

from .remote_doubles import FakeRedisClient, FakeS3Client


def _s3(prefix: str):
    pytest.importorskip("boto3")
    from cash.backends.s3_backend import S3Backend

    with patch("boto3.client"):
        backend = S3Backend(bucket="b", prefix=prefix)
    backend.s3 = FakeS3Client()
    return backend


def _redis(prefix: str):
    pytest.importorskip("redis")
    from cash.backends.redis_backend import RedisBackend

    with patch("redis.Redis"):
        backend = RedisBackend(prefix=prefix)
    backend.client = FakeRedisClient()
    return backend


def _store(backend, *keys):
    for key in keys:
        backend.set(key, {"v": key}, {"execution_time": 1.0})
    backend._writes.wait_all()


@pytest.mark.parametrize("prefix", ["team", "team/", ""])
def test_s3_clear_keeps_objects_that_are_not_entries(prefix):
    backend = _s3(prefix)
    theirs = {
        ("b", "team-reports/q3.pdf"): b"important",
        ("b", "teamdata.csv"): b"a,b",
        ("b", "team/notes.txt"): b"n",
        ("b", "team-reports/model.entry"): b"an object of theirs that happens to end the same way",
        ("b", "f2/readme.txt"): b"a folder that looks like a format's",
        ("other-bucket", f"{prefix}{format_tag()}/k.entry"): b"another bucket",
    }
    backend.s3.store.update(theirs)
    _store(backend, "stmt:one", "mod.func:state:dyn:args")
    assert len(backend.s3.store) == len(theirs) + 2, "precondition: both entries were stored"

    backend.clear()

    assert backend.s3.store == theirs
    assert backend.list_entries() == []


@pytest.mark.parametrize("prefix", ["team", "team:", ""])
def test_redis_clear_keeps_keys_that_are_not_entries(prefix):
    backend = _redis(prefix)
    theirs = {
        "team-reports:q3": b"important",
        "teamdata": b"a,b",
        "team:session:42": b"s",
        "team-reports/x:data": b"a key of theirs that happens to end the same way",
        f"{prefix}lock:stmt:one": b"a lock",
        "f2:readme": b"a key that looks like a format's",
    }
    backend.client.store.update(theirs)
    _store(backend, "stmt:one", "mod.func:state:dyn:args")
    assert len(backend.client.store) == len(theirs) + 4, "precondition: both entries were stored, two keys each"

    backend.clear()

    assert backend.client.store == theirs
    assert backend.list_entries() == []


def test_redis_clear_reads_the_bytes_keys_a_real_client_returns():
    """redis-py hands keys back as bytes unless it was told to decode them."""
    backend = _redis("p:")
    _store(backend, "k")
    client = backend.client
    client.store["p:mine"] = b"x"
    scan, delete = client.scan, client.delete
    client.scan = lambda **kw: (0, [k.encode() for k in scan(**kw)[1]])
    client.delete = lambda *keys: delete(*(k.decode() for k in keys))

    backend.clear()

    assert set(client.store) == {"p:mine"}


@pytest.mark.parametrize("make, separator", [(_s3, "/"), (_redis, ":")])
def test_clear_takes_every_formats_entries_and_those_named_without_one(make, separator, monkeypatch):
    """Clearing is how entries of a format no cash reads any more leave a
    shared store, so it is not limited to the running build's format."""
    from cash.backends import cache_dir

    backend = make(f"p{separator}")
    store = backend.s3.store if hasattr(backend, "s3") else backend.client.store
    monkeypatch.setattr(cache_dir, "CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION - 1)
    _store(backend, "older")
    monkeypatch.setattr(cache_dir, "CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION)
    _store(backend, "current")
    # One from before names carried the format: the current entry's, renamed.
    tag = f"{format_tag()}{separator}"
    for name in list(store):
        text = name[1] if isinstance(name, tuple) else name
        if tag in text:
            renamed = text.replace(tag, "", 1).replace("current", "legacy")
            store[("b", renamed) if isinstance(name, tuple) else renamed] = store[name]
    count = len(store)
    assert count == (3 if hasattr(backend, "s3") else 6), "precondition"

    backend.clear()

    assert not store
