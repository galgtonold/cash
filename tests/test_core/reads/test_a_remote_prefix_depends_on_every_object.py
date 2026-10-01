"""A read of a remote prefix or glob depends on every object under it.

``pd.read_parquet("s3://bucket/p/")`` reads every partition under ``p/``. The
prefix was tracked by what ``fs.info`` says about it, and s3fs reports a
prefix as ``{"type": "directory", "size": 0}``: no ETag, so the token was
``size:0`` for good. A new dated partition never invalidated the entry and the
old total was served, while REMOTE-SIZE-ONLY called that case safe. A glob
fared worse: ``info`` on a pattern raises, so it never cached at all.

The token of a prefix or glob is now its listing: each object's name and
validator, so a new, removed or edited object moves it.
"""

from __future__ import annotations

import io
import warnings

import pytest

from cash import Cash, InMemoryBackend, RemoteFileDataSource
from cash.remote_source import REMOTE_LEDGER

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"), pytest.mark.timeout(120)]

BUCKET = "cash-prefix"


def _memory_fs():
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    fs.rm("/prefix-test", recursive=True) if fs.exists("/prefix-test") else None
    return fs


@pytest.mark.parametrize("url", ["memory://prefix-test/p", "memory://prefix-test/p/", "memory://prefix-test/p/*.csv"])
def test_a_new_object_moves_the_token(url):
    fs = _memory_fs()
    fs.pipe("/prefix-test/p/2026-09-01.csv", b"x\n1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the memory filesystem reports sizes only
        before = RemoteFileDataSource(url).state_token()
        again = RemoteFileDataSource(url).state_token()
        fs.pipe("/prefix-test/p/2026-09-02.csv", b"x\n9\n")
        after = RemoteFileDataSource(url).state_token()
    assert not before.startswith("unresolved:"), before
    assert before == again, "an unchanged prefix must hold its token"
    assert before != after, "a new object under the prefix did not move the token"


def test_an_object_outside_a_glob_does_not_move_it():
    fs = _memory_fs()
    fs.pipe("/prefix-test/p/a.csv", b"x\n1\n")
    url = "memory://prefix-test/p/*.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = RemoteFileDataSource(url).state_token()
        fs.pipe("/prefix-test/p/notes.txt", b"hello")
        assert RemoteFileDataSource(url).state_token() == before


@pytest.fixture(scope="module")
def _moto_server():
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")

    server = moto_server.ThreadedMotoServer(port=0, verbose=False)
    server.start()
    try:
        yield f"http://127.0.0.1:{server.get_host_and_port()[1]}"
    finally:
        server.stop()


@pytest.fixture
def s3(_moto_server, monkeypatch):
    import boto3
    import s3fs

    monkeypatch.setenv("AWS_ENDPOINT_URL", _moto_server)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    client = boto3.client("s3", endpoint_url=_moto_server)
    try:
        client.create_bucket(Bucket=BUCKET)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    for obj in client.list_objects_v2(Bucket=BUCKET).get("Contents", []):
        client.delete_object(Bucket=BUCKET, Key=obj["Key"])
    s3fs.S3FileSystem.clear_instance_cache()
    REMOTE_LEDGER.reset()
    try:
        yield client
    finally:
        REMOTE_LEDGER.reset()


def _put_partition(client, key, value):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buf = io.BytesIO()
    pq.write_table(pa.table({"x": [value]}), buf)
    client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def test_a_glob_token_is_a_listing_of_etags(s3):
    s3.put_object(Bucket=BUCKET, Key="p/a.parquet", Body=b"1")
    url = f"s3://{BUCKET}/p/*.parquet"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        before = RemoteFileDataSource(url).state_token()
    assert before.startswith("listing:1:"), before
    assert not rec, [str(w.message) for w in rec]
    s3.put_object(Bucket=BUCKET, Key="p/a.parquet", Body=b"2")
    assert RemoteFileDataSource(url).state_token() != before, "an edited object did not move the token"


def test_a_new_partition_recomputes(s3):
    target = f"s3://{BUCKET}/p/"
    pd = pytest.importorskip("pandas")
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    calls: list[int] = []

    @cash.cache
    def total(url):
        calls.append(1)
        return int(pd.read_parquet(url)["x"].sum())

    _put_partition(s3, "p/2026-09-01.parquet", 1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(target) == 1
        assert total(target) == 1
    assert len(calls) == 1, "an unchanged prefix must hit"
    codes = [getattr(w.message, "code", None) for w in rec]
    assert "REMOTE-SIZE-ONLY" not in codes and "REMOTE-STATE-UNREADABLE" not in codes, codes

    _put_partition(s3, "p/2026-09-02.parquet", 10)
    assert total(target) == 11, "a new partition was not seen"
    assert len(calls) == 2
