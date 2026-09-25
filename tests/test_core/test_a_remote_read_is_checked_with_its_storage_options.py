"""A remote read is checked for freshness against the store it was read from.

``pd.read_parquet("s3://b/k", storage_options={"client_kwargs": {"endpoint_url":
...}})`` -- MinIO, an on-prem store -- recorded only the URL, so the freshness
check asked the default AWS endpoint with the ambient credentials. It failed,
every call recomputed, and REMOTE-STATE-UNREADABLE blamed the user's access.

The part of the options that names the store is written into the entry, so a
later process asks the same store; credentials stay in the process.
"""

from __future__ import annotations

import io
import os
import warnings

import pytest

from cash import Cash
from cash.remote_source import REMOTE_LEDGER, addressing_options

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"), pytest.mark.timeout(120)]

BUCKET = "cash-options"
SECRET = "do-not-write-this-secret"


def test_only_what_names_the_store_is_kept():
    options = {
        "key": "AKIA",
        "secret": SECRET,
        "token": "t",
        "anon": False,
        "client_kwargs": {"endpoint_url": "http://minio:9000", "aws_secret_access_key": SECRET},
        "config_kwargs": {"s3": {"addressing_style": "path"}},
        "headers": {"Authorization": SECRET},
    }
    assert addressing_options(options) == {
        "anon": False,
        "client_kwargs": {"endpoint_url": "http://minio:9000"},
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }


@pytest.fixture(scope="module")
def endpoint():
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")
    server = moto_server.ThreadedMotoServer(port=0, verbose=False)
    server.start()
    try:
        yield f"http://127.0.0.1:{server.get_host_and_port()[1]}"
    finally:
        server.stop()


@pytest.fixture
def s3(endpoint, monkeypatch):
    """A bucket reachable ONLY through storage_options: nothing in the
    environment points at the emulator, as for a MinIO user."""
    import boto3
    import s3fs

    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_PROFILE", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    # Where the options are missing, the check must fail fast rather than
    # wait on a real endpoint.
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("AWS_CONNECT_TIMEOUT", "2")

    client = boto3.client("s3", endpoint_url=endpoint)
    try:
        client.create_bucket(Bucket=BUCKET)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    s3fs.S3FileSystem.clear_instance_cache()
    REMOTE_LEDGER.reset()
    try:
        yield client
    finally:
        REMOTE_LEDGER.reset()


def _put(client, value):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buf = io.BytesIO()
    pq.write_table(pa.table({"x": [value]}), buf)
    client.put_object(Bucket=BUCKET, Key="k.parquet", Body=buf.getvalue())


def test_a_read_with_storage_options_caches_and_invalidates(s3, endpoint, tmp_path):
    pd = pytest.importorskip("pandas")
    options = {"key": "testing", "secret": SECRET, "client_kwargs": {"endpoint_url": endpoint}}
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    calls: list[int] = []

    @cash.cache
    def load(url):
        calls.append(1)
        return int(pd.read_parquet(url, storage_options=options)["x"].sum())

    url = f"s3://{BUCKET}/k.parquet"
    _put(s3, 1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert load(url) == 1
        # A later process: nothing held in memory, only what the entry wrote.
        REMOTE_LEDGER.reset()
        assert load(url) == 1
    codes = [getattr(w.message, "code", None) for w in rec]
    assert "REMOTE-STATE-UNREADABLE" not in codes, [str(w.message) for w in rec]
    assert len(calls) == 1, "the check did not reach the store the read used"

    _put(s3, 5)
    assert load(url) == 5
    assert len(calls) == 2

    # The store is written in the background: finish it, or the walk lists a
    # temp file that is renamed into place before it can be opened.
    cash.shutdown()
    written = b""
    for root, _dirs, files in os.walk(tmp_path / ".cash"):
        for name in files:
            with open(os.path.join(root, name), "rb") as fh:
                written += fh.read()
    assert endpoint.encode() in written, "the store the read went to was not written down"
    assert SECRET.encode() not in written, "a credential was written to the cache"
