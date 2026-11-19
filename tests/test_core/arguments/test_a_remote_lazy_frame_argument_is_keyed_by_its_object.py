"""A polars ``LazyFrame`` argument that scans a remote object recomputes after
the object changes.

The decorator records the files a scanned frame reads as the call's reads,
but left out a remote source: ``total(pl.scan_parquet("s3://bucket/k.parquet"))``
was keyed by the URL alone, and after the object was replaced the old total was
served, with no warning. The URL is now a remote read of the call, checked by
the store's validator (its ETag) on every lookup, as a read in the body is.
"""

from __future__ import annotations

import io
import warnings

import pytest

from cash.remote_source import REMOTE_LEDGER
from tests._moto import moto_s3_endpoint

pl = pytest.importorskip("polars")

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"), pytest.mark.timeout(120)]

BUCKET = "cash-lazy-remote"


@pytest.fixture(scope="module")
def endpoint():
    with moto_s3_endpoint() as url:
        yield url


@pytest.fixture
def s3(endpoint, monkeypatch):
    """A bucket polars and s3fs both reach through the environment."""
    import boto3
    import s3fs

    for name in ("AWS_ENDPOINT_URL_S3", "AWS_PROFILE", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_ALLOW_HTTP", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
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


def _put(client, key, values):
    buf = io.BytesIO()
    pl.DataFrame({"x": values}).write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def test_a_replaced_object_recomputes(s3, disk_cash):
    runs: list[int] = []

    @disk_cash.cache
    def total(lf):
        runs.append(1)
        return int(lf.collect()["x"].sum())

    url = f"s3://{BUCKET}/k.parquet"
    _put(s3, "k.parquet", [1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert total(pl.scan_parquet(url)) == 1
        assert total(pl.scan_parquet(url)) == 1
    assert "REMOTE-STATE-UNREADABLE" not in _codes(caught), [str(w.message) for w in caught]
    assert len(runs) == 1  # positive control: an unchanged object hits

    _put(s3, "k.parquet", [2, 3])
    assert total(pl.scan_parquet(url)) == 5
    assert len(runs) == 2


def test_a_new_object_under_a_scanned_glob_recomputes(s3, disk_cash):
    runs: list[int] = []

    @disk_cash.cache
    def total(lf):
        runs.append(1)
        return int(lf.collect()["x"].sum())

    pattern = f"s3://{BUCKET}/parts/*.parquet"
    _put(s3, "parts/a.parquet", [1])
    assert total(pl.scan_parquet(pattern)) == 1
    assert total(pl.scan_parquet(pattern)) == 1
    assert len(runs) == 1

    _put(s3, "parts/b.parquet", [10])
    assert total(pl.scan_parquet(pattern)) == 11
    assert len(runs) == 2
