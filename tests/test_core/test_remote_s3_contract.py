"""What s3fs actually puts in ``fs.info()`` — the one thing cash can't decide.

``RemoteFileDataSource`` picks a state token by looking for known keys in an
fsspec ``info`` dict (``_STRONG_INFO_KEYS``). Every other test in the suite
feeds it a *fake* fsspec, which proves the preference logic and nothing about
the key names. If a future s3fs renamed ``ETag``, the fake would still pass and
every real s3 token would quietly degrade to ``mtime|size`` — weaker, still
functional, and invisible.

That is a fact about s3fs, so only s3fs can settle it. These tests run against
an in-process S3 emulator and assert the contract cash depends on.

The failure mode they exist to catch is silent degradation, not a crash — so
they assert on the token's *shape*, not merely that one was produced.
"""
from __future__ import annotations

import pytest

from cash import Cash, InMemoryBackend, RemoteFileDataSource
from cash.remote_source import _reset_remote_warnings

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

BUCKET = "cash-contract"
KEY = "events.parquet"
URL = f"s3://{BUCKET}/{KEY}"


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
    """A seeded bucket, reachable as a bare ``s3://`` URL.

    The endpoint and credentials go in the environment rather than into
    ``storage_options``, so the code under test is byte-for-byte what a user
    would write. No ``AWS_SESSION_TOKEN``: moto files buckets under an account
    derived from the credentials, and a bogus token sends the write and the
    read to different accounts.
    """
    import boto3
    import s3fs

    monkeypatch.setenv("AWS_ENDPOINT_URL", _moto_server)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    client = boto3.client("s3", endpoint_url=_moto_server)
    try:
        client.create_bucket(Bucket=BUCKET)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    for obj in client.list_objects_v2(Bucket=BUCKET).get("Contents", []):
        client.delete_object(Bucket=BUCKET, Key=obj["Key"])
    client.put_object(Bucket=BUCKET, Key=KEY, Body=b"col1,col2\nval1,val2\n")

    # s3fs caches filesystem instances and directory listings per process; a
    # later test would otherwise be served an earlier test's view of the bucket.
    s3fs.S3FileSystem.clear_instance_cache()

    _reset_remote_warnings()
    try:
        yield client
    finally:
        _reset_remote_warnings()


class TestS3fsContract:
    def test_info_still_carries_the_keys_cash_reads(self, s3):
        import fsspec

        fs, path = fsspec.core.url_to_fs(URL)
        info = fs.info(path)
        assert "ETag" in info, (
            "cash keys remote entries on the ETag; if s3fs renamed it, every s3 "
            f"token silently degrades to mtime|size. Got: {sorted(info)}"
        )
        assert "LastModified" in info, "the documented fallback must stay available"

    def test_the_token_is_an_etag_not_a_fallback(self, s3):
        token = RemoteFileDataSource(URL).state_token()
        assert token.startswith("etag:"), (
            f"expected the strong validator, got {token!r} - a size:/mtime: "
            f"prefix means cash fell back and freshness got weaker"
        )

    def test_the_token_moves_when_the_object_does(self, s3):
        before = RemoteFileDataSource(URL).state_token()
        s3.put_object(Bucket=BUCKET, Key=KEY, Body=b"col1,col2\nCHANGED,CHANGED\n")
        after = RemoteFileDataSource(URL).state_token()
        assert before != after

    def test_identical_bytes_hold_the_token(self, s3):
        """An ETag is content-derived, so a no-op rewrite must NOT invalidate.

        This is the half that makes the feature worth having: re-uploading the
        same data does not throw away everyone's cache.
        """
        before = RemoteFileDataSource(URL).state_token()
        s3.put_object(Bucket=BUCKET, Key=KEY, Body=b"col1,col2\nval1,val2\n")
        assert RemoteFileDataSource(URL).state_token() == before


class TestEndToEndAgainstRealS3Api:
    """Auto-tracking against the real S3 API, through a reader cash patches.

    ``pd.read_csv`` is used deliberately: the tracker only sees reads that go
    through its patched entry points, and pandas is one of them. A bare
    ``fsspec.open(url)`` is NOT tracked — the same category as a direct pyarrow
    call — so using it here would test nothing while looking like it did.
    """

    @staticmethod
    def _loader(cash, calls):
        import pandas as pd

        @cash.cache
        def load(url):
            calls.append(1)
            return pd.read_csv(url)

        return load

    def test_a_cached_function_invalidates_when_the_object_changes(self, s3):
        pytest.importorskip("pandas")
        cash = Cash(backend=InMemoryBackend())
        calls: list[int] = []
        load = self._loader(cash, calls)

        assert load(URL).iloc[0]["col2"] == "val2"
        assert load(URL).iloc[0]["col2"] == "val2"
        assert len(calls) == 1, "unchanged object - must hit"

        s3.put_object(Bucket=BUCKET, Key=KEY, Body=b"col1,col2\nCHANGED,CHANGED\n")
        assert load(URL).iloc[0]["col2"] == "CHANGED", "changed object - must recompute"
        assert len(calls) == 2

    def test_an_unreachable_object_recomputes_rather_than_serving_stale(self, s3):
        """Fail-closed against the real API, not a raised OSError in a stub.

        Deleting the object makes freshness unverifiable. The contract is that
        cash re-runs the body — which then fails on the missing object — rather
        than handing back the cached bytes as though nothing had happened. The
        exception is incidental; ``calls`` is the assertion that matters.
        """
        pytest.importorskip("pandas")
        import s3fs

        cash = Cash(backend=InMemoryBackend())
        calls: list[int] = []
        load = self._loader(cash, calls)

        load(URL)
        s3.delete_object(Bucket=BUCKET, Key=KEY)
        s3fs.S3FileSystem.clear_instance_cache()

        with pytest.raises(Exception):
            load(URL)
        assert len(calls) == 2, "freshness could not be confirmed, so it recomputed"
