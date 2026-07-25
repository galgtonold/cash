"""RemoteFileDataSource: track a remote object by the store's own validator.

The point of the class is that a remote object cannot be checked by reading it,
so the store's ETag / version id / generation stands in for its content. These
tests pin the three things that makes-or-breaks:

* the token *moves* when the object moves (or the cache never invalidates),
* it costs nothing when the URL already pins a version (or every check is a
  round trip that could not have told us anything),
* failure is **closed** - an unreachable store recomputes rather than serving a
  result whose freshness nobody verified.

The HTTP path is exercised against a real local server rather than a mocked
``urlopen``: HEAD semantics, header casing and the ranged-GET fallback are
exactly the parts a mock would define into existence.
"""
from __future__ import annotations

import sys
import threading
import types
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cash import Cash, CashCacheIneffectiveWarning, InMemoryBackend, RemoteFileDataSource
from cash.exceptions import DependencyNotFoundError
from cash.remote_source import _reset_remote_warnings, pinned_version


@pytest.fixture(autouse=True)
def _fresh_warning_ledgers():
    """Each test starts with an empty warn-once ledger, as a session would."""
    _reset_remote_warnings()
    yield
    _reset_remote_warnings()


# ---------------------------------------------------------------------------
# A real local origin, so the HTTP path is tested against HTTP.
# ---------------------------------------------------------------------------

class _Origin:
    """Serves one object whose validators the test controls."""

    def __init__(self):
        self.etag: str | None = '"v1"'
        self.last_modified: str | None = None
        self.allow_head = True
        self.requests: list[str] = []


@pytest.fixture
def origin():
    state = _Origin()

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, body: bytes | None):
            state.requests.append(self.command)
            if self.command == "HEAD" and not state.allow_head:
                # What a presigned URL or a strict CDN does: the method is
                # refused, not the object.
                self.send_response(405)
                self.end_headers()
                return
            self.send_response(200)
            if state.etag is not None:
                self.send_header("ETag", state.etag)
            if state.last_modified is not None:
                self.send_header("Last-Modified", state.last_modified)
            self.send_header("Content-Length", "3")
            self.end_headers()
            if body is not None:
                self.wfile.write(body)

        def do_HEAD(self):
            self._respond(None)

        def do_GET(self):
            self._respond(b"abc")

        def log_message(self, *args):
            pass  # keep pytest output clean

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/data.csv"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestHttpToken:
    def test_token_tracks_the_etag(self, origin):
        source = RemoteFileDataSource(origin.url)
        first = source.state_token()
        assert source.state_token() == first, "an unchanged object must hold its token"

        origin.etag = '"v2"'
        assert source.state_token() != first, "a changed ETag must move the token"

    def test_head_is_the_request_made(self, origin):
        RemoteFileDataSource(origin.url).state_token()
        assert origin.requests == ["HEAD"], (
            "checking freshness must not download the object"
        )

    def test_falls_back_to_a_ranged_get_when_head_is_refused(self, origin):
        origin.allow_head = False
        token = RemoteFileDataSource(origin.url).state_token()
        assert token == 'etag:"v1"'
        assert origin.requests == ["HEAD", "GET"]

    def test_last_modified_is_used_when_there_is_no_etag(self, origin):
        origin.etag = None
        origin.last_modified = "Wed, 21 Oct 2026 07:28:00 GMT"
        source = RemoteFileDataSource(origin.url)
        first = source.state_token()
        assert "mtime:" in first

        origin.last_modified = "Thu, 22 Oct 2026 07:28:00 GMT"
        assert source.state_token() != first

    def test_size_only_object_warns_that_it_is_a_weak_token(self, origin):
        origin.etag = None
        origin.last_modified = None
        with pytest.warns(CashCacheIneffectiveWarning, match="by size alone"):
            token = RemoteFileDataSource(origin.url).state_token()
        assert token.startswith("size:")


# ---------------------------------------------------------------------------
# Version-pinned URLs: immutable by the storage contract, so free to check.
# ---------------------------------------------------------------------------

class TestPinnedVersions:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("s3://bucket/key?versionId=abc123", "versionid:abc123"),
            ("s3://bucket/key?other=1&versionId=abc123", "versionid:abc123"),
            ("gs://bucket/key#generation=17", "generation:17"),
            ("https://host/key?versionId=xyz", "versionid:xyz"),
        ],
    )
    def test_recognised(self, url, expected):
        assert pinned_version(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "s3://bucket/key",
            "s3://bucket/releases/v1.2.3/data.parquet",  # looks pinned, isn't
            "https://host/key?versionId=",                # empty pin is no pin
            "gs://bucket/generation",
        ],
    )
    def test_not_inferred_from_anything_but_a_real_pin(self, url):
        assert pinned_version(url) is None
        assert RemoteFileDataSource(url).immutable is False

    def test_pinned_url_resolves_without_touching_the_network(self, monkeypatch):
        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("a pinned version needs no request")

        monkeypatch.setattr("cash.remote_source._http_token", explode)
        monkeypatch.setattr("cash.remote_source._fsspec_token", explode)

        source = RemoteFileDataSource("s3://bucket/key?versionId=abc123")
        assert source.immutable is True
        assert source.state_token() == "versionid:abc123"


class TestImmutableAndMaxAge:
    def test_immutable_resolves_once(self, origin):
        source = RemoteFileDataSource(origin.url, immutable=True)
        first = source.state_token()

        origin.etag = '"v2"'
        assert source.state_token() == first, (
            "immutable=True is a promise the object cannot change; honouring it "
            "is the whole point of the flag"
        )
        assert origin.requests == ["HEAD"]

    def test_max_age_reuses_the_token_inside_the_window(self, origin):
        source = RemoteFileDataSource(origin.url, max_age=300)
        first = source.state_token()
        origin.etag = '"v2"'
        assert source.state_token() == first
        assert origin.requests == ["HEAD"]

    def test_default_revalidates_every_check(self, origin):
        source = RemoteFileDataSource(origin.url)
        source.state_token()
        source.state_token()
        assert origin.requests == ["HEAD", "HEAD"], (
            "max_age defaults to 0 - correctness first"
        )


# ---------------------------------------------------------------------------
# Failure is closed.
# ---------------------------------------------------------------------------

class TestFailureIsClosed:
    def test_unreachable_store_yields_a_fresh_token_each_time(self, monkeypatch):
        def unreachable(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr("cash.remote_source._http_token", unreachable)
        source = RemoteFileDataSource("https://host/key")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = source.state_token()
            second = source.state_token()

        assert first != second, (
            "a repeated 'unresolved' token would let the second failure serve "
            "the entry the first one stored - the stale hit this class prevents"
        )
        ineffective = [w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
        assert len(ineffective) == 1, "warn once, not once per check"
        assert "recompute" in str(ineffective[0].message)

    def test_a_missing_library_is_raised_not_swallowed(self, monkeypatch):
        # A silent forever-recompute would hide an install problem the caller
        # can actually fix.
        monkeypatch.setitem(sys.modules, "fsspec", None)
        with pytest.raises(DependencyNotFoundError, match="fsspec"):
            RemoteFileDataSource("s3://bucket/key").state_token()


# ---------------------------------------------------------------------------
# fsspec-addressed objects.
# ---------------------------------------------------------------------------

def _fake_fsspec(info: dict):
    """A stand-in fsspec exposing exactly the surface the resolver uses."""
    class FS:
        def info(self, path):
            return dict(info)

    module = types.ModuleType("fsspec")
    module.core = types.SimpleNamespace(url_to_fs=lambda url, **kw: (FS(), url))
    return module


class TestFsspecToken:
    @pytest.mark.parametrize(
        "info, expected",
        [
            ({"ETag": "abc", "size": 10}, "etag:abc"),
            ({"etag": "abc", "size": 10}, "etag:abc"),
            ({"generation": "17", "size": 10}, "generation:17"),
            ({"VersionId": "v9", "size": 10}, "versionid:v9"),
        ],
    )
    def test_prefers_a_strong_validator(self, monkeypatch, info, expected):
        monkeypatch.setitem(sys.modules, "fsspec", _fake_fsspec(info))
        assert RemoteFileDataSource("s3://bucket/key").state_token() == expected

    def test_falls_back_to_mtime_and_size(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "fsspec", _fake_fsspec({"LastModified": "2026-07-26", "size": 10})
        )
        token = RemoteFileDataSource("s3://bucket/key").state_token()
        assert token == "mtime:2026-07-26|size:10"

    def test_size_only_warns(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "fsspec", _fake_fsspec({"size": 10}))
        with pytest.warns(CashCacheIneffectiveWarning, match="by size alone"):
            RemoteFileDataSource("s3://bucket/key").state_token()

    def test_storage_options_reach_fsspec(self, monkeypatch):
        seen = {}

        class FS:
            def info(self, path):
                return {"ETag": "abc"}

        module = types.ModuleType("fsspec")

        def url_to_fs(url, **kw):
            seen.update(kw)
            return FS(), url

        module.core = types.SimpleNamespace(url_to_fs=url_to_fs)
        monkeypatch.setitem(sys.modules, "fsspec", module)

        RemoteFileDataSource(
            "s3://bucket/key", storage_options={"profile": "analytics"}
        ).state_token()
        assert seen == {"profile": "analytics"}


# ---------------------------------------------------------------------------
# The whole point: it invalidates a real cache, identically on any machine.
# ---------------------------------------------------------------------------

class TestAsACacheDependency:
    def test_a_changed_object_invalidates_the_entry(self, origin):
        cash = Cash(backend=InMemoryBackend())
        calls = []

        @cash.cache(depends_on=[RemoteFileDataSource(origin.url)])
        def load():
            calls.append(1)
            return len(calls)

        assert load() == 1
        assert load() == 1, "unchanged object - must hit"
        assert len(calls) == 1

        origin.etag = '"v2"'
        assert load() == 2, "changed object - must recompute"

    def test_identity_is_the_url_so_it_matches_across_machines(self):
        # The portability failure this class exists to fix: a local path is a
        # fact about one filesystem, a URL is a fact about the object.
        url = "s3://bucket/events.parquet"
        assert RemoteFileDataSource(url).get_id() == f"remote:{url}"
