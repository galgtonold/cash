"""A remote object replaced while the call that read it is still running.

An entry's token for ``https://...`` or ``s3://...`` was taken when the entry
was stored, after the body had run. Another job publishing a new version of
the object in between left the OLD object's result recorded under the NEW
object's token: every later call found the entry fresh and served it. The
same race on a local file was already caught.

The token is now taken before the read as well. Where the two differ the
result is returned, not cached, and the call warns ``STORE-INPUT-CHANGED``.
"""

from __future__ import annotations

import multiprocessing
import threading
import warnings
from http.server import BaseHTTPRequestHandler

import pytest

from cash import Cash, InMemoryBackend
from cash.remote_source import REMOTE_LEDGER
from cash.tracking.file_tracker import FileAccessTracker
from tests._local_http import LocalThreadingHTTPServer

pd = pytest.importorskip("pandas")

V1 = b"a,b\n1,2\n"
V2 = b"a,b\n100,200\n"


@pytest.fixture(autouse=True)
def _fresh_ledgers():
    REMOTE_LEDGER.reset()
    yield
    REMOTE_LEDGER.reset()


class _Origin:
    """An object store with one object, and another job that replaces it
    at a chosen moment: as the n-th HEAD or the n-th GET is answered, so
    that answer is the last one of the old object."""

    def __init__(self) -> None:
        self.etag = '"v1"'
        self.body = V1
        self.heads = 0
        self.gets = 0
        self.publish_after: tuple[str, int] | None = None
        self.lock = threading.Lock()

    def answer(self, kind: str) -> tuple[str, bytes]:
        """The object as this request finds it."""
        with self.lock:
            if kind == "HEAD":
                self.heads += 1
                count = self.heads
            else:
                self.gets += 1
                count = self.gets
            found = self.etag, self.body
            if self.publish_after == (kind, count):
                self.publish_after = None
                self.etag, self.body = '"v2"', V2
            return found


class _Handler(BaseHTTPRequestHandler):
    def _headers(self, kind):
        etag, body = self.server.state.answer(kind)
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return body

    def do_HEAD(self):
        self._headers("HEAD")

    def do_GET(self):
        self.wfile.write(self._headers("GET"))

    def log_message(self, *args):
        pass


@pytest.fixture
def origin():
    state = _Origin()
    server = LocalThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/data.csv"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _codes(seen: list) -> list[str]:
    return sorted({str(w.message).split("]")[0].lstrip("[") for w in seen if str(w.message).startswith("[")})


def _stored_tokens(cash: Cash, url: str) -> list[str | None]:
    tokens = []
    for metadata in cash.backend.list_entries():
        entry = ((metadata or {}).get("auto_file_deps") or {}).get(url)
        if entry is not None:
            tokens.append(entry.get("hash"))
    return tokens


def total(url):
    df = pd.read_csv(url)
    return int(df["a"].sum())


class TestTheObjectIsReplacedWhileTheCallRuns:
    def test_after_the_read_the_result_is_not_stored_under_the_new_object(self, origin):
        cash = Cash(backend=InMemoryBackend())
        slow = cash.cache(total)
        origin.publish_after = ("GET", 1)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            first = slow(origin.url)
        assert first == 1, "the call itself returns what it computed"
        assert "STORE-INPUT-CHANGED" in _codes(seen)
        assert origin.url in " ".join(str(w.message) for w in seen)
        assert _stored_tokens(cash, origin.url) == [], "the old object's result was stored"

        assert slow(origin.url) == 100, "the next call reads the object as it is now"
        assert _stored_tokens(cash, origin.url) == ['etag:"v2"']
        gets = origin.gets
        assert slow(origin.url) == 100
        assert origin.gets == gets, "and the call after that is a hit"

    def test_between_the_token_and_the_read_it_is_not_stored_either(self, origin):
        """The token taken first describes v1, the bytes read are v2: no
        token says which, so the result is not kept under either."""
        cash = Cash(backend=InMemoryBackend())
        cached = cash.cache(total)
        origin.publish_after = ("HEAD", 1)

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            first = cached(origin.url)
        assert first == 100
        assert "STORE-INPUT-CHANGED" in _codes(seen)
        assert _stored_tokens(cash, origin.url) == []
        assert cached(origin.url) == 100

    def test_an_object_that_holds_still_is_stored_as_before(self, origin):
        cash = Cash(backend=InMemoryBackend())
        cached = cash.cache(total)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert cached(origin.url) == 1
            assert cached(origin.url) == 1
        assert origin.gets == 1
        assert _codes(seen) == []
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"']

    def test_a_miss_asks_the_store_twice_and_a_hit_once(self, origin):
        """One request before the read, one as the entry is stored; the
        check between them reuses the second."""
        cash = Cash(backend=InMemoryBackend())
        cached = cash.cache(total)
        cached(origin.url)
        assert origin.heads == 2
        cached(origin.url)
        assert origin.heads == 3

    def test_read_many_times_in_one_call_it_is_asked_once_before(self, origin):
        cash = Cash(backend=InMemoryBackend())

        @cash.cache
        def thrice(url):
            return sum(total(url) for _ in range(3))

        assert thrice(origin.url) == 3
        assert origin.gets == 3
        assert origin.heads == 2

    def test_inside_a_revalidation_window_the_entry_holds_the_token_from_before(self, origin):
        """With ``remote_revalidate_max_age_seconds`` the store is asked once,
        before the read, and the entry keeps that answer: once the window is
        over, the entry is stale for the object that replaced it."""
        cash = Cash(backend=InMemoryBackend(), remote_revalidate_max_age_seconds=3600)
        origin.publish_after = ("GET", 1)

        slow_total = cash.cache(total)
        assert slow_total(origin.url) == 1
        assert origin.heads == 1
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"'], "the old result is recorded under the new token"
        REMOTE_LEDGER.reset()  # the window is over
        assert slow_total(origin.url) == 100


class TestThroughACachedCallInside:
    def test_replaced_after_the_inner_hit_the_outer_result_is_not_stored(self, origin):
        cash = Cash(backend=InMemoryBackend())
        inner = cash.cache(total)
        assert inner(origin.url) == 1  # stored: heads == 2
        origin.publish_after = ("HEAD", 3)  # the inner hit's own check

        @cash.cache
        def outer(url):
            return inner(url) + 1

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert outer(origin.url) == 2
        assert "STORE-INPUT-CHANGED" in _codes(seen)
        assert outer(origin.url) == 101

    def test_replaced_during_the_inner_miss_neither_is_stored(self, origin):
        cash = Cash(backend=InMemoryBackend())
        origin.publish_after = ("GET", 1)

        inner = cash.cache(total)

        @cash.cache
        def outer(url):
            return inner(url) + 1

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert outer(origin.url) == 2
        assert _stored_tokens(cash, origin.url) == []
        assert outer(origin.url) == 101


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")
def test_read_in_a_pool_worker_then_replaced_it_is_not_stored(origin):
    """The worker takes the token before its read and hands it back with the
    read: asked again as the read is relayed, the store would already answer
    for the new object."""
    cash = Cash(backend=InMemoryBackend())
    origin.publish_after = ("GET", 1)

    @cash.cache
    def in_a_worker(url):
        with multiprocessing.get_context("fork").Pool(1) as pool:
            (value,) = pool.map(total, [url])
        return value

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert in_a_worker(origin.url) == 1
    assert "STORE-INPUT-CHANGED" in _codes(seen)
    assert _stored_tokens(cash, origin.url) == []
    assert in_a_worker(origin.url) == 100


class TestTheCheckItself:
    def test_the_tracker_keeps_the_token_from_before_the_first_read(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
            origin.etag = '"v2"'
            tracker.add_tracked_remote(origin.url)
        assert tracker.remote_as_read == {origin.url: 'etag:"v1"'}
        assert tracker.inputs_changed_since_read() == [origin.url]

    def test_an_unchanged_object_is_not_reported(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
        assert tracker.inputs_changed_since_read() == []

    def test_the_token_request_is_not_a_read_of_the_block(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
        assert tracker.get_accessed_files() == set()
        assert tracker.get_accessed_remote_urls() == {origin.url}

    def test_a_token_handed_over_is_kept_without_a_request(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url, 'etag:"v1"')
        assert origin.heads == 0
        assert tracker.remote_as_read == {origin.url: 'etag:"v1"'}

    def test_a_token_unreadable_before_and_readable_after_counts_as_moved(self):
        from cash.tracking.file_dep_snapshot import remote_moved

        assert remote_moved("unresolved:u:1", 'etag:"v1"')
        assert remote_moved('etag:"v1"', 'etag:"v2"')
        assert not remote_moved('etag:"v1"', 'etag:"v1"')

    def test_a_token_unreadable_now_is_left_to_the_snapshot(self):
        """Recorded as never fresh there; not also reported as changed."""
        from cash.tracking.file_dep_snapshot import remote_moved

        assert not remote_moved('etag:"v1"', "unresolved:u:2")


def test_a_store_that_cannot_be_asked_does_not_fail_the_recording(monkeypatch):
    """No filesystem package for the scheme: the read is recorded, and the
    snapshot reports the missing package when the entry is stored."""
    from cash.exceptions import DependencyNotFoundError
    from cash.tracking import file_dep_snapshot

    def no_package(url, warn=True):
        raise DependencyNotFoundError("no filesystem for gs://")

    monkeypatch.setattr(file_dep_snapshot, "remote_token_now", no_package)
    tracker = FileAccessTracker()
    tracker.track_path("gs://bucket/key.parquet")
    assert tracker.get_accessed_remote_urls() == {"gs://bucket/key.parquet"}
    assert tracker.remote_as_read == {}
    assert tracker.inputs_changed_since_read() == []
