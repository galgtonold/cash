"""A remote object replaced while the call that read it is still running.

An entry records a read of ``https://...`` or ``s3://...`` under the token
the store gave BEFORE the read, asked once. Another job publishing a new
version of the object while the call runs leaves an entry that names the old
object: the next call's check finds another token and recomputes, with no
warning. A token taken after the read would name the new object next to the
old object's result, and every later call would find that entry fresh.
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
    """The token each entry of *cash* that read *url* holds for it."""
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
    def test_after_the_read_the_entry_holds_the_old_token_and_the_next_call_recomputes(self, origin):
        cash = Cash(backend=InMemoryBackend())
        slow = cash.cache(total)
        origin.publish_after = ("GET", 1)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            first = slow(origin.url)
        assert first == 1, "the call itself returns what it computed"
        assert _codes(seen) == [], "a replaced object is found by the next call's check, without a warning"
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"'], (
            "the entry must name the object the result was computed from, never the one found afterwards"
        )

        assert slow(origin.url) == 100, "the next call reads the object as it is now"
        assert origin.gets == 2
        assert _stored_tokens(cash, origin.url) == ['etag:"v2"']
        assert slow(origin.url) == 100
        assert origin.gets == 2, "and the call after that is a hit"

    def test_between_the_token_and_the_read_the_next_call_recomputes_once(self, origin):
        """The token taken first describes v1, the bytes read are v2: the
        entry is stale for v2 and costs one more run, never a wrong result."""
        cash = Cash(backend=InMemoryBackend())
        cached = cash.cache(total)
        origin.publish_after = ("HEAD", 1)

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            first = cached(origin.url)
        assert first == 100
        assert _codes(seen) == []
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"']
        assert cached(origin.url) == 100
        assert origin.gets == 2
        assert _stored_tokens(cash, origin.url) == ['etag:"v2"']
        assert cached(origin.url) == 100
        assert origin.gets == 2

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

    def test_a_miss_asks_the_store_once_and_so_does_a_hit(self, origin):
        """The miss asks before the read and stores that answer."""
        cash = Cash(backend=InMemoryBackend())
        cached = cash.cache(total)
        cached(origin.url)
        assert origin.heads == 1
        cached(origin.url)
        assert origin.heads == 2

    def test_read_many_times_in_one_call_it_is_asked_once(self, origin):
        cash = Cash(backend=InMemoryBackend())

        @cash.cache
        def thrice(url):
            return sum(total(url) for _ in range(3))

        assert thrice(origin.url) == 3
        assert origin.gets == 3
        assert origin.heads == 1

    def test_replaced_between_two_reads_of_one_call_the_first_token_is_kept(self, origin):
        cash = Cash(backend=InMemoryBackend())
        origin.publish_after = ("GET", 1)

        @cash.cache
        def twice(url):
            return total(url) + total(url)

        assert twice(origin.url) == 101
        assert origin.heads == 1
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"']
        assert twice(origin.url) == 200

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
    def test_replaced_after_the_inner_hit_the_outer_entry_holds_the_inner_token(self, origin):
        cash = Cash(backend=InMemoryBackend())
        inner = cash.cache(total)
        assert inner(origin.url) == 1
        assert origin.heads == 1
        origin.publish_after = ("HEAD", 2)  # the inner hit's own check

        @cash.cache
        def outer(url):
            return inner(url) + 1

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert outer(origin.url) == 2
        assert "STORE-INPUT-CHANGED" not in _codes(seen)
        assert origin.heads == 2, "the token the inner entry was found fresh under is handed over, not asked again"
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"', 'etag:"v1"']
        assert outer(origin.url) == 101

    def test_replaced_during_the_inner_miss_both_recompute_next_time(self, origin):
        cash = Cash(backend=InMemoryBackend())
        origin.publish_after = ("GET", 1)

        inner = cash.cache(total)

        @cash.cache
        def outer(url):
            return inner(url) + 1

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert outer(origin.url) == 2
        assert _codes(seen) == []
        assert _stored_tokens(cash, origin.url) == ['etag:"v1"', 'etag:"v1"']
        assert outer(origin.url) == 101
        assert inner(origin.url) == 100


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")
def test_read_in_a_pool_worker_the_entry_holds_the_worker_s_token(origin):
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
    assert _codes(seen) == []
    assert _stored_tokens(cash, origin.url) == ['etag:"v1"']
    assert in_a_worker(origin.url) == 100


class TestTheRecordItself:
    def test_the_tracker_keeps_the_token_from_before_the_first_read(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
            origin.etag = '"v2"'
            tracker.add_tracked_remote(origin.url)
        assert tracker.remote_as_read == {origin.url: 'etag:"v1"'}

    def test_the_snapshot_records_that_token_without_a_request(self, origin):
        from cash.tracking.file_dep_snapshot import snapshot_remote_deps

        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
        origin.etag = '"v2"'
        heads = origin.heads
        snapshot = snapshot_remote_deps(tracker.get_accessed_remote_urls(), tracker.remote_as_read)
        assert snapshot == {origin.url: {"remote": True, "hash": 'etag:"v1"'}}
        assert origin.heads == heads

    def test_the_check_that_no_input_moved_does_not_ask_the_store(self, origin):
        with FileAccessTracker() as tracker:
            tracker.add_tracked_remote(origin.url)
        origin.etag = '"v2"'
        heads = origin.heads
        assert tracker.inputs_changed_since_read() == []
        assert origin.heads == heads

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

    def test_a_token_unreadable_before_the_read_is_never_replaced_by_one_from_after(self, origin):
        """The store answers again once the read is over: that answer may
        describe another object than the one read, so the read is recorded
        as one whose state cannot be read, and the entry is never fresh."""
        from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_remote_deps

        snapshot = snapshot_remote_deps([origin.url], {origin.url: f"unresolved:{origin.url}:7"})
        assert snapshot == {origin.url: {"remote": True, "unresolved": True}}
        assert file_dep_is_fresh(origin.url, snapshot[origin.url]) == (False, "remote-unresolved")

    def test_a_token_unreadable_before_and_after_warns_as_the_entry_is_stored(self):
        from cash.tracking.file_dep_snapshot import snapshot_remote_deps

        url = "http://127.0.0.1:9/gone.csv"
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with FileAccessTracker() as tracker:
                tracker.add_tracked_remote(url)
            assert _codes(seen) == [], "recording a read does not warn"
            assert tracker.remote_as_read[url].startswith("unresolved:")
            snapshot = snapshot_remote_deps([url], tracker.remote_as_read)
        assert snapshot == {url: {"remote": True, "unresolved": True}}
        assert _codes(seen) == ["REMOTE-STATE-UNREADABLE"]

    def test_a_read_no_tracker_took_a_token_for_is_asked_as_it_is_recorded(self, origin):
        from cash.tracking.file_dep_snapshot import snapshot_remote_deps

        assert snapshot_remote_deps([origin.url]) == {origin.url: {"remote": True, "hash": 'etag:"v1"'}}
        assert origin.heads == 1


def test_a_store_that_cannot_be_asked_does_not_fail_the_recording(monkeypatch):
    """No filesystem package for the scheme: the read is recorded, and the
    snapshot reports the missing package when the entry is stored."""
    from cash.exceptions import DependencyNotFoundError
    from cash.tracking import file_dep_snapshot

    def no_package(url):
        raise DependencyNotFoundError("no filesystem for gs://")

    monkeypatch.setattr(file_dep_snapshot, "remote_token_before_read", no_package)
    tracker = FileAccessTracker()
    tracker.track_path("gs://bucket/key.parquet")
    assert tracker.get_accessed_remote_urls() == {"gs://bucket/key.parquet"}
    assert tracker.remote_as_read == {}
    assert tracker.inputs_changed_since_read() == []


def test_a_token_request_that_fails_some_other_way_records_no_token_from_after(monkeypatch):
    """The entry of such a read is never fresh: a token asked for after the
    read could name an object that replaced the one read."""
    from cash.tracking import file_dep_snapshot

    def broken(url):
        raise RuntimeError("the client library failed")

    monkeypatch.setattr(file_dep_snapshot, "remote_token_before_read", broken)
    tracker = FileAccessTracker()
    tracker.track_path("http://127.0.0.1:9/data.csv")
    assert tracker.remote_as_read["http://127.0.0.1:9/data.csv"].startswith("unresolved:")
