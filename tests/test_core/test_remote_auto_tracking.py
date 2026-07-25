"""A remote read inside a cached function is tracked, by default.

Reading ``s3://bucket/key`` used to leave the entry with *no* recorded
dependency at all - the URL was mangled by ``realpath``, failed to stat, and was
dropped - so the function hit forever even after the object changed. The read is
now recorded on the tracker's remote channel and revalidated against the store's
own validator, exactly as a local file is revalidated against its content hash.

Auto-tracking is ON by default. The reasoning, from CAS-236: validation only
fires for a function that *already* reads from the network, so no purely local
call pays for it; a metadata request is tens of milliseconds against a download
that may be hundreds of megabytes; and on a hit you skip the download entirely.
The alternative default is silent staleness.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cash import Cash, InMemoryBackend
from cash.notebook.file_dep_snapshot import (
    file_dep_is_fresh,
    snapshot_file_deps,
    snapshot_remote_deps,
)
from cash.remote_source import _reset_remote_warnings


@pytest.fixture(autouse=True)
def _fresh_warning_ledgers():
    _reset_remote_warnings()
    yield
    _reset_remote_warnings()


class _Origin:
    def __init__(self):
        self.etag = '"v1"'
        self.body = b"a,b\n1,2\n"
        self.gets = 0


@pytest.fixture
def origin():
    """A local origin standing in for object storage."""
    state = _Origin()

    class Handler(BaseHTTPRequestHandler):
        def _headers(self):
            self.send_response(200)
            self.send_header("ETag", state.etag)
            self.send_header("Content-Length", str(len(state.body)))
            self.end_headers()

        def do_HEAD(self):
            self._headers()

        def do_GET(self):
            state.gets += 1
            self._headers()
            self.wfile.write(state.body)

        def log_message(self, *args):
            pass

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


class TestSnapshotShape:
    def test_a_remote_entry_carries_the_store_token(self, origin):
        snap = snapshot_remote_deps({origin.url})
        assert snap[origin.url]["remote"] is True
        assert snap[origin.url]["hash"] == 'etag:"v1"'

    def test_freshness_follows_the_object(self, origin):
        snap = snapshot_remote_deps({origin.url})
        assert file_dep_is_fresh(origin.url, snap[origin.url]) == (True, None)

        origin.etag = '"v2"'
        is_fresh, reason = file_dep_is_fresh(origin.url, snap[origin.url])
        assert (is_fresh, reason) == (False, "remote-changed")

    def test_an_unresolvable_object_is_recorded_and_reports_stale(self):
        # Fail-closed: recorded (so it is still checked) but never fresh, so the
        # call recomputes. Dropping it would leave no dependency at all, which
        # is the silent stale hit this mechanism exists to prevent.
        with pytest.warns(Warning):
            snap = snapshot_remote_deps({"http://127.0.0.1:9/gone.csv"})
        entry = snap["http://127.0.0.1:9/gone.csv"]
        assert entry["unresolved"] is True
        assert file_dep_is_fresh("http://127.0.0.1:9/gone.csv", entry) == (
            False,
            "remote-unresolved",
        )

    def test_local_entries_are_untouched_by_the_branch(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        snap = snapshot_file_deps({str(f)})
        assert "remote" not in snap[str(f)]
        assert file_dep_is_fresh(str(f), snap[str(f)]) == (True, None)


class TestTrackedThroughTheDecorator:
    """The real path: ``pd.read_csv(url)`` inside ``@cash.cache``.

    pandas is one of the readers the tracker patches, and its handler records
    the first argument *verbatim* - which for a remote read is the URL. That is
    what made the dependency vanish before, and what makes it trackable now.

    The server's GET counter doubles as the recompute counter: the cached
    function's only observable is the object it downloads, so a GET means the
    body ran. That keeps mutable state out of the function's closure, which
    cash would (correctly) treat as a changing input.
    """

    def test_a_changed_object_invalidates_the_cached_function(self, origin):
        pd = pytest.importorskip("pandas")
        cash = Cash(backend=InMemoryBackend())

        @cash.cache
        def load(url):
            return pd.read_csv(url)

        assert load(origin.url).iloc[0]["b"] == 2
        assert origin.gets == 1

        assert load(origin.url).iloc[0]["b"] == 2
        assert origin.gets == 1, "unchanged object - must hit without downloading"

        origin.body = b"a,b\n3,4\n"
        origin.etag = '"v2"'
        assert load(origin.url).iloc[0]["b"] == 4, "changed object - must recompute"
        assert origin.gets == 2

    def test_the_url_is_recorded_as_a_remote_dependency(self, origin):
        pd = pytest.importorskip("pandas")
        cash = Cash(backend=InMemoryBackend())

        @cash.cache
        def load(url):
            return pd.read_csv(url)

        load(origin.url)
        # Non-vacuity guard for the test above: prove the dependency is on the
        # entry, so a later "it hit" cannot be mistaken for "nothing was tracked".
        recorded = [
            (path, entry)
            for path, entry in _stored_deps(cash).items()
            if entry.get("remote")
        ]
        assert recorded, "the remote read must be recorded as a dependency"
        assert recorded[0][0] == origin.url
        assert recorded[0][1]["hash"] == 'etag:"v1"'


def _stored_deps(cash: Cash) -> dict:
    """Every auto-tracked dependency recorded across this cache's entries."""
    deps: dict = {}
    for metadata in cash.backend.list_entries():
        deps.update((metadata or {}).get("auto_file_deps") or {})
    return deps
