"""A notebook statement reading s3:// must invalidate when the object changes.

The decorator path tracks remote reads by recording the store's validator in
the entry's metadata and re-checking it on a hit (CAS-236). The statement path
works differently: ``compute_file_hash_component`` folds file state directly
into the statement's *lineage*, so the natural home for a remote token is the
same component. A changed ETag then yields a different key, which cannot go
stale by construction — there is nothing to re-validate.

Remote URLs deliberately do NOT enter ``executed_file_deps``. That set is
stat'ed and getmtime'd by its consumers, so a URL there is silently dropped at
best; the key component alone is sufficient. See CAS-237.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.remote_source import _reset_remote_warnings


@pytest.fixture(autouse=True)
def _fresh_ledgers():
    _reset_remote_warnings()
    yield
    _reset_remote_warnings()


class _Origin:
    def __init__(self):
        self.etag = '"v1"'
        self.body = b"a,b\n1,2\n"


@pytest.fixture
def origin():
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
            # Needed as well as HEAD: the freshness token comes from HEAD, but
            # the statement under test actually reads the object.
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


class TestRemoteInTheKeyComponent:
    def test_a_remote_read_contributes_a_component(self, origin):
        component = compute_file_hash_component(set(), {origin.url})
        assert component, "a remote read must change the statement's lineage"

    def test_the_component_moves_when_the_object_moves(self, origin):
        before = compute_file_hash_component(set(), {origin.url})
        origin.etag = '"v2"'
        after = compute_file_hash_component(set(), {origin.url})
        assert before != after, (
            "a changed object must yield a different key, or the statement "
            "serves a stale value forever - the CAS-237 bug"
        )

    def test_an_unchanged_object_holds_the_component(self, origin):
        first = compute_file_hash_component(set(), {origin.url})
        assert compute_file_hash_component(set(), {origin.url}) == first

    def test_no_remote_reads_is_unchanged_from_before(self, tmp_path):
        """The local-file path must be byte-identical when no URL is involved,
        or every existing cached statement's key shifts and the whole cache is
        invalidated on upgrade."""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        assert (
            compute_file_hash_component({str(f)})
            == compute_file_hash_component({str(f)}, set())
            == compute_file_hash_component({str(f)}, None)
        )

    def test_local_and_remote_compose(self, origin, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        local_only = compute_file_hash_component({str(f)})
        both = compute_file_hash_component({str(f)}, {origin.url})
        assert both and both != local_only, (
            "a statement reading both a file and an object must depend on both"
        )


class TestThroughARealStatement:
    """The CAS-237 claim end to end: a notebook statement reading a URL.

    Before this, ``FileAccessTracker`` recorded the URL on its remote channel
    but the statement path never read that channel, so the statement cached and
    then hit forever no matter what happened to the object.
    """

    @pytest.fixture
    def processor(self, tmp_path):
        from traitlets.config.configurable import Configurable

        from cash.backends import FileBackend
        from cash.core import Cash
        from cash.notebook.statement import StatementProcessor

        class MockShell(Configurable):
            def __init__(self):
                super().__init__()
                self.user_ns: dict = {}
                self.input_transformers_cleanup: list = []
                self.ast_transformers: list = []
                self.user_global_ns = self.user_ns

        backend = FileBackend(cache_dir=str(tmp_path))
        cash_instance = Cash(backend=backend, register_magic=False)
        proc = StatementProcessor(MockShell(), cash_instance)
        proc.debug = False
        return proc

    @pytest.mark.xfail(
        reason=(
            "CAS-237 is NOT finished. A statement's cache_key is computed "
            "BEFORE it executes, so a token discovered DURING execution cannot "
            "be in it. compute_file_hash_component feeds lineage, which keys "
            "DOWNSTREAM statements — so a consumer of `df` does invalidate, but "
            "the reading statement itself still hits. Closing this needs the "
            "decorator's shape: record the token with the entry and re-check it "
            "on lookup. The lineage half (below, and the component tests above) "
            "is done and correct as far as it goes."
        ),
        strict=True,
    )
    def test_a_changed_object_gives_the_statement_a_new_key(self, processor, origin):
        pytest.importorskip("pandas")
        code = f'df = pd.read_csv("{origin.url}")'

        processor.shell.user_ns["pd"] = __import__("pandas")
        first = processor.process_statement(code)

        origin.etag = '"v2"'
        second = processor.process_statement(code)

        assert first.get("cache_key") != second.get("cache_key"), (
            "the statement's key must move with the object, or an edited "
            "upstream dataset is served stale forever"
        )

    def test_an_unchanged_object_keeps_the_key(self, processor, origin):
        pytest.importorskip("pandas")
        code = f'df = pd.read_csv("{origin.url}")'
        processor.shell.user_ns["pd"] = __import__("pandas")

        first = processor.process_statement(code)
        second = processor.process_statement(code)
        assert first.get("cache_key") == second.get("cache_key"), (
            "an unchanged object must not churn the key, or nothing ever hits"
        )
