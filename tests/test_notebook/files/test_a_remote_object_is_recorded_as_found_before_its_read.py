"""A statement records a remote object under the token from before its read.

The store is asked once, before the statement's first read of the object;
that token joins the statement's lineage and its entry. An object replaced
while the statement runs leaves an entry that names the old object, so the
next run finds another token and recomputes. A token asked for after the
read would name the new object next to a value computed from the old one,
and every later run would be served that value.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.remote_source import REMOTE_LEDGER
from tests._cell_driver import run_cash_cell
from tests._local_http import LocalThreadingHTTPServer
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _fresh_ledgers():
    REMOTE_LEDGER.reset()
    yield
    REMOTE_LEDGER.reset()


class _Origin:
    """One object, replaced as its *replace_at*-th download is answered: that
    download is the last of the old object."""

    def __init__(self) -> None:
        self.etag, self.body = '"v1"', b"a,b\n1,2\n"
        self.heads = 0
        self.gets = 0
        self.replace_at: int | None = None
        self.lock = threading.Lock()

    def answer(self, download: bool) -> tuple[str, bytes]:
        with self.lock:
            found = self.etag, self.body
            if download:
                self.gets += 1
                if self.gets == self.replace_at:
                    self.etag, self.body = '"v2"', b"a,b\n100,200\n"
            else:
                self.heads += 1
            return found


class _Handler(BaseHTTPRequestHandler):
    """At the top of the module, so the record of what ran finds it by name."""

    def _headers(self, download: bool) -> bytes:
        etag, body = self.server.state.answer(download)
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return body

    def do_HEAD(self):
        self._headers(False)

    def do_GET(self):
        self.wfile.write(self._headers(True))

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


@pytest.fixture
def processor(tmp_path, mock_shell):
    """A statement processor over a disk cache: the entry's records are read
    back from where a restart would find them."""
    from cash.backends import FileBackend
    from cash.core import Cash
    from cash.notebook.statement import StatementProcessor

    cash = Cash(backend=FileBackend(cache_dir=str(tmp_path)), register_magic=False)
    made = StatementProcessor(mock_shell, cash)
    made.shell.user_ns["pd"] = pd
    made.stored_tokens = lambda url: _stored_tokens(cash.backend, url)
    return made


def _stored_tokens(backend, url: str) -> list[str | None]:
    """The token each stored entry that read *url* holds for it."""
    tokens = []
    for metadata in backend.list_entries():
        for field in ("file_dependencies", "auto_file_deps"):
            entry = ((metadata or {}).get(field) or {}).get(url)
            if entry is not None:
                tokens.append(entry.get("hash"))
    return tokens


def test_a_statement_that_misses_asks_the_store_once(processor, origin):
    first = processor.process_statement(f'df = pd.read_csv("{origin.url}")')
    assert first["status"] == CacheStatus.COMPUTED
    assert origin.gets == 1
    assert origin.heads == 1, "asked once, before the read: not again for the lineage or the entry"


def test_replaced_while_the_statement_runs_the_next_run_recomputes(processor, origin, recwarn):
    code = f'df = pd.read_csv("{origin.url}")'
    origin.replace_at = 1
    processor.process_statement(code)
    assert int(processor.shell.user_ns["df"]["a"].sum()) == 1, "the statement read the object before it was replaced"
    assert not [w for w in recwarn if "STORE-INPUT-CHANGED" in str(w.message)]
    assert processor.stored_tokens(origin.url) == ['etag:"v1"'], (
        "the entry must name the object the value was computed from, never the one found afterwards"
    )

    second = processor.process_statement(code)
    assert second["status"] == CacheStatus.COMPUTED
    assert int(processor.shell.user_ns["df"]["a"].sum()) == 100
    third = processor.process_statement(code)
    assert third["status"] != CacheStatus.COMPUTED


def test_the_statement_s_lineage_names_the_object_as_read(processor, origin):
    """What is built on the value is keyed on the old object too: with the
    new object's token in the lineage, the run that reads the new object
    would find the statements below already stored, with old values."""
    code = f'df = pd.read_csv("{origin.url}")'
    origin.replace_at = 1
    processor.process_statement(code)
    replaced_during = processor.tracking_state.variable_lineage["df"]
    processor.process_statement(code)  # reads the new object
    assert processor.tracking_state.variable_lineage["df"] != replaced_during


def test_a_loop_header_s_read_is_recorded_as_found_when_the_header_was_evaluated(processor, origin):
    """A loop run as one unit whose header cash evaluated first is handed
    what that evaluation read, with the token from before it: the unit
    records the object under that token and does not ask the store."""
    processor.shell.user_ns.update(rows=[], time=time)
    loop = (
        f'for row in pd.read_csv("{origin.url}").itertuples():\n'
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    rows.append(row.a)"
    )
    header_value = list(pd.read_csv(origin.url).itertuples())
    origin.etag, origin.body = '"v2"', b"a,b\n100,200\n"  # replaced once the header was read
    origin.heads = 0

    result = processor.process_statement(
        loop, loop_iterable=header_value, header_remote={origin.url: 'etag:"v1"'}, force_outputs={"rows", "row"}
    )
    assert result["status"] == CacheStatus.COMPUTED
    assert processor.shell.user_ns["rows"] == [1]
    assert origin.heads == 0
    # The records its entry, or one built on it, stores.
    records = processor.tracking_state.records_as_run.get(result["cache_key"]).snapshot()
    assert records == {origin.url: {"remote": True, "hash": 'etag:"v1"'}}
    assert origin.heads == 0


def test_a_call_of_a_notebook_function_asks_once_and_its_entry_holds_that_token(
    cash_magics, cash_instance, mock_shell, origin
):
    mock_shell.user_ns.update(pd=pd, time=time)
    run_cash_cell(
        cash_magics,
        f"def load(u):\n    d = pd.read_csv(u)\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return d",
    )
    origin.replace_at = 1
    run_cash_cell(cash_magics, f"df = load({origin.url!r})")
    assert int(mock_shell.user_ns["df"]["a"].sum()) == 1
    assert origin.heads == 1
    tokens = _stored_tokens(cash_instance.backend, origin.url)
    assert tokens and set(tokens) == {'etag:"v1"'}, tokens

    run_cash_cell(cash_magics, f"df = load({origin.url!r})")
    assert int(mock_shell.user_ns["df"]["a"].sum()) == 100
