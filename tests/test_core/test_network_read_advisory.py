"""A decorated function that fetches from a server or queries a database gets
a TTL advisory.

``requests.get(url)`` in a ``@cash.cache`` body used to be reported as a side
effect (IMPURE-SIDE-EFFECTS, "known I/O"), with advice to audit the line for
writes. A GET writes nothing: the hazard is the other way round -- what the
server returns is an input the key cannot see, so the first answer is served
until something changes the key. That is the question ``ttl=`` answers, so the
advisory (KEY-NETWORK-READ) names it, and setting a ``ttl=`` silences it.

A write to the network (a POST) is still a side effect and still reported as
one. A database is judged the same way: a literal SELECT is a read, anything
else sent to ``execute`` is a write.
"""

from __future__ import annotations

import http.server
import sqlite3
import threading
import urllib.request
import warnings

import pytest

from cash import Cash
from cash.analysis.purity_analyzer import ISSUE_NETWORK_READ, PurityAnalyzer
from cash.exceptions import CashImpureFunctionError

pytestmark = [pytest.mark.timeout(120)]


def _codes(record):
    return [getattr(w.message, "code", None) for w in record]


def _message(record, code):
    return next((str(w.message) for w in record if getattr(w.message, "code", None) == code), "")


def _call(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args)
    return rec


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.fixture(scope="module")
def url():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self._answer()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._answer()

        def _answer(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"7")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return int(response.read())


def fetch_audited(url):
    with urllib.request.urlopen(url, timeout=10) as response:  # @cash:assume-safe
        return int(response.read())


def post(url):
    with urllib.request.urlopen(url, b"x=1", timeout=10) as response:
        return int(response.read())


def test_the_analyzer_names_a_read_as_a_read():
    issues = PurityAnalyzer().analyze(fetch).issues
    assert [i.kind for i in issues] == [ISSUE_NETWORK_READ]
    assert "what the server returns is not in the cache key" in issues[0].description


def test_a_network_read_gets_the_ttl_advisory(c, url):
    rec = _call(c.cache(fetch), url)
    assert _codes(rec) == ["KEY-NETWORK-READ"], _codes(rec)
    message = _message(rec, "KEY-NETWORK-READ")
    assert "urllib.request.urlopen()" in message
    assert "ttl=" in message


def test_ttl_silences_it(c, url):
    rec = _call(c.cache(fetch, ttl=3600), url)
    assert _codes(rec) == [], _codes(rec)


def test_the_line_waiver_silences_it(c, url):
    rec = _call(c.cache(fetch_audited), url)
    assert _codes(rec) == [], _codes(rec)


def test_the_observer_does_not_repeat_the_connection(c, url):
    """The first call opens a socket; the advisory already named that read."""
    rec = _call(c.cache(fetch), url)
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(rec), _message(rec, "IMPURE-OBSERVED-EFFECTS")


def test_nor_when_ttl_silenced_the_advisory(c, url):
    rec = _call(c.cache(fetch, ttl=3600), url)
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(rec), _message(rec, "IMPURE-OBSERVED-EFFECTS")


def test_strict_mode_raises_on_it(c, url):
    with pytest.raises(CashImpureFunctionError) as exc:
        c.cache(fetch, strict=True)(url)
    assert "urlopen" in str(exc.value)


def test_strict_mode_accepts_it_with_a_ttl(c, url):
    assert c.cache(fetch, strict=True, ttl=3600)(url) == 7


def test_a_post_is_still_a_side_effect(c, url):
    rec = _call(c.cache(post), url)
    assert "IMPURE-SIDE-EFFECTS" in _codes(rec), _codes(rec)
    assert "KEY-NETWORK-READ" not in _codes(rec)


def test_ttl_does_not_silence_a_post(c, url):
    rec = _call(c.cache(post, ttl=3600), url)
    assert "IMPURE-SIDE-EFFECTS" in _codes(rec), _codes(rec)


def _connect():
    """A connection made elsewhere: a pool, a client library, a server."""
    return sqlite3.connect(":memory:")


def query_elsewhere():
    return _connect().execute("SELECT 1").fetchone()[0]


def count_rows(path):
    with sqlite3.connect(path) as con:
        return con.execute("SELECT count(*) FROM t").fetchone()[0]


def add_row(path):
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO t VALUES (1)")
    return 1


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "rows.db")
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE t (x)")
    return path


def test_a_query_gets_the_ttl_advisory(c):
    rec = _call(c.cache(query_elsewhere))
    assert _codes(rec) == ["KEY-NETWORK-READ"], _codes(rec)
    message = _message(rec, "KEY-NETWORK-READ")
    assert "execute() - what the database returns is not in the cache key" in message


def test_ttl_silences_a_query(c):
    rec = _call(c.cache(query_elsewhere, ttl=60))
    assert _codes(rec) == [], _codes(rec)


def test_a_sqlite_file_the_body_opens_is_already_in_the_key(c, db):
    """`sqlite3.connect(path)` is tracked as a read of that file, so the
    answer is keyed and there is nothing to advise."""
    cached = c.cache(count_rows)
    rec = _call(cached, db)
    assert _codes(rec) == [], _codes(rec)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO t VALUES (1)")
    assert cached(db) == 1


def test_a_database_write_is_still_a_side_effect(c, db):
    rec = _call(c.cache(add_row, ttl=60), db)
    assert "IMPURE-SIDE-EFFECTS" in _codes(rec), _codes(rec)
    assert "KEY-NETWORK-READ" not in _codes(rec)
