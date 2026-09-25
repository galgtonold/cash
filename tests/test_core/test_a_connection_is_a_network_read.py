"""A connection the analyzer could not name is a network read, not an effect.

``requests.Session().get(url)``, ``from requests import get``, a client
library: the static pass names only module-qualified calls, so these reads
were seen only as the connection they open, and reported as
IMPURE-OBSERVED-EFFECTS -- "a hit will not repeat this" -- with advice to
waive it. ``ttl=`` did not silence that, and ``strict=True`` did not raise, so
a CI build with ``strict=True`` passed a function that served yesterday's
rate. The hazard is the one KEY-NETWORK-READ names, and so is the knob.

A remote file a reader opened by URL is a tracked read (its ETag is checked on
every hit), so its fetch is not reported at all.
"""

from __future__ import annotations

import http.server
import sys
import threading
import urllib.request
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpureFunctionError
from cash.tracking.reader_patches import file_registry

pytestmark = [pytest.mark.timeout(120)]


@pytest.fixture(scope="module")
def url():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"7")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _codes(record):
    return [getattr(w.message, "code", None) for w in record]


def _message(record, code):
    return next((str(w.message) for w in record if getattr(w.message, "code", None) == code), "")


# Through an opener object, so no name reaches the analyzer -- the shape of
# `requests.Session().get`.
def rate(url):
    with urllib.request.build_opener().open(url, timeout=10) as response:
        return int(response.read())


def test_an_unnamed_fetch_gets_the_ttl_advisory(c, url):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert c.cache(rate)(url) == 7
    assert _codes(rec) == ["KEY-NETWORK-READ"], _codes(rec)
    message = _message(rec, "KEY-NETWORK-READ")
    assert "socket connect to 127.0.0.1" in message
    assert "test_a_connection_is_a_network_read.py:" in message, "the line that led to it is not named"
    assert "ttl=" in message


def test_ttl_silences_it(c, url):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert c.cache(rate, ttl=3600)(url) == 7
    assert _codes(rec) == [], _codes(rec)


def test_strict_raises_and_stores_nothing(c, url):
    fn = c.cache(rate, strict=True)
    with pytest.raises(CashImpureFunctionError, match="connected to a server"):
        fn(url)
    with pytest.raises(CashImpureFunctionError):
        fn(url)  # nothing was stored, so the second call is not a silent hit


def test_strict_accepts_it_with_a_ttl(c, url):
    assert c.cache(rate, strict=True, ttl=3600)(url) == 7


def _fetch_by_url(target):
    with urllib.request.build_opener().open(target, timeout=10) as response:
        return response.read()


def test_a_tracked_remote_read_is_not_reported(c, url, monkeypatch):
    """A reader handed a URL records it as a remote dependency before the
    fetch; the connection it then opens is that read, not a hidden one."""
    registry = file_registry()
    module = type(pytest)("cash_test_url_reader")
    module.read_url = _fetch_by_url
    monkeypatch.setitem(sys.modules, "cash_test_url_reader", module)
    registry.register("cash_test_url_reader", "read_url", registry._create_path_arg_handler)
    try:

        def load(u):
            return int(module.read_url(u))

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert c.cache(load)(url) == 7
        assert "KEY-NETWORK-READ" not in _codes(rec), _message(rec, "KEY-NETWORK-READ")
        assert "IMPURE-OBSERVED-EFFECTS" not in _codes(rec), _message(rec, "IMPURE-OBSERVED-EFFECTS")
    finally:
        registry.handlers.pop("cash_test_url_reader", None)
