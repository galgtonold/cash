"""Round-18 warning noise: quiet about harmless code, loud about real problems.

* A static finding about a log line suppressed IMPURE-OBSERVED-EFFECTS for
  the whole function, so a network read in it was never reported (30 starts),
  and the only waiver offered was ``assume_safe=True`` for everything.
* A timestamp passed to a one-line log helper, or into ``json.dumps`` for a
  logger, drew KEY-AMBIENT-READ: "frozen into every later result", when it
  reaches no result at all.
* ``mutable_global`` fired on a setter-configured global that the key folds by
  value on every call -- the one warning class the docs say never to ignore.
* KEY-OPAQUE-CALLABLE named neither the parameter nor what the object wraps,
  so a new hole printed the same text as one already handled.
"""
from __future__ import annotations

import http.server
import json
import logging
import sys
import threading
import time
import urllib.request
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core, pytest.mark.timeout(120)]

_app_log = logging.getLogger("cash_test_app")


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


# -- 1. an observed effect is reported whatever the static pass said ---------

@pytest.fixture(scope="module")
def url():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"7")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def _fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return int(response.read())


def _fetch_audited(url):
    with urllib.request.urlopen(url, timeout=10) as response:  # @cash:assume-safe
        return int(response.read())


def _marker_then_fetch(url):
    print("marker", file=sys.stderr)
    return _fetch(url)


def _audited_marker_then_audited_fetch(url):
    print("marker", file=sys.stderr)  # @cash:assume-safe
    return _fetch_audited(url)


def test_a_network_read_is_reported_beside_a_static_finding(c, url):
    rec = _call(c.cache(_marker_then_fetch), url)
    assert "IMPURE-SIDE-EFFECTS" in _codes(rec), _codes(rec)
    observed = _message(rec, "IMPURE-OBSERVED-EFFECTS")
    assert "network: socket connect" in observed, _codes(rec)
    assert "test_warning_noise.py:" in observed, "the line that led to it is not named"
    assert "# @cash:assume-safe" in observed, "the line-scoped waiver is not offered"


def test_the_line_waiver_covers_an_observed_effect(c, url):
    rec = _call(c.cache(_audited_marker_then_audited_fetch), url)
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(rec), _message(rec, "IMPURE-OBSERVED-EFFECTS")


def test_an_effect_the_static_warning_named_is_not_repeated(c, tmp_path):
    target = tmp_path / "out.txt"

    def write_it(path):
        with open(path, "w") as f:
            f.write("x")
        return 1

    rec = _call(c.cache(write_it), str(target))
    assert "IMPURE-SIDE-EFFECTS" in _codes(rec)
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(rec), _message(rec, "IMPURE-OBSERVED-EFFECTS")


# -- 2. a value that only reaches a log line is not frozen into anything -----

def _log(message):
    print(message, file=sys.stderr)  # @cash:assume-safe


def _not_a_log(message):
    print(message)  # @cash:assume-safe
    return message


def _times_itself(n):
    t0 = time.perf_counter()
    total = sum(range(n))
    _log(f"took {time.perf_counter() - t0:.4f}s")
    _app_log.info(json.dumps({"ts": time.time(), "n": n}))
    return total


def _keeps_the_stamp(n):
    stamp = _not_a_log(str(time.time()))
    return n, stamp


def test_a_timestamp_for_a_log_helper_is_not_an_ambient_read(c):
    rec = _call(c.cache(_times_itself), 10)
    assert "KEY-AMBIENT-READ" not in _codes(rec), _message(rec, "KEY-AMBIENT-READ")


def test_a_timestamp_that_is_returned_still_is(c):
    rec = _call(c.cache(_keeps_the_stamp), 10)
    assert "time.time()" in _message(rec, "KEY-AMBIENT-READ"), _codes(rec)


# -- 3. a global the key folds by value is not "won't reflect changes" --------

_FLAG = 0.10
_TALLY = {"n": 0}


def _configure(flag):
    global _FLAG
    _FLAG = flag


def _flagged(xs):
    return [x for x in xs if x > _FLAG]


def _counted(xs):
    _TALLY["n"] += 1
    return len(xs)


def test_a_setter_configured_global_is_not_reported(c):
    flagged = c.cache(_flagged)
    rec = _call(flagged, (0.05, 0.2))
    assert "mutable_global" not in _message(rec, "IMPURE-SIDE-EFFECTS"), _message(rec, "IMPURE-SIDE-EFFECTS")
    try:
        _configure(0.01)
        assert flagged((0.05, 0.2)) == [0.05, 0.2], "the warning was right after all"
    finally:
        _configure(0.10)


def test_a_global_the_function_itself_mutates_still_is(c):
    rec = _call(c.cache(_counted), (1, 2))
    assert "mutable_global" in _message(rec, "IMPURE-SIDE-EFFECTS"), _codes(rec)


# -- 5. an opaque callable is named by where it arrived -----------------------

class _Opaque:
    __call__ = staticmethod(abs)


def _applies(score, prep=None):
    return 1


@pytest.fixture
def warned_unhashable():
    saved = set(Cash._WARNED_UNHASHABLE)
    Cash._WARNED_UNHASHABLE.clear()
    yield
    Cash._WARNED_UNHASHABLE.clear()
    Cash._WARNED_UNHASHABLE.update(saved)


def test_an_opaque_callable_names_the_parameter_it_arrived_in(c, warned_unhashable):
    applies = c.cache(_applies, assume_safe=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        applies(_Opaque())
        applies(_Opaque(), prep=_Opaque())
    said = [str(w.message) for w in rec if getattr(w.message, "code", None) == "KEY-OPAQUE-CALLABLE"]
    assert any("`score`" in m for m in said) and any("`prep`" in m for m in said), said
    assert all("every _Opaque in the process" in m for m in said), said
