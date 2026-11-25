"""A program the user's code starts, or a request it sends that is not a
fetch, is seen as the block runs.

A statement's text names ``subprocess.run`` or ``requests.post`` in the
usual spellings. It cannot name ``getattr(subprocess, 'call')(cmd)``, a
function of the user's that starts the program, or a request object built
in an earlier cell. Those were stored, and a hit did not start the program
or send the request. The write observer now records them
(``WrittenPaths.effect``), as it records a file the block writes, and the
statement and the call cache refuse to store on it. What a library starts
on its own way to an answer is that library's business, as before, and so
is a fetch.
"""

from __future__ import annotations

import http.server
import linecache
import platform
import subprocess
import sys
import threading

import pytest

from cash.notebook.write_observer import observe_writes

QUIET = [sys.executable, "-c", "pass"]

#: The user's code, as a cell: a pseudo-filename is the user's.
CELL = """
import os, subprocess, urllib.request

def build(cmd):
    return subprocess.call(cmd)

def waived(cmd):
    return subprocess.call(cmd)  # @cash:assume-safe

def send(opener, url, method):
    return opener.open(urllib.request.Request(url, data=b'x', method=method)).status
"""


@pytest.fixture
def cell():
    """The functions of `CELL`, their source where IPython keeps a cell's."""
    name = "<ipython-input-1-effects>"
    linecache.cache[name] = (len(CELL), None, CELL.splitlines(keepends=True), name)
    namespace: dict = {}
    exec(compile(CELL, name, "exec"), namespace)
    yield namespace
    linecache.cache.pop(name, None)


def _observed(run) -> tuple[str, str] | None:
    with observe_writes() as written:
        run()
    return written.effect


@pytest.mark.parametrize(
    ("call", "event"),
    [
        ("build(cmd)", "subprocess.Popen"),
        ("getattr(subprocess, 'check_call')(cmd)", "subprocess.Popen"),
        ("[f(cmd) for f in (subprocess.call,)]", "subprocess.Popen"),
        ("os.system('exit 0')", "os.system"),
        ("os.popen('exit 0').close()", "subprocess.Popen"),
    ],
)
def test_a_program_the_users_code_starts_is_recorded(cell, call, event):
    code = compile(call, "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(code, {**cell, "cmd": QUIET})) == ("system", f"started another program ({event})")


def test_nothing_is_recorded_when_nothing_is_started(cell):
    code = compile("build.__name__ + str(len(cmd))", "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(code, {**cell, "cmd": QUIET})) is None


def test_a_program_started_on_a_waived_line_is_not_recorded(cell):
    code = compile("waived(cmd)", "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(code, {**cell, "cmd": QUIET})) is None


def test_a_program_a_library_starts_is_not_the_users(cell):
    """``platform`` asks ``uname`` for the processor through ``subprocess``:
    the code that called the process API is the library's, and a statement
    calling the library is judged as before."""
    code = compile("ask()", "<ipython-input-2-effects>", "eval")
    ask = platform._Processor.from_subprocess  # what ``platform.processor()`` runs once a process
    with observe_writes() as written:
        eval(code, {"ask": ask})
    assert written.unseen == "subprocess.Popen"  # the control: a program was started
    assert written.effect is None


def test_cash_starting_a_program_itself_is_not_the_users():
    with observe_writes() as written:
        subprocess.call(QUIET)  # this file is a test of cash's, run from pytest: no cell above it
    assert written.unseen == "subprocess.Popen"


@pytest.fixture
def server():
    seen: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            seen.append(f"{self.command} {self.path}")
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _answer

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def opener():
    """An opener that ignores the machine's proxy settings: the server is local."""
    import urllib.request

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_request_that_is_not_a_fetch_is_recorded(cell, server, opener, method):
    url, seen = server
    code = compile("send(opener, url, method)", "<ipython-input-2-effects>", "eval")
    effect = _observed(lambda: eval(code, {**cell, "opener": opener, "url": url + "/hook", "method": method}))
    assert effect == ("network", f"sent a {method} request to 127.0.0.1")
    assert seen == [f"{method} /hook"]


def test_a_fetch_is_not_recorded(cell, server, opener):
    url, seen = server
    code = compile("opener.open(url).status", "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(code, {"opener": opener, "url": url + "/page"})) is None
    assert seen == ["GET /page"]


def test_a_post_through_requests_is_recorded_and_a_get_is_not(cell, server):
    requests = pytest.importorskip("requests")
    url, seen = server
    session = requests.Session()
    session.trust_env = False
    post = compile("session.request('POST', url).status_code", "<ipython-input-2-effects>", "eval")
    get = compile("session.get(url).status_code", "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(get, {"session": session, "url": url + "/page"})) is None
    assert _observed(lambda: eval(post, {"session": session, "url": url + "/hook"})) == (
        "network",
        "sent a POST request to 127.0.0.1",
    )
    assert seen == ["GET /page", "POST /hook"]


def test_a_post_a_library_sends_on_its_way_is_not_the_users(server, opener):
    """A client library that posts for a token before it fetches: the code
    that called the HTTP client is the library's."""
    url, seen = server
    library: dict = {"__name__": "somelib.client"}
    exec(
        compile(
            "import urllib.request\n"
            "def read_table(opener, url):\n"
            "    opener.open(urllib.request.Request(url + '/token', data=b'x'))\n"
            "    return opener.open(url + '/rows').status\n",
            platform.__file__,  # a file that is not the user's
            "exec",
        ),
        library,
    )
    code = compile("read_table(opener, url)", "<ipython-input-2-effects>", "eval")
    assert _observed(lambda: eval(code, {**library, "opener": opener, "url": url})) is None
    assert seen == ["POST /token", "GET /rows"]


def test_the_reason_reads_like_the_one_for_the_call_spelled_out():
    from cash.notebook.write_observer import OBSERVED_EFFECT_REASON

    reason = OBSERVED_EFFECT_REASON.format(what="started another program (os.system)", label="system")
    assert reason.startswith("Side effect: started another program (os.system) as it ran (system)")
    assert "# @cash:assume-safe" in reason
