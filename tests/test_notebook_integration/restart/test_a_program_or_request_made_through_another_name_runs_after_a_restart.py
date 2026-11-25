"""A program started or a request sent runs on every Run All, however the
call is spelled.

``subprocess.run(cmd)`` and ``requests.post(url)`` are never stored: a hit
would not start the program or send the request. The same call through an
imported name or an alias (``from subprocess import run``, ``import
subprocess as sp``, ``from os import system``, ``from requests import
post``) and ``urlopen(Request(url, data=..., method='POST'))`` were stored,
and Restart and Run All restored them: the command did not run and the POST
was not sent. Where the text cannot name the call at all (``getattr(sp,
'check_call')``, a session's ``request('POST', url)`` or ``delete(url)``, a
request object built in a cell above, a function of the user's), the
program or the request is seen as the statement runs.

A fetch (``requests.get``), and a program a library starts on its own way
to an answer, are stored as before.
"""

import collections
import http.server
import threading
import time

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

#: Long enough for the statement to be written to disk.
SLOW = 0.4


@pytest.fixture
def server(monkeypatch):
    """A local server that counts the requests to each path; its address."""
    hits = collections.Counter()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            hits[self.path] += 1
            time.sleep(SLOW)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        do_GET = do_POST = do_PUT = do_DELETE = _answer

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # The kernel is started after this, with the test's environment.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield f"http://127.0.0.1:{httpd.server_address[1]}", hits
    httpd.shutdown()
    httpd.server_close()


SETUP = (
    "import subprocess, requests, urllib.request\n"
    "import subprocess as sp\n"
    "from subprocess import run\n"
    "from os import system\n"
    "from requests import post\n"
    "from urllib.request import urlopen, Request\n"
    "U = '{url}'\n"
    "def send(path):\n"
    "    with requests.Session() as session:\n"
    "        return session.request('POST', U + path).status_code\n"
    "def deploy(name):\n"
    "    return sp.call(['sh', '-c', 'sleep {slow}; echo x >> ' + name])\n"
)


def _command(name: str) -> str:
    """A shell command that takes a moment and appends one line to *name*."""
    return f"sleep {SLOW}; echo x >> {name}"


def _run_all_twice_across_a_restart(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()


def _lines(nb_runner, name: str) -> int:
    path = nb_runner.work_dir / name
    return len(path.read_text().splitlines()) if path.exists() else 0


#: label -> (the statement, the file or the server path that counts its runs)
SPELLED = {
    "subprocess.run": (f"subprocess.run(['sh', '-c', '{_command('c0.txt')}'])", "c0.txt"),
    "from subprocess import run": (f"run(['sh', '-c', '{_command('c1.txt')}'])", "c1.txt"),
    "import subprocess as sp": (f"sp.check_call(['sh', '-c', '{_command('c2.txt')}'])", "c2.txt"),
    "its output kept": (f"out = sp.check_output(['sh', '-c', '{_command('c3.txt')}; echo done'])", "c3.txt"),
    "from os import system": (f"system('{_command('c4.txt')}')", "c4.txt"),
    "requests.post": ("r0 = requests.post(U + '/control').status_code", "/control"),
    "from requests import post": ("r1 = post(U + '/from_import', data=b'x').status_code", "/from_import"),
    "urlopen of a POST request": (
        "r2 = urllib.request.urlopen(urllib.request.Request(U + '/urlopen', data=b'x', method='POST')).status",
        "/urlopen",
    ),
    "urlopen of a request with data": ("r3 = urlopen(Request(U + '/data', b'x')).status", "/data"),
}

SEEN = {
    "the call looked up by a string": (f"getattr(sp, 'check_call')(['sh', '-c', '{_command('s0.txt')}'])", "s0.txt"),
    "a function of the user's": ("code = deploy('s1.txt')", "s1.txt"),
    "a session's request": ("r4 = requests.Session().request('POST', U + '/session').status_code", "/session"),
    "a session's delete": ("r5 = requests.Session().delete(U + '/delete').status_code", "/delete"),
    "a request in a function of the user's": ("r7 = send('/helper')", "/helper"),
    "a request built in a cell above": ("r6 = urlopen(req).status", "/built"),
}


def _count(nb_runner, hits, where: str) -> int:
    return hits[where] if where.startswith("/") else _lines(nb_runner, where)


def test_a_program_or_a_post_through_an_imported_name_runs_on_each_run_all(nb_runner, server):
    url, hits = server
    cells = ["import cash\n%load_ext cash\n%cash_badge print\n%cash_on", SETUP.format(url=url, slow=SLOW)] + [
        stmt for stmt, _ in SPELLED.values()
    ]
    _run_all_twice_across_a_restart(nb_runner, cells)

    ran = {label: _count(nb_runner, hits, where) for label, (_, where) in SPELLED.items()}
    assert ran == dict.fromkeys(SPELLED, 2)
    for number, label in enumerate(SPELLED, start=3):
        out = nb_runner.get_output(number)
        assert "Side effect" in out and not shows_cached(out), (label, out)


def test_a_program_or_a_post_the_text_does_not_name_is_seen_as_it_runs(nb_runner, server):
    url, hits = server
    cells = [
        "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
        SETUP.format(url=url, slow=SLOW),
        "req = Request(U + '/built', data=b'x', method='POST')",
    ] + [stmt for stmt, _ in SEEN.values()]
    _run_all_twice_across_a_restart(nb_runner, cells)

    ran = {label: _count(nb_runner, hits, where) for label, (_, where) in SEEN.items()}
    assert ran == dict.fromkeys(SEEN, 2)
    for number, label in enumerate(SEEN, start=4):
        out = nb_runner.get_output(number)
        assert "as it ran" in out and not shows_cached(out), (label, out)


def test_a_fetch_and_a_program_a_library_starts_are_still_served(nb_runner, server):
    url, hits = server
    cells = [
        "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
        SETUP.format(url=url, slow=SLOW) + "import platform, time",
        "page = requests.get(U + '/get').status_code",
        "page2 = urlopen(U + '/urlopen_get').status",
        # ``platform.processor()`` runs ``uname -p`` the first time a process asks.
        "cpu = [time.sleep(0.4), platform.processor()][1]",
        f"getattr(sp, 'check_call')(['sh', '-c', '{_command('waived.txt')}'])  # @cash:assume-safe",
        "print('done', page, page2)",
    ]
    _run_all_twice_across_a_restart(nb_runner, cells)

    assert "done 200 200" in nb_runner.get_output(7)
    assert {path: hits[path] for path in ("/get", "/urlopen_get")} == {"/get": 1, "/urlopen_get": 1}
    assert _lines(nb_runner, "waived.txt") == 1
    # Served whole, or run around a call the cache holds.
    for number in (3, 4, 5, 6):
        out = nb_runner.get_output(number)
        assert "NOT CACHED" not in out and (shows_cached(out) or "1/1 hit" in out), out
