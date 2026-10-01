"""The notebook is found from the server files in the Jupyter runtime dir.

A kernel registered from its own venv or conda env has ``ipykernel`` (and so
``jupyter_core``) but not ``jupyter_server`` or ``notebook``, which live in the
server's environment. Asking those packages for the running servers found
nothing there, so discovery returned None and upstream tracking was off for
the whole session. Every running server leaves a ``jpserver-*.json`` (or
``nbserver-*.json``) in the runtime dir, and reading those files needs nothing
the kernel lacks.

These tests run ``get_notebook_path`` against a real runtime dir
(``JUPYTER_RUNTIME_DIR``) and a real HTTP server on localhost that answers
``/api/sessions`` like Jupyter does, with the server packages made
unimportable.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil
import pytest

from cash.notebook import server_discovery as sd

KERNEL_ID = "5f1c0d8e-2b7a-4c3e-9d61-0a8e4f7b2c19"
NB_PATH = "reports/analysis.ipynb"


class _StubServer:
    """Answers ``GET /api/sessions`` the way a Jupyter server does.

    With *token* set, a request without ``Authorization: token <token>`` gets
    403, as on a real server. Each request's Authorization header (or None) is
    recorded in ``auth_headers``.
    """

    def __init__(self, sessions: list[dict], token: str | None = None) -> None:
        self.auth_headers: list[str | None] = []
        self.paths: list[str] = []
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                auth = self.headers.get("Authorization")
                stub.auth_headers.append(auth)
                # Through a proxy the path is the whole URL.
                stub.paths.append(self.path)
                if not self.path.split("?")[0].endswith("/api/sessions"):
                    self.send_error(404)
                    return
                if token is not None and auth != f"token {token}":
                    self.send_error(403)
                    return
                body = json.dumps(sessions).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _sessions(kernel_id: str = KERNEL_ID) -> list[dict]:
    other = {"id": "s0", "path": "other.ipynb", "type": "notebook", "kernel": {"id": "not-this-kernel"}}
    ours = {"id": "s1", "path": NB_PATH, "type": "notebook", "kernel": {"id": kernel_id}}
    return [other, ours]


@pytest.fixture
def serve():
    """Start stub servers on demand; stop them all at teardown."""
    started: list[_StubServer] = []

    def _start(sessions: list[dict] | None = None, token: str | None = None) -> _StubServer:
        stub = _StubServer(_sessions() if sessions is None else sessions, token)
        started.append(stub)
        return stub

    yield _start
    for stub in started:
        stub.close()


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    """A kernel env with no server packages, pointed at an empty runtime dir."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(runtime))
    # The kernel runs in its own env: the server packages live elsewhere.
    for name in ("jupyter_server", "jupyter_server.serverapp", "notebook", "notebook.notebookapp"):
        monkeypatch.setitem(sys.modules, name, None)
    kernel = types.ModuleType("ipykernel")
    kernel.get_connection_file = lambda: str(runtime / f"kernel-{KERNEL_ID}.json")
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)
    monkeypatch.delenv("JUPYTERHUB_API_TOKEN", raising=False)
    sd.invalidate_notebook_path_cache()
    yield runtime
    sd.invalidate_notebook_path_cache()


def _write_server_file(runtime, name, *, url, root, token="", pid=None, age=0.0, root_key="root_dir"):
    """Write a server info file as jupyter_server does, *age* seconds old."""
    info = {
        "base_url": "/",
        "hostname": "127.0.0.1",
        "password": False,
        "pid": os.getpid() if pid is None else pid,
        "secure": False,
        "sock": "",
        "token": token,
        "url": url,
        root_key: str(root),
    }
    path = runtime / name
    path.write_text(json.dumps(info), encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def _closed_port_url() -> str:
    """A localhost URL nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/"


@pytest.mark.parametrize(
    "file_name,root_key",
    [
        ("jpserver-4242.json", "root_dir"),  # jupyter_server: JupyterLab, Notebook 7
        ("nbserver-4242.json", "notebook_dir"),  # the classic notebook server
    ],
)
def test_found_from_the_runtime_dir_without_server_packages(runtime_dir, serve, tmp_path, file_name, root_key):
    stub = serve()
    root = tmp_path / "work"
    _write_server_file(runtime_dir, file_name, url=stub.url, root=root, token="tok", root_key=root_key)

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.auth_headers == ["token tok"]


def test_jupyterhub_token_is_sent_when_the_entry_has_none(runtime_dir, serve, tmp_path, monkeypatch):
    stub = serve(token="hub-secret")
    root = tmp_path / "home"
    _write_server_file(runtime_dir, "jpserver-4242.json", url=stub.url, root=root, token="")
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "hub-secret")

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.auth_headers == ["token hub-secret"]


def test_no_token_header_without_a_token_anywhere(runtime_dir, serve, tmp_path):
    """The control arm: an open server gets no Authorization header at all."""
    stub = serve()
    root = tmp_path / "work"
    _write_server_file(runtime_dir, "jpserver-4242.json", url=stub.url, root=root, token="")

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.auth_headers == [None]


def test_the_entry_token_wins_over_the_hub_token(runtime_dir, serve, tmp_path, monkeypatch):
    stub = serve(token="own")
    root = tmp_path / "work"
    _write_server_file(runtime_dir, "jpserver-4242.json", url=stub.url, root=root, token="own")
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "hub-secret")

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.auth_headers == ["token own"]


@pytest.mark.parametrize("stale", ["refuses", "never answers"])
def test_a_stale_entry_before_the_live_one_costs_at_most_one_timeout(runtime_dir, serve, tmp_path, stale):
    """The newest file is a leftover; the live server's file is older.

    "refuses": nothing listens on the port any more. "never answers": the port
    accepts connections but no reply comes, the case the timeout is for.
    """
    live = serve()
    root = tmp_path / "work"
    _write_server_file(runtime_dir, "jpserver-1111.json", url=live.url, root=root, age=60)

    with socket.socket() as silent:
        if stale == "refuses":
            stale_url = _closed_port_url()
        else:
            silent.bind(("127.0.0.1", 0))
            silent.listen(8)
            stale_url = f"http://127.0.0.1:{silent.getsockname()[1]}/"
        _write_server_file(runtime_dir, "jpserver-2222.json", url=stale_url, root=tmp_path / "stale", age=0)

        started = time.monotonic()
        found = sd.get_notebook_path()
        elapsed = time.monotonic() - started

    assert found == os.path.join(str(root), NB_PATH)
    assert live.auth_headers, "the live server was never asked"
    # One timeout at most (a refused localhost connect can itself take a
    # moment on Windows), plus slack for a loaded machine.
    assert elapsed < sd._SESSIONS_TIMEOUT_S + 1.0, f"lookup took {elapsed:.2f}s"


@pytest.mark.parametrize("where", ["same url", "another server"])
def test_a_dead_servers_newer_file_does_not_shadow_the_live_one(runtime_dir, serve, tmp_path, where):
    """A dead server's file is asked after the live ones, or not at all.

    "same url": the live server now listens on the port the dead one used, and
    that session would be joined to the dead server's root dir. "another
    server": something still answers at the dead entry's address with the same
    kernel id (a forwarded port, say) and the wrong root.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead_pid = proc.pid
    assert not psutil.pid_exists(dead_pid), "precondition: the child process is gone"

    live = serve()
    right = tmp_path / "right"
    wrong = tmp_path / "wrong"
    _write_server_file(runtime_dir, "jpserver-1111.json", url=live.url, root=right, age=60)
    dead_url = live.url if where == "same url" else serve().url
    _write_server_file(runtime_dir, f"jpserver-{dead_pid}.json", url=dead_url, root=wrong, pid=dead_pid, age=0)

    assert sd.get_notebook_path() == os.path.join(str(right), NB_PATH)


def test_empty_or_corrupt_server_files_are_skipped(runtime_dir, serve, tmp_path):
    stub = serve()
    root = tmp_path / "work"
    _write_server_file(runtime_dir, "jpserver-1111.json", url=stub.url, root=root, age=60)
    now = time.time()
    for name, text in [
        ("jpserver-2222.json", ""),  # a server that has not written its file yet
        ("jpserver-3333.json", '{"url": "http://127.0.0.1:1/", "tok'),  # cut off mid-write
        ("nbserver-4444.json", "[]"),  # JSON, but not a server entry
        ("nbserver-5555.json", '{"pid": 5555}'),  # no url to ask
    ]:
        bad = runtime_dir / name
        bad.write_text(text, encoding="utf-8")
        os.utime(bad, (now, now))

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.auth_headers == [None]


def test_a_server_named_by_two_files_is_asked_once(runtime_dir, serve, tmp_path):
    """A leftover file whose pid was reused looks alive and names the same URL;
    asking again only repeats the answer (or the timeout)."""
    stub = serve(sessions=_sessions(kernel_id="another-kernel"))
    _write_server_file(runtime_dir, "jpserver-1111.json", url=stub.url, root=tmp_path / "a", age=60)
    _write_server_file(runtime_dir, "jpserver-2222.json", url=stub.url.rstrip("/"), root=tmp_path / "b", age=0)

    assert sd.get_notebook_path() is None
    assert stub.auth_headers == [None], "the server was not asked exactly once"


def _proxy_env(monkeypatch, proxy_url: str) -> None:
    """Route plain-HTTP requests through *proxy_url*, with no bypass list."""
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, proxy_url)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_local_server_is_reached_past_an_http_proxy(runtime_dir, serve, tmp_path, monkeypatch, host):
    """A proxy cannot reach the kernel's own loopback interface, and urllib
    would send even 127.0.0.1 through ``http_proxy`` unless ``no_proxy`` names
    it. Here the proxy is dead, so a request through it fails."""
    stub = serve()
    root = tmp_path / "work"
    url = stub.url.replace("127.0.0.1", host)
    _write_server_file(runtime_dir, "jpserver-4242.json", url=url, root=root)
    _proxy_env(monkeypatch, _closed_port_url())

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert stub.paths == ["/api/sessions"]


def test_a_server_on_another_host_still_goes_through_the_proxy(runtime_dir, serve, tmp_path, monkeypatch):
    """The control arm: a JupyterHub server on another host may need the
    proxy, so its request goes there. The stub plays the proxy and answers."""
    proxy = serve()
    root = tmp_path / "home"
    remote = "http://jupyterhub.invalid:8000/user/ada/"
    _write_server_file(runtime_dir, "jpserver-4242.json", url=remote, root=root, token="tok")
    _proxy_env(monkeypatch, proxy.url)

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert proxy.paths == [remote + "api/sessions"]


def test_a_proxy_set_after_an_earlier_request_is_used(runtime_dir, serve, tmp_path, monkeypatch):
    """``urllib.request.urlopen`` keeps one opener per process, and its proxy
    settings are the environment at the first ``urlopen``. Any earlier request
    in the kernel (here one made before the proxy is set) must not freeze
    them: the remote server is still asked through the proxy set now."""
    monkeypatch.setattr(urllib.request, "_opener", None)
    for name in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    with urllib.request.urlopen("data:,x") as primed:
        assert primed.read() == b"x"
    assert urllib.request._opener is not None, "the process-wide opener was not built"

    proxy = serve()
    root = tmp_path / "home"
    remote = "http://jupyterhub.invalid:8000/user/ada/"
    _write_server_file(runtime_dir, "jpserver-4242.json", url=remote, root=root, token="tok")
    _proxy_env(monkeypatch, proxy.url)

    assert sd.get_notebook_path() == os.path.join(str(root), NB_PATH)
    assert proxy.paths == [remote + "api/sessions"]
