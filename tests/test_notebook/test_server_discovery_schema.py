"""Session-lookup must understand the MODERN jupyter_server schema.

`_search_servers_for_notebook` read two keys from the CLASSIC notebook server:
``server['notebook_dir']`` and ``session['notebook']['path']``. Every current
JupyterLab runs `jupyter_server`, whose running-server descriptor calls that
directory ``root_dir`` — so the lookup raised ``KeyError('notebook_dir')``
*after* a perfectly successful 200 response, the blanket ``except`` swallowed
it, and discovery returned None.

The consequence was not a missing convenience: notebook discovery failing turns
upstream dependency tracking off for the whole session, so an edited-but-not-
re-run cell silently feeds a STALE value to everything downstream. Two
independent round-10 testers hit it, and both were misdirected by the log line,
which reported a schema mismatch as "Failed to query sessions".

The fast suite could never see it — it drives NotebookClient, not a real server.
So these tests pin the schema contract directly.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest

from cash.notebook import server_discovery as sd

KERNEL_ID = "36d5c23f-5334-488e-b294-e7e8fb66159c"

# Exactly as `jupyter_server.serverapp.list_running_servers()` yields it.
MODERN_SERVER = {
    "base_url": "/", "hostname": "127.0.0.1", "password": False, "pid": 1234,
    "port": 8901, "root_dir": os.path.join("C:", "work"), "secure": False,
    "sock": "", "token": "tok", "url": "http://127.0.0.1:8901/", "version": "2.14.0",
}

# The classic notebook server's descriptor, which is what the code assumed.
LEGACY_SERVER = dict(MODERN_SERVER)
LEGACY_SERVER.pop("root_dir")
LEGACY_SERVER["notebook_dir"] = os.path.join("C:", "work")


def _session(*, with_notebook_key: bool, with_path: bool = True):
    s = {"id": "s1", "kernel": {"id": KERNEL_ID}, "name": "work.ipynb", "type": "notebook"}
    if with_path:
        s["path"] = "work.ipynb"
    if with_notebook_key:
        s["notebook"] = {"path": "work.ipynb", "name": "work.ipynb"}
    return s


@contextmanager
def _serving(monkeypatch, server, sessions):
    """Point discovery at *server* and make /api/sessions return *sessions*."""
    monkeypatch.setattr(sd, "_collect_running_servers", lambda: [server])

    class _Resp:
        def read(self):
            return json.dumps(sessions).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sd.urllib.request, "urlopen", lambda *a, **k: _Resp())
    yield


def test_modern_jupyter_server_schema_resolves(monkeypatch):
    """root_dir + session['path'] — the shape every current JupyterLab serves."""
    with _serving(monkeypatch, MODERN_SERVER, [_session(with_notebook_key=False)]):
        got = sd._search_servers_for_notebook(KERNEL_ID)
    assert got == os.path.join(MODERN_SERVER["root_dir"], "work.ipynb")


def test_modern_server_with_deprecated_notebook_key(monkeypatch):
    """jupyter_server still emits the deprecated `notebook` dict; root_dir is
    the key that actually broke, so pin this combination explicitly."""
    with _serving(monkeypatch, MODERN_SERVER, [_session(with_notebook_key=True)]):
        got = sd._search_servers_for_notebook(KERNEL_ID)
    assert got == os.path.join(MODERN_SERVER["root_dir"], "work.ipynb")


def test_legacy_notebook_server_still_resolves(monkeypatch):
    """The classic server must keep working — this is a widened contract, not a
    swapped one."""
    with _serving(monkeypatch, LEGACY_SERVER, [_session(with_notebook_key=True, with_path=False)]):
        got = sd._search_servers_for_notebook(KERNEL_ID)
    assert got == os.path.join(LEGACY_SERVER["notebook_dir"], "work.ipynb")


def test_non_matching_kernel_returns_none(monkeypatch):
    with _serving(monkeypatch, MODERN_SERVER, [_session(with_notebook_key=False)]):
        assert sd._search_servers_for_notebook("some-other-kernel") is None


def test_pathless_session_does_not_raise(monkeypatch):
    """A session carrying neither key must be skipped, not crash the lookup."""
    with _serving(monkeypatch, MODERN_SERVER,
                  [_session(with_notebook_key=False, with_path=False)]):
        assert sd._search_servers_for_notebook(KERNEL_ID) is None


def test_schema_mismatch_is_not_reported_as_a_query_failure(monkeypatch, caplog):
    """The old log line blamed the network for a parse error and sent two
    testers hunting proxies and tokens. A missing key must say so."""
    broken = {"url": "http://127.0.0.1:8901/", "token": "tok"}  # no dir key at all
    session = {"id": "s1", "name": "work.ipynb"}                # no 'kernel' key
    with _serving(monkeypatch, broken, [session]):
        with caplog.at_level("DEBUG"):
            assert sd._search_servers_for_notebook(KERNEL_ID) is None
    text = caplog.text.lower()
    assert "missing key" in text or "schema" in text
    assert "failed to query sessions" not in text


@pytest.mark.parametrize("server", [MODERN_SERVER, LEGACY_SERVER])
def test_both_schemas_join_against_their_own_root(monkeypatch, server):
    with _serving(monkeypatch, server, [_session(with_notebook_key=True)]):
        got = sd._search_servers_for_notebook(KERNEL_ID)
    expected_root = server.get("root_dir") or server["notebook_dir"]
    assert got == os.path.join(expected_root, "work.ipynb")
