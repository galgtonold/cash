"""``_try_ipynbname_path`` must never let the ipynbname library crash discovery.

``ipynbname.path()`` raises whatever its internal probing hits when no
discoverable Jupyter server backs the kernel. Under Google Colab and other
server-less runtimes that is an ``IndexError`` from ``_get_kernel_id`` indexing
an empty running-servers list. The old ``except (ImportError, OSError,
AttributeError)`` did not cover it, so the error propagated out of
``get_notebook_path`` → ``check_and_reexecute`` and tripped the upstream
pipeline's broad failure handler, disabling caching for the cell with the
user-visible "Cash auto-caching failed: list index out of range".

Discovery is best-effort, so ANY failure here must degrade to "no path found".
"""
from __future__ import annotations

import sys
import types

import pytest

from cash.notebook import server_discovery as sd


@pytest.mark.parametrize("exc", [
    IndexError("list index out of range"),   # the real Colab / server-less shape
    RuntimeError("boom"),                     # any other library-internal failure
    ValueError("nope"),
])
def test_try_ipynbname_path_swallows_library_errors(monkeypatch, exc):
    fake = types.ModuleType("ipynbname")

    def _boom():
        raise exc

    fake.path = _boom
    monkeypatch.setitem(sys.modules, "ipynbname", fake)

    # Must return None, NOT propagate — a discovery library cannot crash cash.
    assert sd._try_ipynbname_path() is None


def test_try_ipynbname_path_returns_str_on_success(monkeypatch):
    fake = types.ModuleType("ipynbname")
    fake.path = lambda: "/some/dir/notebook.ipynb"
    monkeypatch.setitem(sys.modules, "ipynbname", fake)

    assert sd._try_ipynbname_path() == "/some/dir/notebook.ipynb"
