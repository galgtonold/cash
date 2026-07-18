"""CAS-205: a connection filename with no kernel id must degrade, not raise.

``get_notebook_path`` parsed the kernel id with
``os.path.basename(f).split('-', 1)[1].split('.')[0]``. Under bare nbclient /
papermill / nbconvert the connection file can be a plain ``kernel.json`` with no
``-``, so ``[1]`` raised ``IndexError`` — which was NOT in the caller's except
tuple ``(ImportError, AttributeError, OSError, RuntimeError)``. It escaped
``get_notebook_path``, disabled caching for the entire run, and printed

    Cash auto-caching failed: list index out of range. Falling back to normal execution.

on EVERY cell, instead of cash's intended one-time "notebook not found"
disclosure.

The integration fixture never caught this because it INJECTS the notebook path,
so resolution returns long before this branch — the same harness-blindness class
as CAS-190. These tests therefore drive the parse directly rather than through
``nb_runner``.
"""
import sys
import types

import pytest

import cash.notebook.server_discovery as sd
from cash.notebook.server_discovery import _kernel_id_from_connection_file


@pytest.mark.parametrize("filename,expected", [
    # Well-formed: semantics identical to the original expression.
    (r"C:\rt\kernel-abc123.json", "abc123"),
    ("/rt/kernel-abc123.json", "abc123"),
    # A UUID's own dashes survive (split on the FIRST '-' only).
    ("/rt/kernel-2f9a-4b1c-88de.json", "2f9a-4b1c-88de"),
    # The CAS-205 shapes: no id to extract -> None, never an exception.
    (r"C:\rt\kernel.json", None),
    ("kernel.json", None),
    ("kernel-.json", None),
    ("", None),
    (None, None),
])
def test_kernel_id_parse_is_total(filename, expected):
    assert _kernel_id_from_connection_file(filename) == expected


@pytest.fixture
def _isolated_discovery(monkeypatch):
    """Neutralise caches and the earlier probes so the ipykernel branch is hit."""
    monkeypatch.setattr(sd, "_cached_notebook_path", None, raising=False)
    monkeypatch.setattr(sd, "_negative_cache_time", 0.0, raising=False)
    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)
    monkeypatch.setattr(sd, "_try_ipynbname_path", lambda: None)


def _fake_ipykernel(monkeypatch, connection_file):
    mod = types.ModuleType("ipykernel")
    mod.get_connection_file = lambda: connection_file
    monkeypatch.setitem(sys.modules, "ipykernel", mod)


def test_get_notebook_path_degrades_on_idless_connection_file(
    monkeypatch, _isolated_discovery,
):
    """The CAS-205 regression: return None rather than raising IndexError."""
    _fake_ipykernel(monkeypatch, r"C:\Temp\runtime\kernel.json")
    assert sd.get_notebook_path() is None


def test_get_notebook_path_degrades_when_connection_file_raises(
    monkeypatch, _isolated_discovery,
):
    """An unavailable connection file is still handled (pre-existing behaviour)."""
    mod = types.ModuleType("ipykernel")

    def _boom():
        raise RuntimeError("no connection file")

    mod.get_connection_file = _boom
    monkeypatch.setitem(sys.modules, "ipykernel", mod)
    assert sd.get_notebook_path() is None
