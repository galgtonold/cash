"""Colab cell-source reader: cash reads live cells via ``google.colab._message``.

Colab notebooks live in Google Drive, so ``get_notebook_path()`` returns a
non-openable ``/fileId=...`` and the file reader yields ``[]`` — which silently
disables upstream auto-resolution (a downstream-only run NameErrors). Reading
the live notebook JSON from the Colab frontend restores it. These tests fake a
Colab runtime so the branch is exercised off Colab.
"""
from __future__ import annotations

import sys
import types

import pytest

from cash.notebook import server_discovery as sd

IPYNB = {
    "ipynb": {
        "cells": [
            {"cell_type": "code", "source": ["total = 1\n", "print(total)"], "id": "c1"},
            {"cell_type": "markdown", "source": ["# heading"]},
            {"cell_type": "code", "source": "answer = total + 1", "id": "c2"},
        ]
    }
}


def _install_fake_colab(monkeypatch, blocking_request):
    """Make ``sd._in_colab()`` true and ``from google.colab import _message`` work."""
    google = sys.modules.get("google") or types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    message = types.ModuleType("google.colab._message")
    message.blocking_request = blocking_request
    colab._message = message
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab._message", message)
    sd.invalidate_notebook_cells_cache()


def test_reads_code_cells_via_get_ipynb(monkeypatch):
    _install_fake_colab(monkeypatch, lambda req, timeout_sec=None: IPYNB)
    # Markdown filtered out; list + string sources both handled.
    assert sd.get_notebook_cells() == ["total = 1\nprint(total)", "answer = total + 1"]


def test_reads_cells_with_ids(monkeypatch):
    _install_fake_colab(monkeypatch, lambda req, timeout_sec=None: IPYNB)
    assert sd.get_notebook_cells_with_ids() == [
        ("c1", "total = 1\nprint(total)"),
        ("c2", "answer = total + 1"),
    ]


def test_frontend_failure_falls_through_gracefully(monkeypatch):
    def _boom(req, timeout_sec=None):
        raise TimeoutError("no frontend")

    _install_fake_colab(monkeypatch, _boom)
    monkeypatch.setattr(sd, "get_notebook_path", lambda: None)
    # Must not raise; degrades to the empty file-reader result.
    assert sd.get_notebook_cells() == []


def test_result_cached_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _counting(req, timeout_sec=None):
        calls["n"] += 1
        return IPYNB

    _install_fake_colab(monkeypatch, _counting)
    sd.get_notebook_cells()
    sd.get_notebook_cells()
    assert calls["n"] == 1  # second read served from the short-TTL cache


def test_no_op_when_not_in_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert sd._in_colab() is False
    assert sd._try_colab_notebook_cells(False) is None
