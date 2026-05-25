"""Fixtures for docs feature-parity tests.

PR1: minimal conftest. PR3 will add autouse fixtures that monkeypatch SDK
boundaries (Anthropic, OpenAI, Redis, S3) based on what each doc page
imports. For now the 3 PR1 pages have no external deps so no fixtures
are needed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_cash_state(tmp_path, monkeypatch):
    """Each doc page gets a fresh Cash default instance with an isolated
    cache directory in tmp_path. This prevents tests from interfering
    via shared on-disk caches.
    """
    monkeypatch.setenv("CASH_DEFAULT_BACKEND_TYPE", "in_memory")
    # The default Cash singleton lazily initializes on first import; force
    # a fresh import by clearing module-level state. The exact mechanism
    # depends on Cash's internals — see src/cash/__init__.py for the
    # default-instance hook.
    import cash
    if hasattr(cash, "_reset_default_for_tests"):
        cash._reset_default_for_tests()
    yield
