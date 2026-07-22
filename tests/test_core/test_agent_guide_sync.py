"""The agent guide has two surfaces that must never drift.

``cash._agent_guide.AGENT_GUIDE`` (what ``cash.help()`` returns, shipped in the
wheel) and ``docs/for-coding-agents.md`` (the website page) are one document. This
test keeps them byte-identical so editing one without the other fails CI.
"""
import os
from pathlib import Path

import pytest

from cash._agent_guide import AGENT_GUIDE


def _docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "for-coding-agents.md"


def test_agent_guide_matches_docs_page():
    doc = _docs_path()
    if not doc.exists():
        pytest.skip("docs/ not present (installed package, not the repo)")
    assert doc.read_text(encoding="utf-8") == AGENT_GUIDE, (
        "cash._agent_guide.AGENT_GUIDE and docs/for-coding-agents.md have drifted "
        "— they are one guide, two surfaces; edit both."
    )


def test_cash_help_returns_the_guide():
    import cash
    assert cash.help() == AGENT_GUIDE


def test_cash_help_survives_a_legacy_codepage_console():
    """``cash.help()`` must not raise where a Windows console actually runs it.

    The guide contains ``→``/``—``. On a default Windows console (cp1252) printing
    those raised ``UnicodeEncodeError``, so the one call this library tells coding
    agents to make first died on the machine most likely to make it. It survived
    the suite because pytest captures stdout through a UTF-8 buffer, and the
    in-process assertion above redirects into a ``StringIO`` -- which has no
    encoding at all. Only a real subprocess with a legacy codepage reproduces it.
    """
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    cp = subprocess.run(
        [sys.executable, "-c", "import cash; cash.help()"],
        capture_output=True, text=True, env=env, encoding="cp1252", errors="replace",
    )
    assert cp.returncode == 0, f"cash.help() crashed under cp1252 stdout:\n{cp.stderr}"
    assert "UnicodeEncodeError" not in cp.stderr
    # The prose still has to arrive -- degrading the glyphs must not blank the page.
    assert "guide for coding agents" in cp.stdout
