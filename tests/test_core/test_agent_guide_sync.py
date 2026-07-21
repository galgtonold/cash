"""The agent guide has two surfaces that must never drift.

``cash._agent_guide.AGENT_GUIDE`` (what ``cash.help()`` returns, shipped in the
wheel) and ``docs/for-coding-agents.md`` (the website page) are one document. This
test keeps them byte-identical so editing one without the other fails CI.
"""
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
