"""Unit tests for the docs-parity harness."""
from pathlib import Path

import pytest

from tests.docs._harness import Fence, extract_fences

FIXTURES = Path(__file__).parent / "_fixtures"


def test_extract_fences_returns_only_python_fences():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    assert len(fences) == 3
    assert all(isinstance(f, Fence) for f in fences)


def test_extract_fences_preserves_code_verbatim():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    assert fences[0].code == "import cash"
    assert "def f(x):" in fences[1].code
    assert fences[2].code == "result = f(10)"


def test_extract_fences_records_line_numbers():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    # Fence 1: ```python at line 5 → code at line 6 → ``` at line 7
    assert fences[0].line_start >= 5
    assert fences[0].line_end >= fences[0].line_start
