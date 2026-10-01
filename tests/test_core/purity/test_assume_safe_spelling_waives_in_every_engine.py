"""``# @cash:assume-safe`` waives the same lines in every engine.

The notebook reads it per statement, the decorator's purity analysis per line
of a function (``audited_lines``) and the runtime effect observer per line it
sees an effect on (``line_waived``). One spelling must waive in all three or in
none, and a misspelling must be reported wherever it is written.
"""

from __future__ import annotations

import warnings

import pytest

from cash.analysis import annotations
from cash.analysis.annotations import audited_lines, parse_annotation_line
from cash.effect_observer import line_waived

pytestmark = pytest.mark.core

SPELLINGS = [
    ("x = f()  # @cash: assume-safe", True),
    ("x = f()  # @cash:assume-safe", True),
    ("x = f()  # @cash: ASSUME-SAFE", True),
    ("x = f()  # @cash: no-cache  # @cash: assume-safe", True),
    ("x = f()  # @cash: assume-safe-later", False),
    ("x = f()  # @cash: assume_safe", False),
    ("x = f()  # a plain comment", False),
]


@pytest.fixture(autouse=True)
def _fresh_unknown_directive_memo(monkeypatch):
    monkeypatch.setattr(annotations, "_warned_unknown_directives", set())


@pytest.mark.parametrize(("line", "waives"), SPELLINGS)
def test_every_engine_gives_the_same_answer(line, waives, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        notebook = bool(getattr(parse_annotation_line(line), "assume_safe", False))
        decorator = 1 in audited_lines(line)[0]
    source = tmp_path / "mod.py"
    source.write_text(line + "\n", encoding="utf-8")
    runtime = line_waived(str(source), 1)
    assert (notebook, decorator, runtime) == (waives, waives, waives)


def test_two_directives_on_one_line_both_apply():
    ann = parse_annotation_line("x = f()  # @cash: no-cache  # @cash: assume-safe")
    assert ann is not None and ann.no_cache and ann.assume_safe


def test_the_decorator_reports_a_misspelled_waiver():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audited_lines("def f():\n    post()  # @cash: assume_safe\n")
    assert any("assume-safe" in str(w.message) for w in caught)
