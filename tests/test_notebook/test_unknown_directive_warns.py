"""An unknown ``# @cash:`` directive warns instead of dropping silently.

The run-together spellings (``nocache``, ``allowrandom``, ...) and the opt-in
``cache-calls`` were removed. With unknown directives ignored in silence, an
old ``# @cash:nocache`` went on caching the statement it was written to stop,
and the comment in the cell still looked as if it worked.
"""

from __future__ import annotations

import ast
import warnings

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook import annotations
from cash.notebook.annotations import (
    KNOWN_DIRECTIVES,
    get_statement_annotations,
    parse_annotation_line,
    parse_annotations_in_range,
)


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch):
    """Each test starts as a new session, with nothing warned yet."""
    monkeypatch.setattr(annotations, "_warned_unknown_directives", set())


def _unknown_directive_warnings(*lines: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for line in lines:
            parse_annotation_line(line)
    return [w for w in caught if getattr(w.message, "code", None) == "ANNOT-UNKNOWN-DIRECTIVE"]


@pytest.mark.parametrize(
    "written, meant",
    [
        ("nocache", "no-cache"),
        ("allowrandom", "allow-random"),
        ("cachefit", "cache-fit"),
        ("nocachecalls", "no-cache-calls"),
        ("no_cache", "no-cache"),
        ("NoCache", "no-cache"),
        ("assumesafe", "assume-safe"),
    ],
)
def test_a_removed_spelling_warns_and_suggests_the_hyphenated_one(written, meant):
    caught = _unknown_directive_warnings(f"# @cash:{written}")

    assert len(caught) == 1
    assert issubclass(caught[0].category, CashCacheIneffectiveWarning)
    message = str(caught[0].message)
    assert f"`# @cash:{written.lower()}`" in message
    assert f"did you mean `# @cash:{meant}`?" in message
    assert "IGNORED" in message


def test_an_unknown_directive_with_no_near_match_lists_the_real_ones():
    caught = _unknown_directive_warnings("# @cash:cache-calls")

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "did you mean" not in message
    for known in KNOWN_DIRECTIVES:
        assert f"`{known}`" in message


def test_an_unknown_directive_still_parses_to_nothing():
    """The warning is the only change: the comment still does nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert parse_annotation_line("# @cash:nocache") is None


def test_each_unknown_name_warns_once_per_session():
    """A cell is parsed on every run, and the upstream checker parses every
    cell above the one running, so without a ledger one typo would warn on
    every cell execution."""
    caught = _unknown_directive_warnings(
        "# @cash:nocache",
        "# @cash:nocache",
        "x = 1  # @cash:nocache",
        "# @cash:perist",
        "# @cash:perist",
    )

    assert [str(w.message).split("`")[1] for w in caught] == ["# @cash:nocache", "# @cash:perist"]


def test_the_ledger_is_per_name_not_per_suggestion():
    """``nocache`` and ``no_cache`` are different typos of the same directive;
    each is its own line to fix, so each is reported."""
    caught = _unknown_directive_warnings("# @cash:nocache", "# @cash:no_cache")

    assert len(caught) == 2


@pytest.mark.parametrize(
    "line",
    [f"# @cash:{name}" for name in KNOWN_DIRECTIVES if name != "ttl"]
    + ["# @cash:ttl=60", "x = f()  # @cash:assume-safe"],
)
def test_a_known_directive_does_not_warn(line):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_annotation_line(line)

    assert caught == []


def test_a_bare_ttl_is_not_an_unknown_directive_but_a_bad_ttl():
    """``ttl`` is a real directive; without ``=N`` it is the malformed-value
    case, which already has its own code."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert parse_annotation_line("# @cash:ttl") is None

    assert [w.message.code for w in caught] == ["ANNOT-TTL-INVALID"]
    assert "`# @cash:ttl` gives no number of seconds" in str(caught[0].message)


def test_a_statement_under_an_old_nocache_is_still_cacheable():
    """What the warning is about: the old spelling does not stop caching."""
    source = "# @cash:nocache\nx = compute()\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ann = get_statement_annotations(source, ast.parse(source).body[0])

    assert ann.no_cache is False
    assert [w.message.code for w in caught] == ["ANNOT-UNKNOWN-DIRECTIVE"]


def test_the_warning_points_at_the_line_in_the_cell():
    """Resolved from the frame, a warning raised while a cell runs names
    ipykernel's ``run_cell`` line. With the cell line known, it names that."""
    lines = ["x = 1", "# @cash:nocache", "y = compute()"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_annotations_in_range(lines, 3, 3)

    assert [(w.filename, w.lineno) for w in caught] == [("<cash>", 2)]


def test_without_a_line_number_the_warning_blames_the_caller():
    """The decorator path parses a function's source line by line with no cell
    to point into; there the caller's own frame is the right place."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_annotation_line("# @cash:nocache")

    assert caught[0].filename == __file__
