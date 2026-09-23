"""A magic must do what it was asked, or say it didn't.

A trailing ``# comment`` used to defeat an ``== 'reset'``-style match, so the
flag fell through to the magic's default branch: a *different* operation,
reported as a success. ``%cash_stats reset  # comment`` printed the stats and
reset nothing; ``%cash_persist on  # comment`` toggled persistence off.

The same fall-through shape sits under every magic that compares a stripped arg
string to a literal, so these tests cover the shared parse (``_args.py``) and
each mutating magic.
"""

from __future__ import annotations

import pytest

from cash.notebook.ipython._args import parse_mode, strip_inline_comment

# ---------------------------------------------------------------------------
# The shared parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("--full  # comment", "--full"),
        ("--full#comment", "--full"),
        ("  --full  ", "--full"),
        ("# only a comment", ""),
        ("", ""),
        (None, ""),
        ("reset # reset the stats", "reset"),
        # A '#' inside quotes is data, not a comment - IPython passes the raw
        # string through, so a quoted path must survive intact.
        ('"my#file.json"', '"my#file.json"'),
        ("'a#b' # trailing", "'a#b'"),
        # An escaped quote must not be read as closing the string.
        (r'"a\"#b" # trailing', r'"a\"#b"'),
    ],
)
def test_strip_inline_comment(line, expected):
    assert strip_inline_comment(line) == expected


def test_parse_mode_distinguishes_default_from_unknown():
    """'' (default asked) and None (not understood) must not collapse.

    Collapsing them is the bug: an unrecognised flag becomes "run the default"
    and then reports success.
    """
    known = ("", "json", "reset")
    assert parse_mode("", known) == ""
    assert parse_mode("reset  # c", known) == "reset"
    assert parse_mode("RESET", known) == "reset"
    assert parse_mode("rest", known) is None  # typo
    assert parse_mode("reset json", known) is None  # junk
    assert parse_mode("nonsense", known) is None


# ---------------------------------------------------------------------------
# The mutating magics
# ---------------------------------------------------------------------------


def test_stats_reset_with_trailing_comment_actually_resets(
    cash_magics,
    mock_shell,
    cash_instance,
    capsys,
):
    """A comment must not turn `reset` into "print the stats"."""
    cash_magics._session.stats["cells_executed"] = 7
    cash_magics._session.stats["statements_computed"] = 3

    cash_magics.cash_stats("reset  # start over")
    out = capsys.readouterr().out

    assert "reset" in out.lower()
    assert cash_magics._session.stats["cells_executed"] == 0
    assert cash_magics._session.stats["statements_computed"] == 0


def test_persist_on_with_trailing_comment_does_not_toggle_off(cash_magics, capsys):
    """`%cash_persist on # c` fell through to the TOGGLE - inverting the request."""
    cash_magics._persist_all = True

    cash_magics.cash_persist("on  # keep everything")

    assert cash_magics._persist_all is True, "an explicit 'on' turned persistence OFF"


def test_persist_unknown_argument_refuses_and_leaves_mode_alone(cash_magics, capsys):
    cash_magics._persist_all = False

    cash_magics.cash_persist("onn")  # typo
    out = capsys.readouterr().out

    assert "unrecognised" in out.lower()
    assert cash_magics._persist_all is False, "a typo toggled persistence on"


def test_badge_mode_with_trailing_comment_is_applied(cash_magics, capsys):
    cash_magics.badges.mode = "html"

    cash_magics.cash_badge("off  # too noisy")

    assert cash_magics.badges.mode == "off"


def test_cash_on_ttl_with_trailing_comment_is_parsed(cash_magics, capsys):
    cash_magics.cash_on("ttl=3600  # one hour")

    assert cash_magics._auto_cache_enabled is True
    assert cash_magics.global_ttl == 3600


def test_cash_on_rejects_a_bad_ttl_visibly(cash_magics, capsys):
    """Refusing to enable is fine; refusing SILENTLY is not.

    The old code logged a warning (invisible in a notebook by default) and
    returned, so the user believed caching was on when it was not.
    """
    cash_magics.cash_on("ttl=one-hour")
    out = capsys.readouterr().out

    assert cash_magics._auto_cache_enabled is False
    assert "NOT enabled" in out
