"""A magic must do what it was asked, or say it didn't (CAS-181).

``%cash_repair --full  # comment`` ran the NON-full repair, left the cache
intact, and printed ``[OK] Repair complete.`` The trailing comment defeated an
``== '--full'`` match, so the flag fell through to the default branch.

Why this class is worth pinning rather than patching once: ``%cash_repair
--full`` is the only documented recovery from a poisoned cache entry. Cash
poisons a value -> the user reaches for the escape hatch -> it silently no-ops
-> it reports success -> the user now trusts the poisoned number. A recovery
tool that falsely reports success converts a recoverable state into a
confidently wrong one.

The same fall-through shape sits under every magic that compares a stripped arg
string to a literal, so these tests cover the shared parse (``_args.py``) and
each mutating magic, not just the reported call site.
"""
from __future__ import annotations

import pytest

from cash.notebook.ipython._args import parse_mode, strip_inline_comment


# ---------------------------------------------------------------------------
# The shared parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
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
])
def test_strip_inline_comment(line, expected):
    assert strip_inline_comment(line) == expected


def test_parse_mode_distinguishes_default_from_unknown():
    """'' (default asked) and None (not understood) must not collapse.

    Collapsing them is the bug: an unrecognised flag becomes "run the default"
    and then reports success.
    """
    known = ('', '--full', '--state')
    assert parse_mode('', known) == ''
    assert parse_mode('--full  # c', known) == '--full'
    assert parse_mode('--FULL', known) == '--full'
    assert parse_mode('--ful', known) is None      # typo
    assert parse_mode('--full --state', known) is None  # junk
    assert parse_mode('nonsense', known) is None


# ---------------------------------------------------------------------------
# %cash_repair - the reported call site
# ---------------------------------------------------------------------------

def test_repair_full_with_trailing_comment_clears_the_cache(
    cash_magics, mock_shell, cash_instance, capsys,
):
    """The reported bug: a comment must not downgrade --full to a no-op."""
    backend = cash_instance.backend
    backend.set('poisoned', b'value', {'func_name': 't'})
    cash_magics._tracking_state.variable_lineage['x'] = 'hash1'

    cash_magics.cash_repair('--full  # comment')
    out = capsys.readouterr().out

    assert backend.list_entries() == [], "the cache was NOT cleared"
    assert len(cash_magics._tracking_state.variable_lineage) == 0
    assert 'Full repair' in out


def test_repair_unknown_argument_refuses_loudly_and_repairs_nothing(
    cash_magics, mock_shell, cash_instance, capsys,
):
    """A typo'd flag must not silently run a different repair and claim success."""
    backend = cash_instance.backend
    backend.set('entry', b'value', {'func_name': 't'})
    cash_magics._tracking_state.variable_lineage['ghost'] = 'hash1'

    cash_magics.cash_repair('--ful')  # typo for --full
    out = capsys.readouterr().out

    assert 'Repair complete' not in out, "reported success for a repair it did not run"
    assert '--ful' in out and 'unrecognised' in out.lower()
    # Nothing was touched: neither the repair the user asked for nor another one.
    assert backend.list_entries() != []
    assert 'ghost' in cash_magics._tracking_state.variable_lineage


def test_repair_default_names_the_repair_that_ran(
    cash_magics, mock_shell, cash_instance, capsys,
):
    """"Repair complete." must not read as "your cache is clean now"."""
    cash_magics.cash_repair('')
    out = capsys.readouterr().out

    assert 'default mode' in out
    assert '--full' in out, "should point at the command that DOES clear the cache"


# ---------------------------------------------------------------------------
# The same gap in the other mutating magics
# ---------------------------------------------------------------------------

def test_stats_reset_with_trailing_comment_actually_resets(
    cash_magics, mock_shell, cash_instance, capsys,
):
    """A comment must not turn `reset` into "print the stats" (CAS-157 surface)."""
    cash_magics._session.stats['cells_executed'] = 7
    cash_magics._session.stats['statements_computed'] = 3

    cash_magics.cash_stats('reset  # start over')
    out = capsys.readouterr().out

    assert 'reset' in out.lower()
    assert cash_magics._session.stats['cells_executed'] == 0
    assert cash_magics._session.stats['statements_computed'] == 0


def test_persist_on_with_trailing_comment_does_not_toggle_off(cash_magics, capsys):
    """`%cash_persist on # c` fell through to the TOGGLE - inverting the request."""
    cash_magics._persist_all = True

    cash_magics.cash_persist('on  # keep everything')

    assert cash_magics._persist_all is True, "an explicit 'on' turned persistence OFF"


def test_persist_unknown_argument_refuses_and_leaves_mode_alone(cash_magics, capsys):
    cash_magics._persist_all = False

    cash_magics.cash_persist('onn')  # typo
    out = capsys.readouterr().out

    assert 'unrecognised' in out.lower()
    assert cash_magics._persist_all is False, "a typo toggled persistence on"


def test_badge_mode_with_trailing_comment_is_applied(cash_magics, capsys):
    cash_magics._badge_mode = 'html'

    cash_magics.cash_badge('off  # too noisy')

    assert cash_magics._badge_mode == 'off'


def test_cash_on_ttl_with_trailing_comment_is_parsed(cash_magics, capsys):
    cash_magics.cash_on('ttl=3600  # one hour')

    assert cash_magics._auto_cache_enabled is True
    assert cash_magics._global_ttl == 3600


def test_cash_on_rejects_a_bad_ttl_visibly(cash_magics, capsys):
    """Refusing to enable is fine; refusing SILENTLY is not.

    The old code logged a warning (invisible in a notebook by default) and
    returned, so the user believed caching was on when it was not.
    """
    cash_magics.cash_on('ttl=one-hour')
    out = capsys.readouterr().out

    assert cash_magics._auto_cache_enabled is False
    assert 'NOT enabled' in out
