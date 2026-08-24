"""A `ttl` value that is not whole seconds is rejected, loudly (CAS-249).

`ANNOTATION_PATTERN`'s value group used to be an unanchored `\\d+`, which
matched the leading digit run and dropped the rest:

    # @cash:ttl=5m    -> CacheAnnotation(ttl=5)     FIVE SECONDS
    # @cash:ttl=1h    -> CacheAnnotation(ttl=1)     one second

A unit suffix is the natural thing to write, so this was a 60x error for `5m`
and a 3600x one for `1h`, with no warning. Its only symptom is a cache that
keeps missing, which reads as "cash isn't working" rather than "my annotation
was truncated" -- so the wrong conclusion is the easy one to reach.

The fix captures the whole value token and rejects anything that is not a plain
non-negative integer, with a `CashCacheIneffectiveWarning`. Silently ignoring it
(the ticket's minimum option) would still leave a reader believing they had set
a TTL when they had not; the point of failure here is the silence.
"""
from __future__ import annotations

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.annotations import parse_annotation_line


@pytest.mark.parametrize("line, shown", [
    ("# @cash:ttl=5m", "5m"),
    ("# @cash:ttl=2h", "2h"),
    ("# @cash:ttl=5min", "5min"),
    ("# @cash:ttl=1d", "1d"),
    ("# @cash:ttl=1.5", "1.5"),
    ("# @cash:ttl=-30", "-30"),
    ("# @cash:ttl=abc", "abc"),
])
def test_a_ttl_that_is_not_whole_seconds_is_ignored_and_warns(line, shown):
    with pytest.warns(CashCacheIneffectiveWarning, match=r"ttl"):
        ann = parse_annotation_line(line)

    assert ann is None, (
        f"{line!r} produced an annotation instead of being rejected -- the "
        "value was truncated to its leading digits"
    )


def test_the_warning_shows_the_value_and_the_correct_form():
    """A warning that does not name the offending text leaves the reader
    hunting, and one that does not show the fix leaves them guessing."""
    with pytest.warns(CashCacheIneffectiveWarning) as record:
        parse_annotation_line("# @cash:ttl=5m")

    message = str(record[0].message)
    assert "ttl=5m" in message, message
    assert "ttl=300" in message, "the warning should show the correct spelling"
    assert "IGNORED" in message, "the reader must know the annotation did nothing"


@pytest.mark.parametrize("line, expected", [
    ("# @cash:ttl=300", 300),
    ("# @cash:ttl=0", 0),
    ("# @cash: ttl = 300", 300),
    ("# @cash:ttl=3600  # one hour", 3600),
])
def test_a_valid_ttl_still_parses_and_stays_quiet(line, expected, recwarn):
    """Positive control. A fix that rejected everything would satisfy the
    tests above and break every annotated notebook.

    `ttl=0` is here deliberately: it is a real value meaning "always miss", and
    a falsy-vs-None slip would drop it (CAS-221).
    """
    ann = parse_annotation_line(line)

    assert ann is not None and ann.ttl == expected
    assert not [w for w in recwarn if issubclass(w.category, CashCacheIneffectiveWarning)], (
        "a valid ttl must not warn"
    )


def test_other_directives_are_unaffected_by_the_value_group_change(recwarn):
    """The regex change widened the value group for every directive, not just
    ttl. Valueless directives must still parse, and must not warn."""
    for line, attr in [
        ("# @cash:no-cache", "no_cache"),
        ("# @cash:persist", "persist"),
        ("# @cash:allow-random", "allow_random"),
        ("# @cash:no-cache-calls", "no_cache_calls"),
    ]:
        ann = parse_annotation_line(line)
        assert ann is not None and getattr(ann, attr) is True, line

    # Filtered by category, like its sibling above. Asserting on the whole
    # recorder made this fail on any ambient warning that happened to land in
    # the test's window -- on Windows, a GC'd ResourceWarning or a contended
    # cache write (WinError 5) under xdist. Twice on windows-3.13, never for
    # the reason the test is about. Name the offenders so the next one is
    # diagnosable instead of a list of object reprs.
    offenders = [w for w in recwarn if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert not offenders, "a valueless directive warned: " + "; ".join(
        f"{w.category.__name__}: {w.message}" for w in offenders
    )
