"""Reading the cash badge in a cell's text output."""


#
# Commit 342185c unified the badge on ONE word per state: the row labels
# `RESTORED` / `COMPUTED` became `CACHED` / `EXECUTED`, matching the header.
# The tests below were not updated with it, which is how 22 of them came to
# assert a word the renderer no longer emits.
#
# These helpers exist because the naive replacement is WRONG in both
# directions, and silently so:
#
#   * `"CACHED" in out` also matches `NOT CACHED` -- the label an uncacheable
#     row carries INSTEAD of EXECUTED. A positive check would pass on a row
#     that was never cached; a negative check would fail on one.
#   * `"EXECUTED" in out` also matches the cell HEADER, `[Cash] EXECUTED (...)`,
#     which is present whatever the rows say. The old word `COMPUTED` appeared
#     only in rows, so a bare swap quietly widens the assertion to always-true.
#
# Rows are rendered as `"  {tag}: {code}"` (renderers/text.py), and an upstream
# row prefixes the tag with `^`. So the colon is what separates a row from the
# header, and stripping `NOT CACHED:` first is what separates a cached row from
# an uncacheable one.
def shows_cached(output: str) -> bool:
    """True when the badge shows at least one CACHED (restored) statement."""
    return "CACHED:" in output.replace("NOT CACHED:", "")


def shows_executed(output: str) -> bool:
    """True when the badge shows at least one EXECUTED statement ROW.

    Deliberately requires the colon, so the cell header does not count.
    """
    return "EXECUTED:" in output
