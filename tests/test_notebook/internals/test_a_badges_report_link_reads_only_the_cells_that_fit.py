"""The badge's bug-report link reads the notebook's cells only as far as
they fit in the link.

Every badge (two or more a cell) builds the link, which carries the
notebook's source when it fits. Each build read and trimmed every cell, twice,
before finding the text too long: more work on every cell the longer the
notebook. It now stops reading once the text is past what a link can hold.

Counted in cells read, not timed.
"""

from __future__ import annotations

from cash.notebook.badge_renderer.view_builder import build_bug_report_url


class _CountedCell(str):
    """A cell's source that counts how often it is read (``strip``)."""

    reads = 0

    def strip(self, *args):
        type(self).reads += 1
        return super().strip(*args)


def _cells_read(n: int) -> int:
    _CountedCell.reads = 0
    cells = [_CountedCell(f"value_{i} = compute(value_{i - 1}, {i})  # step {i}") for i in range(n)]
    metrics = [{"code": "x = 1", "status": "EXECUTED", "execution_time": 0.01}]
    url = build_bug_report_url(metrics, {"notebook_source": cells})
    assert url.startswith("https://"), url
    return _CountedCell.reads


def test_a_long_notebook_is_read_no_further_than_a_short_one_that_overflows():
    overflowing = _cells_read(300)
    long = _cells_read(3000)

    assert overflowing > 0, "the link read no cell: the counter saw no work"
    # Both overflow the link; the ten times longer notebook reads no more.
    assert long == overflowing, f"{long} cells read at 3000 cells against {overflowing} at 300"


def test_a_notebook_that_fits_is_in_the_link():
    _CountedCell.reads = 0
    cells = [_CountedCell("a = 1"), _CountedCell("b = a + 1")]
    url = build_bug_report_url([{"code": "b = a + 1", "status": "EXECUTED"}], {"notebook_source": cells})
    assert "Cell%201" in url and "b%20%3D%20a%20%2B%201" in url, url
