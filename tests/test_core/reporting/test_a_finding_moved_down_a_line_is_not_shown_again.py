"""A once-per-cache warning is not shown again because its lines moved.

KEY-AMBIENT-READ and IMPURE-SIDE-EFFECTS are shown once per cache: the next
run on the same cache stays quiet. Each run compared the whole text, line
numbers and all, so an edit elsewhere in the file -- a line added above the
function -- showed the same findings again, on every run after every edit.
"""

from __future__ import annotations

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.timeout(120)

_BODY = """
import sys, time, warnings
warnings.simplefilter("always")
import cash
{above}
@cash.cache
def cv_score(n):
    t = time.perf_counter()
    score = n * 2{extra}
    return score, time.perf_counter() - t

cv_score(int(sys.argv[1]))  # a new argument: a miss every run
"""


def _run(tmp_path, n, *, above="", extra=""):
    script = tmp_path / "pipeline.py"
    script.write_text(_BODY.format(above=above, extra=extra), encoding="utf-8")
    return run_python(script, str(n), cwd=tmp_path, timeout=100).stderr


def test_lines_moved_by_an_edit_above_do_not_show_it_again(tmp_path):
    assert "KEY-AMBIENT-READ" in _run(tmp_path, 1)
    # Two lines added above the function: the same findings, two lines down.
    assert "KEY-AMBIENT-READ" not in _run(tmp_path, 2, above="X = 1\nY = 2\n")


def test_an_edit_that_changes_the_findings_shows_them(tmp_path):
    """The control: a new finding is new text, and is shown."""
    assert "KEY-AMBIENT-READ" in _run(tmp_path, 1)
    assert "KEY-AMBIENT-READ" in _run(tmp_path, 2, extra="\n    score += time.time() * 0")
