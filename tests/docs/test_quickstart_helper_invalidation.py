"""The quickstart's "Edit a helper — the cache notices" example must be true.

The page shows two fences: call ``pipeline`` twice (second is a hit), then
redefine ``clean`` two calls below it and call again — a miss, with a different
answer. It replaced a schematic example (``def clean(x): ...`` /
``return clean(x) + ...``) that could not be run at all: ``x + Ellipsis`` raises
TypeError.

**Why this lives here and not in the docs harness.** The harness infers expected
hits/misses from call sites, and that model cannot express an invalidation
between two calls with identical arguments — it reported ``expected hits=2
misses=1, got hits=1 misses=2``, which is the harness being right about its own
model rather than the page being wrong. So the second fence is ``test:skip``
there and asserted here.

**Why this runs the fences as a real script.** Helper invalidation needs the
edited helper to live in a module with real source on disk. Measured: running
both fences through ``exec`` gives 3 (stale) in every configuration tried —
separate execs or one script, with and without ``linecache`` registration, with
and without a ``sys.modules`` entry. Only a genuine ``.py`` file gives 4. The
example sits in the page's *In a script* section, so running it as a script is
also the faithful test; an ``exec``-based one would have reported the page as
broken when it is correct.

The fences are read out of the page rather than copied, so the code under test
cannot drift from what a reader sees.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "getting-started" / "quickstart.md"

_FENCE = re.compile(r"^```python\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _helper_fences() -> tuple[str, str]:
    """The two python fences under 'Edit a helper — the cache notices'."""
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("### Edit a helper")
    end = text.index("###", start + 10)
    fences = [m.group(1) for m in _FENCE.finditer(text[start:end])]
    assert len(fences) == 2, (
        f"expected 2 python fences in the helper section, found {len(fences)}. "
        "If the example was restructured, update this test with it."
    )
    return fences[0], fences[1]


@pytest.fixture
def script_output(tmp_path) -> str:
    setup, edit = _helper_fences()
    script = tmp_path / "helper_demo.py"
    script.write_text(setup + "\n" + edit, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, script.name],
        cwd=tmp_path,                 # sandboxes the default relative .cash dir
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"the page's example does not run:\n{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout


def test_the_example_caches_at_all(script_output):
    """Non-vacuity. Cheap string functions can sit below the cost model's
    floor and never be stored, in which case a later miss would prove nothing
    about invalidation."""
    assert "'hits': 1" in script_output, (
        f"the second identical call was not a cache hit:\n{script_output}"
    )


def test_editing_a_helper_two_levels_down_invalidates(script_output):
    values = [ln.strip() for ln in script_output.splitlines() if ln.strip()]
    printed = [v for v in values if v in {"3", "4"}]
    assert printed == ["3", "3", "4"], (
        "the page says the first two calls print 3 and the call after editing "
        f"`clean` prints 4; got {printed}\n{script_output}"
    )
    assert "'misses': 2" in script_output, (
        "misses did not rise after the helper edit, so the edit did NOT "
        f"invalidate:\n{script_output}"
    )
