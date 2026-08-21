"""CAS-173: one SyntaxError in an upstream cell must not silently disable
caching for the whole notebook.

Two testers (P3, P5) independently hit this: SAVING (not even running) a
half-written upstream cell made every downstream cell recompute, with no
message reaching the user. cash detected the parse failure precisely
(``[UPSTREAM] Syntax error in cell N``) and swallowed it, while the badge and
``auto_cache_enabled`` still claimed caching was on.

The fix has two parts:

1. DISCLOSE  - a visible ``CashUpstreamSyntaxWarning`` names the offending cell.
2. CONTAIN   - a downstream cell that does NOT depend on the broken cell keeps
   caching; the unparseable cell is skipped, not fatal to the whole simulation.

These tests drive a REAL kernel via ``nb_runner`` and exercise the
SAVE-not-run trigger: the broken cell is only written to disk, never executed.

Distinct from CAS-163 (VALID multi-line ``%``-format code falsely rejected):
here the code is GENUINELY broken and must be reported, not swallowed - while
a valid cell must still cache (the ``control`` test pins that CAS-163 stays
fixed).
"""
import pytest
from conftest import shows_cached

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(240)]

SETUP = (
    "import numpy as np\n"
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print"
)

# An independent, comfortably-cacheable cell that depends ONLY on numpy - never
# on the cell we break. Its cache fate is the whole point: a broken UNRELATED
# upstream cell must not touch it.
EXPENSIVE = (
    "data = np.linspace(0.0, 1.0, 2_000_000)\n"
    "result = float((data.reshape(2000, 1000) @ data.reshape(1000, 2000)).sum())\n"
    "print('result=', result)"
)


def test_broken_upstream_cell_keeps_independent_downstream_cache(nb_runner):
    """A/B/A: breaking an unrelated upstream cell (save-only) must not evict the
    cache of a downstream cell that does not depend on it (CAS-173 CONTAIN)."""
    nb_runner.create_notebook([SETUP, "y = 1", EXPENSIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "result=" in nb_runner.get_output(3)

    # A) baseline: an isolated re-run of the expensive cell restores from cache.
    nb_runner.run_cell(3)
    out_a = nb_runner.get_output(3)
    assert shows_cached(out_a), f"baseline did not cache:\n{out_a}"

    # B) break cell 2 by SAVING only (never execute it), then re-run cell 3.
    nb_runner.set_cell_source(2, "y = 1 +")  # SyntaxError, saved but not run
    nb_runner.run_cell(3)
    out_b = nb_runner.get_output(3)
    assert shows_cached(out_b), (
        "A downstream cell that does NOT depend on the broken cell lost its "
        "cache when an unrelated upstream cell had a SyntaxError (CAS-173).\n"
        f"{out_b}"
    )

    # A') fix cell 2 again; cell 3 still restores.
    nb_runner.set_cell_source(2, "y = 1")
    nb_runner.run_cell(3)
    out_a2 = nb_runner.get_output(3)
    assert shows_cached(out_a2), f"post-fix did not cache:\n{out_a2}"


def test_dependent_downstream_cell_never_serves_wrong_value(nb_runner):
    """Correctness guard for the case the containment fix could get wrong: a
    downstream cell that DOES depend on the broken cell.

    The broken cell never executed, so its output is unchanged in memory; the
    dependent cell must therefore still compute the correct value (whether it
    restores or recomputes) - never a stale/wrong one. This pins that skipping
    the unparseable cell does not corrupt a genuine dependent."""
    nb_runner.create_notebook([
        SETUP,
        "base = 100",
        "derived = base + 1\nprint('derived=', derived)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "derived= 101" in nb_runner.get_output(3)

    # Break the producer of `base` by SAVING only (never run it). `base` stays
    # 100 in memory because the broken cell cannot execute.
    nb_runner.set_cell_source(2, "base = 100 +")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "derived= 101" in out, (
        "a downstream cell that depends on the broken cell served a wrong value "
        f"after the upstream cell was skipped (CAS-173 correctness).\n{out}"
    )


def test_valid_upstream_cell_still_caches_control(nb_runner):
    """Control (CAS-163 guard): with a perfectly VALID upstream cell, the
    downstream cell still restores. If the containment logic were too eager and
    treated valid cells as broken, this would regress."""
    nb_runner.create_notebook([SETUP, "y = 1", EXPENSIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "result=" in nb_runner.get_output(3)

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert shows_cached(out), f"valid-upstream control did not cache:\n{out}"
