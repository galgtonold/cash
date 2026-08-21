"""examples/cache_calls_demo.ipynb must behave the way its prose says (CAS-243).

Call-level caching is on by DEFAULT now (task 10); the notebook was rewritten
to demonstrate that, with ``# @cash:no-cache-calls`` as the escape hatch shown
alongside each default-on example, rather than an opt-in directive shown
alongside a "without it" baseline. The polarity of every claim below is the
mirror image of what this file asserted before the flip.

A shipped demo that lies is worse than no demo: a reader who follows it and
sees the opposite concludes the feature is broken. Writing the original
version of this file caught a real error before the notebook shipped --
`out = []` in the SAME cell as the loop makes the whole cell cacheable as a
unit, so the append scenario restored and demonstrated the opposite of its
own claim. That structural note (list built in its own cell) is preserved.

Replays the notebook's own cells through a real kernel in notebook order, then
re-runs cells the way the markdown tells the reader to, and asserts the
claims. Any drift between the prose and the engine fails here.
"""
import pathlib

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

REPO = pathlib.Path(__file__).resolve().parents[2]
NB = REPO / "examples" / "cache_calls_demo.ipynb"


def _code_cells():
    """The notebook's code cells, from the COMMITTED version.

    Read from git rather than the working tree on purpose. This notebook is
    meant to be run and edited by hand — extra scratch cells, saved outputs —
    and a guard that indexes into the working copy fails the moment someone
    uses it as intended. What must not drift is the committed demo.

    Falls back to the working tree when git is unavailable (a source export, a
    detached checkout), which is still better than not checking at all.
    """
    import subprocess
    import nbformat
    try:
        raw = subprocess.run(
            ["git", "show", f"HEAD:{NB.relative_to(REPO).as_posix()}"],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        nb = nbformat.reads(raw, as_version=4)
    except (subprocess.SubprocessError, OSError, ValueError):
        nb = nbformat.read(str(NB), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _n_calls(out: str) -> int:
    """Read the executions count the notebook's own `show()` prints."""
    import re
    m = re.findall(r"\((\d+) real executions so far\)", out)
    assert m, f"no counter line in output:\n{out}"
    return int(m[-1])


def _sub_calls(out: str) -> list[tuple[int, int]]:
    """The badge's ``sub-call compute(x): n/1 hit`` rows, as (hits, total).

    This is the notebook's work signal, and it replaced ``len(CALLS)``.
    ``CALLS`` is a global written from inside ``compute``'s body, so CAS-260/265
    captures it per call and restores it on a hit: a served call reproduces its
    append without executing. That is the feature working -- the observable
    state matches an uncached run either way -- and it means the counter cannot
    distinguish "ran" from "was served". Any instrument written inside the
    callee has the same problem, so the signal has to come from outside it.

    An empty list means interception never engaged for that statement, i.e.
    every call really executed.

    Rows under the badge's ``Upstream:`` section are EXCLUDED. Those are marked
    ``^EXECUTED`` and belong to reconstructed cells, not to the statement under
    test -- a re-run of section 2 reconstructs section 1's loop and reports its
    sub-calls too, so a flat scan of the output reads another cell's hits as
    this one's and no "interception is off here" assertion can ever fail.
    """
    import re
    rows: list[tuple[int, int]] = []
    owner_is_upstream = True          # anything before the first row is a header
    for line in out.splitlines():
        stripped = line.lstrip()
        # Any row line marks a new owner -- including an uncacheable one,
        # whose label is "NOT CACHED:" and which "CACHED:" deliberately
        # matches. Before CAS-272 renamed the labels, such a row said
        # COMPUTED and was caught by the same scan.
        if "EXECUTED:" in stripped or "CACHED:" in stripped:
            owner_is_upstream = stripped.startswith("^")
            continue
        m = re.search(r"sub-call\s+\S+:\s*(\d+)/(\d+)\s+hit", stripped)
        if m and not owner_is_upstream:
            rows.append((int(m.group(1)), int(m.group(2))))
    return rows


def test_demo_notebook_claims_hold(nb_runner):
    cells = _code_cells()
    assert len(cells) == 9, f"expected 9 code cells, got {len(cells)}"

    # 1 %cash_badge, then the notebook's own cells at 2..10, then a sampler.
    nb_runner.create_notebook(["%cash_badge print", *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()

    APPEND_PLAIN, APPEND_NO_CACHE_CALLS = 5, 7
    FOLD_PLAIN, FOLD_NO_CACHE_CALLS = 8, 9
    INELIGIBLE = 10

    # The cold run must MISS, or every "it hit" assertion below would also hold
    # for a build where interception never engaged at all.
    cold = _sub_calls(nb_runner.get_output(APPEND_PLAIN))
    assert cold == [(0, 1)] * 3, (
        f"the first run should show three missed sub-calls, got {cold}"
    )

    # Claim 1: the plain append loop is cached automatically -- no directive,
    # a re-run does no work.
    nb_runner.run_cell(APPEND_PLAIN)
    out = nb_runner.get_output(APPEND_PLAIN)
    assert _sub_calls(out) == [(1, 1)] * 3, (
        "notebook claims the undirected append loop is cached by default; the "
        f"re-run's sub-calls did not all hit:\n{out}"
    )
    assert "[intercepted]" in out, (
        f"the badge does not confirm interception on the undirected call:\n{out}"
    )

    # Claim 2: with the opt-out, interception genuinely switches off -- no
    # sub-call rows at all, so every call really executed.
    nb_runner.run_cell(APPEND_NO_CACHE_CALLS)
    out = nb_runner.get_output(APPEND_NO_CACHE_CALLS)
    assert _sub_calls(out) == [], (
        "notebook claims no-cache-calls disables call caching; the badge still "
        f"reports intercepted sub-calls:\n{out}"
    )
    assert "[intercepted]" not in out, (
        f"a no-cache-calls statement must not carry the intercepted tag:\n{out}"
    )
    assert "out2 = [2, 3, 4]" in out, f"the append stopped running:\n{out}"

    # Claim 3: reordering the UNDIRECTED fold costs nothing -- this is the
    # headline behavioural flip from the opt-in era.
    nb_runner.set_cell_source(
        FOLD_PLAIN,
        cells[FOLD_PLAIN - 2].replace("[10, 20, 30]", "[30, 20, 10]"),
    )
    nb_runner.run_cell(FOLD_PLAIN)
    out = nb_runner.get_output(FOLD_PLAIN)
    assert _sub_calls(out) == [(1, 1)] * 3, (
        f"reordering the undirected fold cost executions:\n{out}"
    )
    assert "SUM 63" in out, f"the reordered fold gave a different answer:\n{out}"

    # Claim 4: the SAME reorder, under no-cache-calls, re-executes -- the
    # positive control proving claim 3 is the directive's doing and not some
    # other cause (e.g. the values simply being small).
    nb_runner.set_cell_source(
        FOLD_NO_CACHE_CALLS,
        cells[FOLD_NO_CACHE_CALLS - 2].replace("[11, 22, 33]", "[33, 22, 11]"),
    )
    nb_runner.run_cell(FOLD_NO_CACHE_CALLS)
    out = nb_runner.get_output(FOLD_NO_CACHE_CALLS)
    assert _sub_calls(out) == [], (
        f"no-cache-calls should have left the reorder uncached:\n{out}"
    )
    assert "SUM 69" in out, f"the reordered fold gave a different answer:\n{out}"

    # Claim 5: an ineligible call (merge reads its own target) is silently not
    # intercepted -- no warning (the old opt-in era's noop warning is gone),
    # no wrong value, no [intercepted] tag.
    out = nb_runner.get_output(INELIGIBLE)
    assert "acc = 5" in out, f"the ineligible-call cell produced a wrong value:\n{out}"
    assert "[intercepted]" not in out, (
        f"merge() reads its own target and must not be tagged intercepted:\n{out}"
    )
    assert "matched no cacheable call" not in out, (
        "the noop warning was removed under default-on; it must not resurface:\n"
        + out
    )
