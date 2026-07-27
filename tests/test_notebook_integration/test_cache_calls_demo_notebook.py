"""examples/cache_calls_demo.ipynb must behave the way its prose says (CAS-243).

A shipped demo that lies is worse than no demo: a reader who follows it and
sees the opposite concludes the feature is broken. Writing this caught a real
error before the notebook shipped -- `out = []` in the SAME cell as the loop
makes the whole cell cacheable as a unit, so the append scenario restored and
demonstrated the opposite of its own claim.

Replays the notebook's own cells through a real kernel in notebook order, then
re-runs cells the way the markdown tells the reader to, and asserts the three
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


class _Counter:
    """Samples the LIVE global counter, not a cell's stale output.

    ``CALLS`` accumulates across every cell, so reading it out of a cell's
    recorded output tells you what the total was when that cell last ran — which
    is not the same as what it is now. A dedicated sampler cell is the only
    honest way to bracket a single re-run.
    """

    def __init__(self, nb_runner, cell_index):
        self._nb, self._i = nb_runner, cell_index

    def __call__(self) -> int:
        import re
        self._nb.run_cell(self._i)
        out = self._nb.get_output(self._i)
        # The text badge shares this cell's output, so match the marker rather
        # than trusting position.
        m = re.search(r"LIVE (\d+)", out)
        assert m, f"sampler produced no LIVE line:\n{out}"
        return int(m.group(1))


def test_demo_notebook_claims_hold(nb_runner):
    cells = _code_cells()
    assert len(cells) == 9, f"expected 9 code cells, got {len(cells)}"

    # 1 %cash_badge, then the notebook's own cells at 2..10, then a sampler.
    nb_runner.create_notebook([
        "%cash_badge print", *cells,
        "# @cash:no-cache\nprint('LIVE', len(CALLS))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    APPEND_PLAIN, APPEND_DIRECTIVE, FOLD_DIRECTIVE = 5, 7, 9
    live = _Counter(nb_runner, len(cells) + 2)

    # Claim 1: the plain append loop re-runs in full every time.
    before = live()
    nb_runner.run_cell(APPEND_PLAIN)
    assert live() == before + 3, (
        "notebook claims the plain append loop re-runs in full; it did not\n"
        + nb_runner.get_output(APPEND_PLAIN)
    )

    # Claim 2: with the directive, a re-run adds nothing.
    before = live()
    nb_runner.run_cell(APPEND_DIRECTIVE)
    out = nb_runner.get_output(APPEND_DIRECTIVE)
    assert live() == before, (
        f"notebook claims the second run adds no executions:\n{out}"
    )
    assert "[via @cash:cache-calls]" in out, (
        f"notebook tells the reader to look for this tag:\n{out}"
    )
    assert "out2 = [2, 3, 4]" in out, f"the append stopped running:\n{out}"

    # Claim 3: reordering the directive'd fold costs nothing.
    before = live()
    nb_runner.set_cell_source(
        FOLD_DIRECTIVE,
        cells[FOLD_DIRECTIVE - 2].replace("[11, 22, 33]", "[33, 22, 11]"),
    )
    nb_runner.run_cell(FOLD_DIRECTIVE)
    out = nb_runner.get_output(FOLD_DIRECTIVE)
    assert live() == before, f"reordering cost executions:\n{out}"
    assert "SUM 69" in out, f"the reordered fold gave a different answer:\n{out}"

    # Claim 4: the impossible directive warns rather than failing silently.
    assert "matched no cacheable call" in nb_runner.get_output(10), (
        nb_runner.get_output(10)
    )
