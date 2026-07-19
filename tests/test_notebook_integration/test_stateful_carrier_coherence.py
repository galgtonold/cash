"""CAS-175 / CAS-178: the sim must never re-execute a SUBSET of a carrier's history.

Both tickets are one defect. The dependency model tracks **value lineage**; a
statement can depend on a predecessor through an object's *internal* state --
a Generator's position, a Figure's accumulated artists -- with no value-level
edge. Re-executing the consumer without the state-establishing statement yields
a result matching **no possible execution of the notebook**:

    [7] OMIT  rng = np.random.default_rng(7)     <- establishes the position
    [8] RUN   steps = rng.standard_normal(n)     <- redraws from an ADVANCED rng

    [9]  RUN   fig, ax = plt.subplots()          <- fresh BLANK figure
    [10] OMIT  ax.bar(names, totals)             <- fills it
    [12] RUN   fig.savefig(path)                 <- writes the BLANK one to disk

Both assert against a GROUND TRUTH rendered outside cash, because both bugs are
invisible from inside the notebook: the RNG returns a plausible float, and the
chart corruption is only observable by reading the PNG from outside the kernel.

Note on scope. Carrier-history completion only ever fires for a write the plan
ALREADY schedules, and reconstruction is scoped to files a relevant consumer
reads (CAS-193/196/200) -- a writer whose output nothing reads is a terminal
side effect and is never re-fired. So the chart tests below come in a pair: one
drives the redraw through a genuine reader of the PNG, the other pins that an
unrelated cell leaves the artifact alone even after an upstream edit. Demanding
a redraw from an unrelated cell would contradict that scoping directly, and the
contradiction is not academic -- lifting the gate duplicates a line in a
``mode='a'`` audit log, corrupting a file rather than staling one.
"""
import hashlib

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(120)]


# ----------------------------------------------------------------------
# CAS-178 -- seeded RNG redrawn from an already-advanced generator
# ----------------------------------------------------------------------

def test_seeded_rng_redraw_after_upstream_edit_matches_truth(nb_runner):
    """Editing an upstream param must NOT redraw from an advanced generator.

    ``rng`` carries no value-level edge to ``n``, so editing ``n`` schedules
    ``steps = rng.standard_normal(n)`` while leaving ``rng = default_rng(7)``
    behind -- and the live ``rng`` is still advanced by the previous run's draw.

    The two constants are the whole point. ``truth`` is what a live kernel (and
    ``run_all``) produces; ``advanced`` is the exact arithmetic of redrawing from
    the already-advanced generator, which is what cash returned. Asserting only
    ``== truth`` would be enough to catch it, but pinning ``!= advanced``
    documents the failure mode: cash's answer was not merely stale, it
    corresponded to no execution of the notebook at all.
    """
    np = pytest.importorskip("numpy")

    truth = float(np.random.default_rng(7).standard_normal(6).sum())
    _spent = np.random.default_rng(7)
    _spent.standard_normal(5)                       # the first run's draw
    advanced = float(_spent.standard_normal(6).sum())
    assert truth != advanced                        # the probe itself is sound

    nb_runner.create_notebook([
        "import numpy as np\nimport cash\n%cash_on\n%cash_badge print",     # 1
        "n = 5",                                                            # 2
        "rng = np.random.default_rng(7)\nsteps = rng.standard_normal(n)",   # 3
        "# @cash:no-cache\nprint('RESULT', repr(float(steps.sum())))",      # 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "RESULT" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "n = 6")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    value = float(out.split("RESULT")[1].split()[0])

    assert value != pytest.approx(advanced), (
        f"cash redrew from an ALREADY-ADVANCED generator: got {value!r}, which is "
        f"exactly the advanced-redraw value -- a number no execution of this "
        f"notebook can produce (truth {truth!r}). The sim re-executed "
        f"'steps = rng.standard_normal(n)' without 'rng = np.random.default_rng(7)'."
    )
    assert value == pytest.approx(truth, abs=0.0), (
        f"cash did not reproduce a live kernel bit-for-bit: {value!r} != {truth!r}"
    )


def test_seeded_rng_unedited_notebook_is_not_disturbed(nb_runner):
    """The pass must not change a notebook that needs no reconstruction.

    The carrier trigger only ever fires for a statement the plan ALREADY
    schedules, so a plain ``run_all`` -- where nothing is broken -- must still
    produce the ordinary seeded value with no re-derivation.
    """
    np = pytest.importorskip("numpy")
    truth = float(np.random.default_rng(7).standard_normal(5).sum())

    nb_runner.create_notebook([
        "import numpy as np\nimport cash\n%cash_on\n%cash_badge print",
        "n = 5",
        "rng = np.random.default_rng(7)\nsteps = rng.standard_normal(n)",
        "# @cash:no-cache\nprint('RESULT', repr(float(steps.sum())))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert float(out.split("RESULT")[1].split()[0]) == pytest.approx(truth, abs=0.0), out


# ----------------------------------------------------------------------
# CAS-175 -- builder rebuilt + re-saved without the statements that fill it
# ----------------------------------------------------------------------

def _render_truth(path, totals, with_bars=True):
    """Render the reference chart directly, outside cash, in this process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if with_bars:
        ax.bar(["a", "b", "c"], totals)
        ax.set_title("Totals")
    fig.savefig(str(path))
    plt.close(fig)
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_builder_chart_on_disk_survives_an_unrelated_cell(nb_runner, tmp_path):
    """Running an unrelated cell must not overwrite a saved chart with a blank one.

    ``fig.savefig`` is scheduled (it writes) and ``fig, ax = plt.subplots()`` is
    scheduled (it produces the writer's input) -- but ``ax.bar`` / ``ax.set_title``
    are not, because they fill the figure through ``ax``, whose only tie to ``fig``
    is their CO-PRODUCTION by ``plt.subplots()``. So cash rebuilt a blank figure
    and flushed it over the good chart.

    Asserted against a truth render rather than a size threshold: the blank
    artifact is *larger* than the real chart here (7442 B vs 5999 B), so any
    "smaller means blank" heuristic reads exactly backwards. The corruption is
    also invisible from inside the notebook -- ``ax``/``fig`` look right in the
    kernel afterwards -- so the PNG bytes on disk are the only honest oracle.
    """
    pytest.importorskip("matplotlib")

    chart = tmp_path / "chart.png"
    truth_bars = _render_truth(tmp_path / "truth_bars.png", [3, 5, 2])
    truth_blank = _render_truth(tmp_path / "truth_blank.png", [3, 5, 2], with_bars=False)
    assert truth_bars != truth_blank

    nb_runner.create_notebook([
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "import cash\n%cash_on\n%cash_badge print",                     # 1
        "names = ['a', 'b', 'c']\ntotals = [3, 5, 2]",                  # 2
        f"fig, ax = plt.subplots()\nax.bar(names, totals)\n"            # 3
        f"ax.set_title('Totals')\nfig.savefig(r'{chart.as_posix()}')",
        "grand_total = sum(totals)",                                    # 4 (unrelated)
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    after_runall = hashlib.md5(chart.read_bytes()).hexdigest()
    assert after_runall != truth_blank, (
        "run_all itself wrote a BLANK chart: cell 4's upstream sim rebuilt the "
        "figure and re-saved it without the ax.bar/ax.set_title that fill it"
    )
    assert after_runall == truth_bars, "run_all did not write the real bar chart"

    # THE assertion: an unrelated cell must not touch the user's artifact.
    nb_runner.run_cell(4)
    after_unrelated = hashlib.md5(chart.read_bytes()).hexdigest()
    assert after_unrelated != truth_blank, (
        "running an UNRELATED cell silently overwrote the chart with a blank "
        "image (CAS-175)"
    )
    assert after_unrelated == after_runall, (
        "running an unrelated cell changed the chart on disk"
    )


def test_builder_edit_redraws_coherently_when_the_chart_is_consumed(nb_runner, tmp_path):
    """The repair must RE-DRAW, not merely refuse -- wherever the write is in scope.

    The complement to the guard above, and the ticket's second acceptance
    bullet: *if* the sim re-executes a write-performing statement, it must also
    re-execute every statement that contributed to the written object's state.
    A fix that simply declined to re-run the builder would leave the stale
    [3, 5, 2] bars on disk and still pass a "not blank" check.

    The consuming cell READS the chart, which is what puts the write in scope.
    Reconstruction is scoped to files a relevant consumer actually reads
    (CAS-193/196/200): a writer whose output nothing reads is a terminal side
    effect and is deliberately never re-fired, so an *unrelated* cell cannot
    reach this path at all -- see the companion test below, which pins that.
    Asserting the redraw through a genuine reader is therefore the only way to
    exercise carrier-history completion without demanding the very re-fire the
    scope gate exists to prevent.
    """
    pytest.importorskip("matplotlib")

    chart = tmp_path / "chart.png"
    truth_new = _render_truth(tmp_path / "truth_new.png", [3, 5, 9])
    truth_old = _render_truth(tmp_path / "truth_old.png", [3, 5, 2])
    truth_blank = _render_truth(tmp_path / "truth_blank.png", [3, 5, 2], with_bars=False)
    assert truth_new != truth_old != truth_blank

    nb_runner.create_notebook([
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "import cash\n%cash_on\n%cash_badge print",
        "names = ['a', 'b', 'c']\ntotals = [3, 5, 2]",
        f"fig, ax = plt.subplots()\nax.bar(names, totals)\n"
        f"ax.set_title('Totals')\nfig.savefig(r'{chart.as_posix()}')",
        # A report cell that quotes the total AND embeds the chart, so the
        # savefig's output is read by a consumer this reconstruction needs.
        f"grand_total = sum(totals)\n"
        f"chart_size = len(open(r'{chart.as_posix()}', 'rb').read())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert hashlib.md5(chart.read_bytes()).hexdigest() == truth_old

    nb_runner.set_cell_source(2, "names = ['a', 'b', 'c']\ntotals = [3, 5, 9]")
    nb_runner.run_cell(4)
    redrawn = hashlib.md5(chart.read_bytes()).hexdigest()
    assert redrawn != truth_blank, (
        "the re-derived chart is BLANK: fig.savefig was re-executed without the "
        "ax.bar/ax.set_title that fill the figure (CAS-175)"
    )
    assert redrawn == truth_new, (
        "the re-derived chart does not reflect the edited data -- the carrier's "
        "history was re-executed incoherently"
    )


def test_unrelated_cell_leaves_the_chart_alone_even_after_an_edit(nb_runner, tmp_path):
    """An upstream edit does not license re-firing a write no consumer reads.

    Pins the boundary the test above stops at. Editing ``totals`` and running a
    cell that merely sums it must leave the chart byte-identical: cell 3 is a
    SIBLING consumer of ``totals``, not an ancestor of cell 4, and a live kernel
    would not re-run it either. The PNG is then stale in exactly the ordinary
    notebook sense -- the user re-runs the plot cell -- rather than silently
    rewritten behind their back.

    This is the direction CAS-193/196/200 settled, and it is not merely
    stylistic: re-firing out-of-scope writers duplicates a line in a
    ``mode='a'`` audit log, which corrupts a file rather than staling one.
    """
    pytest.importorskip("matplotlib")

    chart = tmp_path / "chart.png"
    nb_runner.create_notebook([
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "import cash\n%cash_on\n%cash_badge print",
        "names = ['a', 'b', 'c']\ntotals = [3, 5, 2]",
        f"fig, ax = plt.subplots()\nax.bar(names, totals)\n"
        f"ax.set_title('Totals')\nfig.savefig(r'{chart.as_posix()}')",
        "# @cash:no-cache\ngrand_total = sum(totals)\nprint('TOTAL', grand_total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    before = chart.read_bytes()
    assert "TOTAL 10" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "names = ['a', 'b', 'c']\ntotals = [3, 5, 9]")
    nb_runner.run_cell(4)

    assert chart.read_bytes() == before, (
        "running an unrelated cell rewrote the user's chart after an upstream "
        "edit; reconstruction must not re-fire a write no consumer reads "
        "(CAS-193/196/200)"
    )
    # Suppressing the out-of-scope WRITE must not suppress the value the cell
    # genuinely depends on: the edit still has to reach grand_total.
    assert "TOTAL 17" in nb_runner.get_output(4), (
        "the upstream edit did not reach the cell that consumes it: "
        f"{nb_runner.get_output(4)!r}"
    )
