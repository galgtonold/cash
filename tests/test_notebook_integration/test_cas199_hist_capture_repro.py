"""CAS-199: a draw whose return is CAPTURED must still re-execute on a warm run.

CAS-194 fixed the bare form ``ax.hist(data)`` -- a method call on a live Axes
draws on it whatever it returns, so it is re-executed, not cached. But the
classifier only inspected bare-``Expr`` method calls, so the far more common
CAPTURED form ::

    counts, bins, patches = ax.hist(data, bins=50)

-- an ``ast.Assign`` whose RHS is the draw -- slipped straight through: the
tuple was cached and, on a warm re-run, CACHED while the draw (the bars on the
Axes) was never replayed. ``fig.savefig()`` then flushed a BLANK histogram, with
a green badge and ``counts`` printing fine -- silent. The fix routes an
identity-coupled (live Axes/Figure) receiver as a draw even when its return is
captured, keyed on the RECEIVER (not the return type, not the statement shape),
so a genuine pure capture (``m = df.corr()``, DataFrame receiver) still caches.
"""
import pytest
from conftest import shows_cached

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.libraries

SETUP = "import matplotlib.pyplot as plt\nimport numpy as np\nimport cash\n%cash_on\n%cash_badge print"


@pytest.mark.timeout(120)
def test_captured_hist_survives_figure_reconstruction(nb_runner, tmp_path):
    """``counts, bins, patches = ax.hist(...)`` must redraw on reconstruction.

    Same reconstruction trigger as the CAS-194 bare-form test, but with the
    return CAPTURED into an assignment. The saver cell consumes only ``axh``
    (via ``len(axh.patches)``), NOT the captured tuple -- so nothing forces the
    hist to re-run by data-dependency; the figure's reconstruction must redraw
    it from its receiver-mutation lineage. Under the bug the captured hist was
    cached, never part of that lineage, so editing the data and re-saving left
    the rebuilt Axes empty (PATCH: 0) -- a blank histogram with a green badge.
    The bare control ``axp.plot(...)`` (CAS-194) is the survivor comparison.
    """
    chart = (tmp_path / "panel.png").as_posix()
    blank = (tmp_path / "blank.png").as_posix()
    nb_runner.create_notebook([
        SETUP,
        "data = np.arange(500) % 11",
        "fig, (axp, axh) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)",
        "axp.plot(range(20), [i * i for i in range(20)])",       # artist-return control (CAS-194)
        "counts, bins, patches = axh.hist(data, bins=11)",       # CAPTURED data-tuple return
        f"fig.savefig('{chart}')\n"
        "print('LINES:', len(axp.lines), 'PATCH:', len(axh.patches))",
        # A same-geometry blank baseline for the pixel check.
        f"figb, axb = plt.subplots(1, 2, figsize=(8, 3), dpi=100)\n"
        f"figb.savefig('{blank}')\nprint('blank saved')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "LINES: 1 PATCH: 11" in nb_runner.get_output(6)

    # Edit the histogram's data -> the saver must reconstruct the whole figure,
    # redrawing BOTH panels before it writes. Under the bug the captured hist was
    # omitted from the figure's derivation and the rebuilt Axes stayed empty
    # (PATCH: 0), blanking the saved chart.
    nb_runner.set_cell_source(2, "data = np.arange(500) % 7")
    nb_runner.run_cells([2, 6])
    out = nb_runner.get_output(6)
    assert "PATCH: 11" in out, (
        f"the captured-return histogram blanked on reconstruction: "
        f"`counts, bins, _ = ax.hist(...)` was cached and its draw skipped "
        f"(CAS-199). Got:\n{out}"
    )
    assert "LINES: 1" in out, f"control .plot() panel also blanked. Got:\n{out}"

    # The chart on disk must not be a blank two-axes render.
    import numpy as np
    from matplotlib import image as mpimg
    chart_px = mpimg.imread(tmp_path / "panel.png")
    blank_px = mpimg.imread(tmp_path / "blank.png")
    differing = float((np.abs(chart_px - blank_px) > 0.01).any(axis=-1).mean())
    assert differing > 0.01, (
        f"fig.savefig() wrote a blank chart on the warm re-run: only "
        f"{differing:.3%} of pixels differ from an empty two-axes render (CAS-199)."
    )


@pytest.mark.timeout(120)
def test_captured_pie_sibling_survives_reconstruction(nb_runner, tmp_path):
    """Sibling audit: ``wedges, texts = ax.pie(...)`` returns a data tuple too.

    The same receiver rule covers it (identity-coupled Axes receiver), so no
    per-method list is needed. Here the saver consumes only ``ax`` (not the
    captured tuple), so editing the data and re-saving forces the figure's
    reconstruction path to redraw the pie from its receiver-mutation lineage.
    Under the bug the captured pie was cached, never part of that lineage, and
    its wedges were never redrawn (WEDGES: 0).
    """
    chart = (tmp_path / "pie.png").as_posix()
    nb_runner.create_notebook([
        SETUP,
        "sizes = [30, 20, 50]",
        "fig, ax = plt.subplots(figsize=(3, 3), dpi=100)",
        "wedges, texts = ax.pie(sizes)",                         # CAPTURED tuple return
        f"fig.savefig('{chart}')\nprint('WEDGES:', len(ax.patches))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "WEDGES: 3" in nb_runner.get_output(5)

    nb_runner.set_cell_source(2, "sizes = [10, 40, 25, 25]")
    nb_runner.run_cells([2, 5])
    out = nb_runner.get_output(5)
    assert "WEDGES: 4" in out, (
        f"the captured-return pie blanked on reconstruction: `wedges, texts = "
        f"ax.pie(...)` was cached and its wedges never redrawn (CAS-199). Got:\n{out}"
    )


@pytest.mark.timeout(120)
def test_pure_capture_on_ordinary_receiver_still_caches(nb_runner):
    """No over-invalidation: ``m = df.corr()`` (DataFrame receiver) still caches.

    The discriminator is the identity-coupled receiver check, NOT "the RHS is a
    method call". A DataFrame is not identity-coupled, so a captured pure read
    must stay on the caching path (CACHED on warm re-run), never routed to
    skip-cache. ``corr()`` on a wide frame is expensive enough that the cost
    model keeps it, so CACHED is a real signal (a cheap op the cost model
    declines would make the assertion vacuous).
    """
    nb_runner.create_notebook([
        "import pandas as pd\nimport numpy as np\nimport cash\n%cash_on\n%cash_badge print",
        "df = pd.DataFrame(np.random.RandomState(0).randn(300000, 40))",
        "m = df.corr()\nprint('SHAPE:', m.shape)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cells([3])
    out = nb_runner.get_output(3)
    assert shows_cached(out), (
        f"a captured pure read `m = df.corr()` stopped caching -- the CAS-199 "
        f"draw rule over-reached onto an ordinary (non-Axes) receiver. Got:\n{out}"
    )
    assert "In-place mutation" not in out, (
        f"`m = df.corr()` was wrongly routed to skip-cache as a mutation -- the "
        f"CAS-199 rule must fire ONLY for identity-coupled receivers. Got:\n{out}"
    )
