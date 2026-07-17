"""CAS-194: ax.hist() must be treated as an in-place Axes draw, not cached.

``.hist()`` returns a ``(counts, bins, BarContainer)`` data tuple, so it used to
slip the receiver-mutation classifier (which keyed on the return type) and was
left un-scheduled during figure reconstruction — ``fig.savefig()`` then flushed a
BLANK histogram over the good chart. The fix keys on the RECEIVER: any method
call on a live matplotlib Axes/Figure draws on it, whatever it returns, so the
fill statement is rebuilt with the figure. A sibling ``.plot()`` panel is the
control — both must survive reconstruction.
"""
import pytest

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.libraries

SETUP = "import matplotlib.pyplot as plt\nimport numpy as np\nimport cash\n%cash_on\n%cash_badge print"


@pytest.mark.timeout(120)
def test_hist_panel_survives_figure_reconstruction(nb_runner, tmp_path):
    chart = (tmp_path / "panel.png").as_posix()
    blank = (tmp_path / "blank.png").as_posix()
    nb_runner.create_notebook([
        SETUP,
        "data = np.arange(500) % 11",
        "fig, (axp, axh) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)",
        "axp.plot(range(20), [i * i for i in range(20)])",   # artist-return control
        "axh.hist(data, bins=11)",                            # data-tuple return
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
    # redrawing BOTH panels before it writes. Under the bug, the fills were
    # omitted and the saved chart came back blank (PATCH: 0).
    nb_runner.set_cell_source(2, "data = np.arange(500) % 7")
    nb_runner.run_cells([2, 6])
    out = nb_runner.get_output(6)
    assert "PATCH: 11" in out, (
        f"the histogram panel blanked on reconstruction: ax.hist() was not "
        f"re-executed with the rebuilt figure (CAS-194). Got:\n{out}"
    )
    assert "LINES: 1" in out, f"control .plot() panel also blanked. Got:\n{out}"

    # The chart on disk must not be a blank two-axes render.
    import numpy as np
    from matplotlib import image as mpimg
    chart_px = mpimg.imread(tmp_path / "panel.png")
    blank_px = mpimg.imread(tmp_path / "blank.png")
    differing = float((np.abs(chart_px - blank_px) > 0.01).any(axis=-1).mean())
    assert differing > 0.01, (
        f"fig.savefig() wrote a blank chart after reconstruction: only "
        f"{differing:.3%} of pixels differ from an empty two-axes render (CAS-194)."
    )
