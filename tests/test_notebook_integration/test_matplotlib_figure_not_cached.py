"""CAS-144: caching a matplotlib Figure/Axes makes plt.savefig() write a blank image.

``fig, ax = plt.subplots()`` used to be cached like any other assignment.  The
RAM tier deep-copies every value it stores, and ``Figure.__getstate__`` records
that the figure was registered with pyplot -- so ``Figure.__setstate__`` on the
COPY calls ``Gcf._set_new_active_manager`` and makes *the cache's private
snapshot* pyplot's current figure.  The user then draws on their own figure
while ``plt.savefig()`` writes the snapshot: a blank PNG, on the FIRST run, with
no error and no badge complaint.

The fix refuses to cache Figure/Axes outright -- they are identity-coupled to
pyplot's globals and cost ~0.04s to build, so caching them is all risk and no
reward.  These tests pin the user-visible contract, not the implementation.
"""
import pytest

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.libraries

SETUP = "import matplotlib.pyplot as plt\nimport cash\n%cash_on\n%cash_badge print"


@pytest.mark.timeout(90)
def test_savefig_writes_the_real_chart_not_a_blank_image(nb_runner, tmp_path):
    """The acceptance test: the deliverable PNG has the bars the user drew.

    Compared against a *blank baseline* rendered in the same kernel at the same
    figsize/dpi, per-pixel.  Rationale for not asserting bytes: a byte-exact
    match across runs is brittle (matplotlib metadata), and file size is a
    genuinely weak discriminator here -- a 4-bar chart is only ~1.3x an
    empty-axes render, so any size threshold would be arbitrary.  Pixels ask the
    real question directly: "is this image blank?".

    The bug's artifact is exactly an empty-axes figure at the user's dimensions
    (the deep-copy snapshot is taken *before* ``ax.bar()`` runs), so under the
    bug ``chart.png`` renders identically to the baseline and the differing
    fraction collapses to ~0.  Both files come from one matplotlib install, so
    version drift cancels out.
    """
    chart = (tmp_path / "chart.png").as_posix()
    blank = (tmp_path / "blank.png").as_posix()

    nb_runner.create_notebook([
        SETUP,
        # The exact idiom from the bug report: draw on ax, save via pyplot's global.
        "fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)\n"
        "ax.bar(['ent/low', 'ent/med', 'pro/low', 'free/high'], [103399, 24212, 24095, 812])\n"
        f"plt.savefig('{chart}')\n"
        "print('chart saved')",
        # Baseline: same geometry, axes but no data. fig.savefig() is object-bound,
        # so this renders the same whether or not the bug is present.
        "figb, axb = plt.subplots(figsize=(8, 4.5), dpi=120)\n"
        f"figb.savefig('{blank}')\n"
        "print('blank saved')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    import numpy as np
    from matplotlib import image as mpimg

    chart_px = mpimg.imread(tmp_path / "chart.png")
    blank_px = mpimg.imread(tmp_path / "blank.png")
    assert chart_px.shape == blank_px.shape, "baseline geometry must match the chart"

    differing = float((np.abs(chart_px - blank_px) > 0.01).any(axis=-1).mean())
    assert differing > 0.01, (
        f"plt.savefig() wrote a blank chart under %cash_on: only {differing:.3%} of "
        f"pixels differ from an empty-axes render, so the bars the user drew are "
        f"missing. Cash detached pyplot's current figure from the user's (CAS-144)."
    )


@pytest.mark.timeout(90)
def test_user_figure_is_still_pyplots_current_figure(nb_runner):
    """The crisp invariant: caching must not sever `fig is plt.gcf()`.

    This is the whole bug in one line -- it flipped True->False purely by
    turning cash on, on the first run.
    """
    nb_runner.create_notebook([
        SETUP,
        "fig9, ax9 = plt.subplots(figsize=(3, 2))\n"
        "ax9.plot([1, 2, 3], [1, 4, 9])\n"
        "print('IS_GCF:', fig9 is plt.gcf())\n"
        "print('LINES_ON_SAVED_FIG:', len(plt.gcf().axes[0].lines) if plt.gcf().axes else 0)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "IS_GCF: True" in output, f"fig is plt.gcf() must hold under %cash_on. Got:\n{output}"
    # The figure plt.savefig() would write must carry the user's line.
    assert "LINES_ON_SAVED_FIG: 1" in output, f"pyplot's current figure lost the user's data. Got:\n{output}"


@pytest.mark.timeout(90)
def test_ordinary_statements_still_cache_alongside_a_figure(nb_runner):
    """Control: refusing Figures must not disable caching generally.

    The work has to be genuinely slow *and* bind a name, and ``slow`` must be
    defined in its own cell *after* ``%cash_on`` so cash tracks its lineage: the
    cost model declines to store a statement too cheap to be worth restoring, a
    generator is a consumable that never caches, and an untracked callee yields
    "Input variable missing lineage" -- any of the three would make this control
    vacuous rather than protective.
    """
    nb_runner.create_notebook([
        "import matplotlib.pyplot as plt\nimport cash\nimport time\n%cash_on\n%cash_badge print",
        "def slow(i):\n    time.sleep(0.05)\n    return i * i",
        "vals = [slow(i) for i in range(6)]\nprint('VALS:', vals)",
        "fig, ax = plt.subplots()\nprint('made fig')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run the ordinary statement: it must come back from cache.
    nb_runner.run_cells([3])
    output = nb_runner.get_output(3)
    assert "RESTORED" in output, (
        f"An ordinary cached statement stopped caching -- the CAS-144 rule over-reached. Got:\n{output}"
    )


@pytest.mark.timeout(90)
def test_badge_tells_the_truth_about_the_refused_figure(nb_runner):
    """Control: the refusal is surfaced with an honest reason, not silent."""
    nb_runner.create_notebook([
        SETUP,
        "fig, ax = plt.subplots()\nprint('made fig')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "NOT CACHED" in output, f"The refused figure must show as NOT CACHED. Got:\n{output}"
    assert "Identity-coupled" in output, f"The badge must say WHY the figure was refused. Got:\n{output}"
