"""CAS-187: the sim must never re-save pyplot's current figure orphaned from its producer.

``plt.savefig(path)`` writes pyplot's process-global current figure; its only
variable input is the module ``plt``, so -- unlike the receiver-bound
``fig.savefig(path)`` that CAS-175 defends -- there is NO value-level edge for
the planner to follow to the figure. If the plan ever schedules the write while
the ``plt.subplots()`` that registered the current figure is NOT scheduled,
re-running the write makes ``plt.gcf()`` invent a blank default figure and flush
it over the user's chart: a silent on-disk wrong answer (proven by the geometry
collapsing from the user's 960x540 to matplotlib's 640x480 default).

The defect is LATENT on main -- the file-writer scheduling pass' provenance
guard keeps a plain re-run from orphaning the write -- but it is one unrelated
change away (CAS-174 induced it by accident). This guard bounds the consequence:
refuse the orphaned write and warn, rather than write a figure the user did not
draw. The oracle is the PNG geometry on disk; the corruption is invisible from
inside the kernel.
"""
import os

import pytest

pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.upstream, pytest.mark.libraries, pytest.mark.timeout(120)]

# Clears the session-scoped record of executed file-writers, reproducing the
# post-restart state in which the file-writer pass sees ``plt.savefig`` as
# never-run-this-session. Combined with an unresolvable ``os.path.join`` path
# (which the provenance freshness guard cannot verify) this is what schedules
# the write while its ``plt.subplots()`` producer stays out of the plan.
_CLEAR_WRITE_RECORD = (
    "# @cash:no-cache\n"
    "_ts = get_ipython().magics_manager.registry.get('CashMagics')._tracking_state\n"
    "_ts.executed_write_stmt_codes.clear()\n"
    "print('write-record cleared')"
)


def _png_geometry(path):
    from matplotlib import image as mpimg
    h, w = mpimg.imread(path).shape[:2]
    return (w, h)


def _chart_path(tmp_path):
    # Short path (Windows MAX_PATH hygiene) with a unique name per test.
    p = f"C:/Temp/cas187_{os.getpid()}_{id(tmp_path) & 0xffff}.png"
    if os.path.exists(p):
        os.remove(p)
    return p


def test_healthy_plt_savefig_writes_the_real_chart_and_does_not_refuse(nb_runner, tmp_path):
    """The healthy path: a normal run draws + saves the real chart; the guard,
    which only fires when the producer is missing from the plan, is a no-op."""
    os.makedirs("C:/Temp", exist_ok=True)
    chart = _chart_path(tmp_path)

    nb_runner.create_notebook([
        "import os\nimport matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\nimport cash\n%cash_on\n%cash_badge print",
        "names = ['a', 'b', 'c']\ntotals = [3, 5, 2]",
        "fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)\n"
        "ax.bar(names, totals)\nax.set_title('Totals')\n"
        f"plt.savefig(r'{chart}')",
        "grand_total = sum(totals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # The user's geometry, not matplotlib's 640x480 default -> a real chart.
    assert _png_geometry(chart) == (960, 540), (
        "the healthy run must write the user's figure at its own geometry"
    )

    # A plain unrelated re-run must not disturb it either.
    nb_runner.run_cell(4)
    assert _png_geometry(chart) == (960, 540)
    os.remove(chart)


def test_orphaned_plt_savefig_is_refused_not_blanked(nb_runner, tmp_path):
    """The vulnerable shape, forced: the write is scheduled without its producer.

    Without the guard, re-running the unrelated cell overwrites the 960x540 chart
    with a 640x480 blank (``plt.gcf()`` invents a default figure). The guard must
    refuse the orphaned write, leaving the real chart untouched on disk.
    """
    os.makedirs("C:/Temp", exist_ok=True)
    chart = _chart_path(tmp_path)

    nb_runner.create_notebook([
        "import os\nimport matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\nimport cash\n%cash_on\n%cash_badge print",  # 1
        "d = r'C:/Temp'\nnames = ['a', 'b', 'c']\ntotals = [3, 5, 2]",                 # 2
        "fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)\n"                          # 3
        "ax.bar(names, totals)\nax.set_title('Totals')\n"
        f"plt.savefig(os.path.join(d, {os.path.basename(chart)!r}))\nplt.close('all')",
        "grand_total = sum(totals)",                                                   # 4 unrelated
        _CLEAR_WRITE_RECORD,                                                           # 5
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _png_geometry(chart) == (960, 540), "run_all must produce the real chart first"

    nb_runner.run_cell(5)   # model the post-restart write-record state
    nb_runner.run_cell(4)   # unrelated cell -> its sim would orphan the savefig

    assert _png_geometry(chart) == (960, 540), (
        "cash silently overwrote the user's chart with a blank 640x480 default "
        "figure: the sim scheduled plt.savefig() without the plt.subplots() that "
        "registered the current figure (CAS-187). The guard must refuse the write."
    )
    os.remove(chart)
