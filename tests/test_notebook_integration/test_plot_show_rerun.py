"""A cell that both plots (``plt.show()``) and returns a value must not
duplicate the plot when re-run with a changed parameter.

``plt.show()`` flushes pyplot's process-global current figure and has no value
input, so cash used to cache it with a stable key. On a re-run it then served a
STALE figure from cache while the (always re-executed, identity-coupled)
plotting statements drew the new one — producing TWO plots and scrambling the
output order (reported on Colab). ``plt.show()`` is now a display side-effect
(uncacheable), so exactly one plot is produced on every run.
"""

SETUP = (
    "%matplotlib inline\n"
    "import pandas as pd\n"
    "stats = pd.DataFrame({'total': [50, 40, 30, 20, 10]}, index=list('abcde'))"
)


def _plotcell(n):
    return (
        f"TOP_N = {n}\n"
        "top = stats.head(TOP_N)\n"
        "import matplotlib.pyplot as plt\n"
        "ax = top['total'][::-1].plot(kind='barh')\n"
        "ax.set_title(f'Top {TOP_N}')\n"
        "plt.show()\n"
        "top"
    )


def _plot_count(cell):
    return sum(
        1 for o in cell.get("outputs", [])
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})
    )


def test_plt_show_not_duplicated_on_changed_rerun(nb_runner):
    nb_runner.create_notebook([SETUP, _plotcell(3)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _plot_count(nb_runner.get_cell(2)) == 1, "first run should draw exactly one plot"

    # Change the parameter and re-run the plot cell, as a user would.
    nb_runner.set_cell_source(2, _plotcell(5))
    nb_runner.run_cell(2)
    assert _plot_count(nb_runner.get_cell(2)) == 1, (
        "a changed re-run must not replay a stale cached plt.show() figure "
        "alongside the freshly drawn one"
    )
