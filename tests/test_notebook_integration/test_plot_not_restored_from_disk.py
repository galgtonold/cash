"""A plot must never be restored from the DISK cache as a stale/duplicate figure.

The feature-tour chart cell (plot + trailing DataFrame) is run to warm the
on-disk ``.cash``; then the kernel is RESTARTED (fresh session, ``.cash``
persists) and the notebook re-run. The plotting statements (``ax=...plot``,
``plt.tight_layout``, ``plt.show``) are display side-effects, so they always
re-execute and the chart cell shows exactly ONE freshly drawn plot — never a
figure replayed from cache. Guards the "TOP_N shows a stale Top-3 on a warm
session" report.
"""

SETUP = "%matplotlib inline\nimport pandas as pd\nimport matplotlib.pyplot as plt"
AGG = "category_stats = pd.DataFrame({'total_spend': [50, 40, 30, 20, 10]}, index=list('abcde'))"
CHART = (
    "TOP_N = 3\n"
    "top = category_stats.head(TOP_N)\n"
    "ax = top['total_spend'][::-1].plot(kind='barh')\n"
    "ax.set_title(f'Top {TOP_N}')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "top"
)


def _plots(cell):
    return sum(1 for o in cell.get("outputs", [])
               if o.get("output_type") in ("display_data", "execute_result")
               and "image/png" in o.get("data", {}))


def test_plot_not_restored_from_disk_cache(nb_runner):
    nb_runner.create_notebook([SETUP, AGG, CHART])
    nb_runner.start_kernel()
    nb_runner.enable_persist()          # cache even the cheap statements to disk
    nb_runner.run_all()
    assert _plots(nb_runner.get_cell(3)) == 1

    # Fresh kernel, but the on-disk .cash from the first run persists.
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert _plots(nb_runner.get_cell(3)) == 1, (
        "a plot was restored from the disk cache instead of being re-drawn"
    )
