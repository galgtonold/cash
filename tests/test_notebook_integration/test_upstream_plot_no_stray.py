"""A reconstructed UPSTREAM plot must not leak into a downstream cell's output.

When a downstream cell needs a ``fig``/``ax`` produced by an upstream plot cell,
cash re-executes that plot cell to rebuild the object. Re-executing it opens a
matplotlib figure, which the inline backend's post-execute hook would otherwise
flush into the (non-plotting) downstream cell — a stray plot. cash now closes
figures it opens during reconstruction (a normally-run cell closes its figures
on flush anyway), so the downstream cell shows only its own output while still
getting a usable ``ax``.
"""

SETUP = "%matplotlib inline\nimport matplotlib.pyplot as plt"
def DATA(n): return f"n = {n}"
PLOT = (
    "fig, ax = plt.subplots()\n"
    "ax.bar(range(n), range(n))\n"
    "ax.set_title('bars')\n"
    "plt.show()"
)
DOWN = "print('AXTITLE', ax.get_title(), 'N', n)"


def _plot_count(cell):
    return sum(
        1 for o in cell.get("outputs", [])
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})
    )


def _text(cell):
    return "".join(
        o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    )


def test_upstream_plot_not_leaked_into_downstream(nb_runner):
    nb_runner.create_notebook([SETUP, DATA(5), PLOT, DOWN])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _plot_count(nb_runner.get_cell(3)) == 1, "the plot cell should have its plot"
    assert _plot_count(nb_runner.get_cell(4)) == 0, "the downstream cell should have no plot"

    # Edit the data and run ONLY the downstream cell -> cash rebuilds `ax` by
    # re-executing the plot cell.
    nb_runner.set_cell_source(2, DATA(8))
    nb_runner.run_cell(4)

    # It must recompute correctly using the reconstructed `ax`...
    down_text = _text(nb_runner.get_cell(4))
    assert "N 8" in down_text and "bars" in down_text, down_text
    # ...without the reconstructed figure leaking in as a stray plot.
    assert _plot_count(nb_runner.get_cell(4)) == 0, (
        "the reconstructed upstream figure leaked into the downstream cell"
    )
