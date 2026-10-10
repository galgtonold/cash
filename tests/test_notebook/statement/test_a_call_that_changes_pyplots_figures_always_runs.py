"""A statement that makes, picks or draws on pyplot's current figure always runs.

pyplot's open figures and its current figure and axes are process-global
state no name or key holds. A hit of ``plt.figure(figsize=(4, 3))`` (the same
text in two cells, so the same key) made no new figure, and the next cell
drew on the previous one. The same for ``sns.barplot(...)`` or ``s.plot()``
drawing on the current axes, a notebook function that calls ``plt.figure``,
an intercepted call that draws, and a loop that does any of these.

Each case runs its cells under ``persist_all`` (every statement stored, so a
cheap ``plt.figure`` is too) and compares pyplot's figures with what the same
cells leave in plain Python.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _agg_and_clean_registry():
    import matplotlib

    matplotlib.use("Agg")
    plt.close("all")
    yield
    plt.close("all")


SETUP = (
    "import matplotlib\nimport matplotlib.pyplot as plt\nimport matplotlib.pyplot as mp\n"
    "def new_figure():\n    plt.figure(figsize=(4, 3))\n"
    "def drawn(k):\n    plt.figure(figsize=(4, 3))\n    plt.plot([1, k])\n    return k * 2\n"
)

NEW_FIGURE = "plt.figure(figsize=(4, 3))\nplt.plot([1])"

CASES = {
    "plt.figure": ["plt.figure(figsize=(4, 3))\nplt.plot([1, 2])"] * 2,
    "unbound plt.subplots": ["plt.subplots(figsize=(4, 3))\nplt.plot([1, 2])"] * 2,
    "plt.subplot": [
        NEW_FIGURE,
        "plt.subplot(2, 1, 1)\nplt.plot([1, 2])",
        "plt.subplot(2, 1, 2)\nplt.plot([1, 2])",
        "plt.subplot(2, 1, 1)\nplt.plot([1, 2])",
    ],
    "plt.axes": [NEW_FIGURE] + ["plt.axes([0.1, 0.1, 0.3, 0.3])\nplt.plot([1, 2])"] * 2,
    "plt.gca drawn on": [NEW_FIGURE] + ["plt.gca().plot([1, 2])\nprint(1)"] * 2,
    "matplotlib.pyplot spelled out": ["matplotlib.pyplot.figure(figsize=(4, 3))\nplt.plot([1, 2])"] * 2,
    "another alias": ["mp.figure(figsize=(4, 3))\nmp.plot([1, 2])"] * 2,
    "a notebook function": ["new_figure()\nplt.plot([1, 2])"] * 2,
    "an intercepted call that draws": ["v = drawn(3)\nprint(v)"] * 2,
    "plt.close": [NEW_FIGURE, NEW_FIGURE, "plt.close()\nprint(1)", "plt.close()\nprint(1)"],
}

PANDAS_CASES = {
    "pandas plot": ["plt.figure(figsize=(4, 3))\npd.Series([1, 2]).plot()\nprint(1)"] * 2,
}

SEABORN_CASES = {
    "seaborn on the current axes": ["plt.figure(figsize=(4, 3))\nsns.barplot(x=['a', 'b'], y=[1, 2])\nprint(1)"] * 2,
    "a loop": ["for k in [1, 2]:\n    plt.figure(figsize=(4, 3))\n    sns.barplot(x=['a', 'b'], y=[1, k])"] * 2,
    "a loop drawing through seaborn only": [
        "plt.figure(figsize=(4, 3))",
        "for k in [1, 2]:\n    sns.barplot(x=['a', 'b'], y=[1, k])\n    print(k)",
    ]
    * 2,
}


def _figures():
    """Each open figure's axes and what they hold, and which is current."""
    held = [(n, [(len(ax.lines), len(ax.patches)) for ax in plt.figure(n).axes]) for n in plt.get_fignums()]
    return held, plt.gcf().number if plt.get_fignums() else None


def _plain(setup, cells):
    plt.close("all")
    ns: dict = {}
    exec(setup, ns)
    for cell in cells:
        exec(cell, ns)
    figures = _figures()
    plt.close("all")
    return figures


def _check(cash_magics, setup, cells):
    plain = _plain(setup, cells)
    cash_magics.cash_persist("on")
    run_cash_cell(cash_magics, setup)
    for cell in cells:
        run_cash_cell(cash_magics, cell)
    assert _figures() == plain


@pytest.mark.parametrize("cells", list(CASES.values()), ids=list(CASES))
def test_pyplot_figures_are_as_in_plain_python(cells, cash_magics, mock_shell):
    _check(cash_magics, SETUP, cells)


@pytest.mark.parametrize("cells", list(PANDAS_CASES.values()), ids=list(PANDAS_CASES))
def test_a_pandas_drawing_runs_every_time(cells, cash_magics, mock_shell):
    pytest.importorskip("pandas")
    _check(cash_magics, SETUP + "import pandas as pd\n", cells)


@pytest.mark.parametrize("cells", list(SEABORN_CASES.values()), ids=list(SEABORN_CASES))
def test_a_seaborn_drawing_runs_every_time(cells, cash_magics, mock_shell):
    pytest.importorskip("seaborn")
    _check(cash_magics, SETUP + "import seaborn as sns\n", cells)


def test_a_statement_that_leaves_the_figures_alone_is_still_served(cash_magics, statement_processor, mock_shell):
    """The positive control: with figures open, a statement that does not
    touch them is stored and served as before."""
    from cash.notebook.cache_status import CacheStatus

    cash_magics.cash_persist("on")
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, NEW_FIGURE)
    first = statement_processor.process_statement("total = sum(range(1000))")
    second = statement_processor.process_statement("total = sum(range(1000))")
    assert first["status"] == CacheStatus.COMPUTED
    assert second["status"] in (CacheStatus.RESTORED, CacheStatus.SKIPPED)
