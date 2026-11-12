"""An edited cell reading only a pipeline keeps the output a cell below fitted.

The shape of a housing chapter: build a pipeline, fit it and transform the
data into ``prepared``, a cell reading the pipeline, then a later section that
fits the same pipeline again with the same statement text on new data. The
reader cell, edited and re-run alone, reads the pipeline as the cells above
leave it, so the fit above runs again and binds ``prepared`` on the way. The
cell does not read ``prepared``: afterwards it holds what the cell below made,
as in plain Jupyter, not the older value from above.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

FIT = "prepared = pipe.fit_transform(data)"


def _cells(reader: str) -> list[str]:
    return [
        "import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler",
        "data = np.array([[1.0], [2.0], [3.0]])",
        "pipe = make_pipeline(StandardScaler())",
        FIT,
        reader,
        "data = np.array([[10.0], [20.0], [30.0], [40.0]])",
        # The same statement; the cell differs, so the harness, which sends
        # no cell ids, can tell the two cells apart.
        f"{FIT}\nprepared.shape",
    ]


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for src in cells:
        exec(compile(src, "<plain>", "exec"), ns)
    return ns


def test_the_edited_reader_keeps_the_fit_below(nb_runner):
    before = "mean = float(pipe[0].mean_[0])"
    after = "mean = round(float(pipe[0].mean_[0]), 3)"
    nb_runner.create_notebook(_cells(before))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("prepared.shape") == "(4, 1)"
    nb_runner.set_cell_source(5, after)
    nb_runner.run_cell(5)
    # What the cell reads: the pipeline a run from the top has there.
    assert nb_runner.peek("mean") == repr(_plain(_cells(after)[:5])["mean"])
    # What it leaves: plain Jupyter re-running the cell keeps the fit below.
    assert nb_runner.peek("prepared.shape") == "(4, 1)"
