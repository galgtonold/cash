"""A cell below an edited draw starts where the edited draw leaves the stream.

How far a statement moves a seeded stream can follow its inputs:
``w = np.random.rand(h)`` draws ``h`` numbers, ``Model(hidden)`` initialises
``hidden`` weights. After an edit of ``h`` in the drawing cell, a run from the
top gives the cell below the stream as the statement leaves it under the new
``h``, not where it left it on the earlier run, whose text was the same.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

pytestmark = [pytest.mark.libraries, pytest.mark.fresh_kernel]

ON = "import cash\n%cash_on"


def _printed(text: str) -> list[str]:
    return re.findall(r"^R .*$", text, re.M)


def _plain(cells: list[str]) -> list[str]:
    """What plain Python prints for *cells* run top to bottom."""
    out: list[str] = []
    ns: dict = {"print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    for src in cells:
        exec(compile(src, "<plain>", "exec"), ns)
    return [line for line in out if line.startswith("R ")]


HEAD = "import time\nimport numpy as np\nclass Model:\n    def __init__(self, h):\n        self.w = np.random.rand(h)"
BELOW = f"z = (time.sleep({F}), np.random.rand(2))[1]\nprint('R', repr(float(z.sum())))"

# The draw spelled in the cell, and hidden in a constructor (the E11 iris
# classifier: `model = Model(input_size, hidden_size, output_size)`).
SHAPES = {
    "spelled": "h = 16\nw = np.random.rand(h)",
    "hidden": "h = 16\nmodel = Model(h)",
}


def _cells(draw: str) -> list[str]:
    return [HEAD, "np.random.seed(1)\na = np.random.rand(3)", "np.random.seed(1)\nn = 2", draw, BELOW]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_cell_below_an_edited_draw_gets_the_run_from_the_top(nb_runner, shape):
    cells = _cells(SHAPES[shape])
    edited = SHAPES[shape].replace("16", "8")
    nb_runner.create_notebook([ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, edited)
    nb_runner.run_cell(5)
    nb_runner.run_cell(6)
    want = _plain([*cells[:3], edited, cells[4]])
    assert want != _plain(cells), "the edit must move the stream below"
    assert _printed(nb_runner.get_output(6)) == want


@pytest.mark.timeout(180)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_edited_draw_not_run_yet_is_not_taken_as_run(nb_runner, shape):
    """The edited cell has not run: where its draw leaves the stream under
    the new `h` is known nowhere. The cell below draws from where the stream
    stands and says so, never from where the old `h` left it."""
    cells = _cells(SHAPES[shape])
    edited = SHAPES[shape].replace("16", "8")
    nb_runner.create_notebook([ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, edited)
    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert _printed(out) != _plain(cells), "the stream the old `h` left"
    assert "RANDOM-STREAM-UNRESTORED" in out
