"""A ``%time`` line in a loop or ``if`` body changes what its Python changes.

``acc = []`` then ``for i in range(2): %time acc.append(i * k)``: after an
edit of ``k`` above, even a full Run All served the cell below from the cache
with the list the old ``k`` left. The plain ``acc.append(i * k)`` was right.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

HEAD = "import time\ndef slow(v):\n    time.sleep(0.3); return v\nk = 2"
READER = "print('r', slow(list(acc)))"


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        (["acc = []\nfor i in range(2):\n    %time acc.append(i * k)"], "r [0, 3]"),
        (["acc = []\nif k:\n    %time acc.append(k)"], "r [3]"),
        (["acc = []", "for i in range(2):\n    %time acc.append(i * k)"], "r [0, 3]"),
    ],
    ids=["loop", "branch", "a cell of the loop alone"],
)
def test_an_edit_above_reaches_the_reader_on_run_all(nb_runner, cells, expected):
    nb_runner.create_notebook([HEAD, *cells, READER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, HEAD.replace("k = 2", "k = 3"))
    nb_runner.run_all()
    reader = len(cells) + 2
    assert expected in nb_runner.get_output(reader), nb_runner.get_raw_output(reader)


def test_an_edit_above_reaches_the_reader_run_alone(nb_runner):
    nb_runner.create_notebook([HEAD, "acc = []\nfor i in range(2):\n    %time acc.append(i * k)", READER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, HEAD.replace("k = 2", "k = 3"))
    nb_runner.run_cell(3)
    assert "r [0, 3]" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
