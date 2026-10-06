"""A name ``%xdel`` deleted stays deleted, as ``del`` leaves it.

The upstream check read only Python ``del`` statements, so a cell below that
read a name ``%xdel`` (or a hand-written ``run_line_magic("xdel", ...)``)
deleted brought it back from the cache instead of raising NameError, and
loaded the memory the user freed again.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


@pytest.mark.parametrize("mid", ["%xdel big", "get_ipython().run_line_magic('xdel', 'big')"])
def test_reading_it_raises_name_error(nb_runner, mid):
    nb_runner.create_notebook(["big = list(range(5))", mid, "total = sum(big)"])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])

    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(3)
    assert [o.ename for o in nb_runner.nb.cells[2].outputs if o.output_type == "error"] == ["NameError"]
    assert nb_runner.peek("'big' in globals()") == "False"
