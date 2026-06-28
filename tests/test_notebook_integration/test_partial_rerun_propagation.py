"""Editing an upstream cell's DATA then re-running only a downstream cell (not
the edited one) must reflect the new upstream value — "a cell == running from
the start". These all pass (data-edit propagation is robust); the un-definition
counterpart is tracked separately under CAS-62.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _edit_then_run(nb_runner, cells, edit_idx, new_src, run_idx, expect):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(edit_idx, new_src)
    nb_runner.run_cell(run_idx)  # run ONLY downstream, not the edited cell
    out = nb_runner.get_output(run_idx)
    assert expect in out, f"after edit: {out!r}"


def test_edit_list_data_propagates(nb_runner):
    _edit_then_run(nb_runner,
        ["items = [1, 2, 3]", "x = sum(items)", "print(x * 10)"],
        edit_idx=1, new_src="items = [1, 2, 3, 4]", run_idx=3, expect="100")


def test_edit_df_data_propagates(nb_runner):
    _edit_then_run(nb_runner,
        ["import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
         "total = df['a'].sum()",
         "print(total)"],
        edit_idx=1, new_src="import pandas as pd\ndf = pd.DataFrame({'a': [10, 20, 30]})",
        run_idx=3, expect="60")


def test_edit_middle_dict_propagates(nb_runner):
    _edit_then_run(nb_runner,
        ["cfg = {'mult': 2}", "factor = cfg['mult'] + 1", "print(factor * 100)"],
        edit_idx=1, new_src="cfg = {'mult': 5}", run_idx=3, expect="600")


def test_edit_upstream_two_hops(nb_runner):
    _edit_then_run(nb_runner,
        ["a = 1", "b = a * 10", "c = b + 5\nprint(c)"],
        edit_idx=1, new_src="a = 2", run_idx=3, expect="25")


def test_edit_string_concat_propagates(nb_runner):
    _edit_then_run(nb_runner,
        ["name = 'world'", "greeting = 'hello ' + name", "print(greeting.upper())"],
        edit_idx=1, new_src="name = 'there'", run_idx=3, expect="HELLO THERE")
