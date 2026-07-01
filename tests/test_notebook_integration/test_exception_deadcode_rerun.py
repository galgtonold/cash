"""A statement after a raise / failing assert must NOT execute when the upstream
simulation reconstructs a variable — the cell halts at the raise like a real
kernel, so dead post-raise code does not run (CAS-64).
"""
import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _run_all_allow_error(nb_runner):
    try:
        nb_runner.run_all()
    except CellExecutionError:
        pass  # the raising cell is expected to error


def test_post_raise_variable(nb_runner):
    # cell 1 halts at the raise: z stays 1, the dead `z = 2` never runs.
    nb_runner.create_notebook(["z = 1\nraise ValueError('halt')\nz = 2", "print('Z', z)"])
    nb_runner.start_kernel()
    _run_all_allow_error(nb_runner)
    nb_runner.run_cell(2)
    assert "Z 1" in nb_runner.get_output(2), f"got: {nb_runner.get_output(2)!r}"


def test_post_raise_list_mutation(nb_runner):
    nb_runner.create_notebook([
        "data = [0]",
        "data.append(1)\nraise ValueError('stop')\ndata.append(2)",
        "print('LEN', len(data))"])
    nb_runner.start_kernel()
    _run_all_allow_error(nb_runner)
    nb_runner.run_cell(3)
    assert "LEN 2" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"


@pytest.mark.xfail(reason="CAS-64: a failing assert is runtime-dependent; the "
                          "pure simulation can't know it aborts without evaluating it",
                   strict=False)
def test_post_assert_failure_variable(nb_runner):
    # a failing assert halts the cell -> x is never defined.
    nb_runner.create_notebook([
        "flag = False",
        "assert flag, 'off'\nx = 99",
        "print('X', x if 'x' in dir() else 'MISSING')"])
    nb_runner.start_kernel()
    _run_all_allow_error(nb_runner)
    nb_runner.run_cell(3)
    assert "X MISSING" in nb_runner.get_output(3), f"got: {nb_runner.get_output(3)!r}"
