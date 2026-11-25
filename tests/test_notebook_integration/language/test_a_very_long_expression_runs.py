"""A statement Python runs is never failed by cash's analysis of it.

A sum of a few hundred terms, a long ``elif`` chain or a long chain of method
calls nests one level per term in the syntax tree. Python's compiler takes
thousands of levels; cash's walks of the tree, inside the kernel's already
deep stack, ran out at a few hundred, and the RecursionError was raised as
if the cell had raised it. Such a cell runs as written, uncached, and says
that a statement is nested too deep to analyse.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SUM_400 = "x = " + " + ".join(["a"] * 400)
ELIF_800 = "if a == 0:\n    x = 0\n" + "".join(f"elif a == {i}:\n    x = {i}\n" for i in range(1, 800))
CHAIN_300 = "x = s" + ".strip()" * 300

CASES = {
    "a_sum_of_400_terms": (["a = 1", SUM_400, "print('R', x)"], "R 400"),
    "an_elif_chain_of_800_branches": (["a = 7", ELIF_800, "print('R', x)"], "R 7"),
    "a_chain_of_300_method_calls": (["s = '  padded  '", CHAIN_300, "print('R', x)"], "R padded"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_cell_runs_and_says_it_was_not_cached(nb_runner, case):
    cells, expected = CASES[case]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()

    nb_runner.run_all()

    assert expected in nb_runner.get_output(3)
    assert "nested too deep for cash to analyse" in nb_runner.get_raw_output(2)


def test_a_shorter_expression_is_cached_as_before(nb_runner):
    nb_runner.create_notebook(["a = 1", "x = " + " + ".join(["a"] * 100), "print('R', x)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "R 100" in nb_runner.get_output(3)
    assert "nested too deep" not in nb_runner.get_raw_output(2)


def test_a_recursion_error_of_the_cells_own_code_is_still_the_cells_error(nb_runner):
    nb_runner.create_notebook(["def down(n):\n    return down(n + 1)", "down(0)"])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    with pytest.raises(Exception, match=r"down\(0\)"):
        nb_runner.run_cell(2)
    assert "nested too deep" not in nb_runner.get_raw_output(2)
