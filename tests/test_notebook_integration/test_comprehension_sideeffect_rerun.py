"""A mutating call hidden inside a comprehension / generator expression / f-string
must be attributed to its receiver, so the receiver resets on isolated re-run
instead of accumulating (CAS-67).

Each cell appends a FIXED number of items (range(2), independent of the list) so
the correct value is identical on the first run and every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_list_comprehension_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "result = [base.append(0) for _ in range(2)]\nprint('len', len(base))", "len 5")


def test_generator_expression_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "result = list(base.append(0) for _ in range(2))\nprint('len', len(base))", "len 5")


def test_dict_comprehension_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "result = {i: base.append(0) for i in range(2)}\nprint('len', len(base))", "len 5")


def test_set_comprehension_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "result = {base.append(0) or 1 for _ in range(2)}\nprint('len', len(base))", "len 5")


def test_fstring_embedded_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "s = f'{base.append(0)}'\nprint('len', len(base))", "len 4")


def test_nested_comprehension_sideeffect(nb_runner):
    _rerun(nb_runner, "base = [1, 2, 3]",
           "result = [base.append(i) for i in range(2) for _ in range(1)]\nprint('len', len(base))", "len 5")
