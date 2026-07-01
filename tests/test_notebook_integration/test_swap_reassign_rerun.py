"""A cell that reassigns variables from each other (swap / rotate / temp-swap)
must be idempotent on isolated re-run — the read-and-written names reset to
their cell-entry base first, instead of composing on the already-swapped state
(CAS-65).
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


def test_tuple_swap(nb_runner):
    _rerun(nb_runner, "a = 1\nb = 2", "a, b = b, a\nprint('ab', a, b)", "ab 2 1")


def test_three_way_rotate(nb_runner):
    _rerun(nb_runner, "a = 1\nb = 2\nc = 3", "a, b, c = c, a, b\nprint('r', a, b, c)", "r 3 1 2")


def test_temp_swap(nb_runner):
    _rerun(nb_runner, "a = 1\nb = 2", "tmp = a\na = b\nb = tmp\nprint('ab', a, b)", "ab 2 1")


def test_partial_upstream_swap(nb_runner):
    # a is upstream; b is created in-cell, then swapped with a.
    _rerun(nb_runner, "a = 1", "b = 9\na, b = b, a\nprint('ab', a, b)", "ab 9 1")


def test_selfref_reassign_control(nb_runner):
    # Control: single-target self-referential reassign already resets correctly.
    _rerun(nb_runner, "x = 5", "x = x + 1\nprint('x', x)", "x 6")


def test_list_swap(nb_runner):
    _rerun(nb_runner, "p = [1]\nq = [2]", "p, q = q, p\nprint('pq', p, q)", "pq [2] [1]")
