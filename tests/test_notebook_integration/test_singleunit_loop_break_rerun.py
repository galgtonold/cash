"""Isolated re-run idempotency for control-flow paths that execute as a single
unit (for-loops with break/continue) and instance-method self-mutation.

These take the `_execute_as_single_unit` path (like while/with), so they are
regression coverage for the CAS-59 family — confirmed correct, no fix needed.
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


def test_for_with_break_accumulator(nb_runner):
    _rerun(nb_runner, "total = 0",
           "for i in range(100):\n    if i >= 5:\n        break\n    total += i\nprint(total)", "10")


def test_for_with_continue_accumulator(nb_runner):
    _rerun(nb_runner, "total = 0\nacc = []",
           "for i in range(6):\n    if i % 2:\n        continue\n    total += i\n    acc.append(i)\nprint(total, acc)",
           "6 [0, 2, 4]")


def test_for_with_break_list(nb_runner):
    _rerun(nb_runner, "out = []",
           "for i in range(10):\n    if i > 3:\n        break\n    out.append(i * i)\nprint(out)", "[0, 1, 4, 9]")


def test_class_instance_self_mutation(nb_runner):
    _rerun(nb_runner,
           "class Acc:\n    def __init__(self):\n        self.items = []\n    def add(self, v):\n        self.items.append(v)\nacc = Acc()",
           "acc.add(1)\nacc.add(2)\nprint(acc.items)", "[1, 2]")
