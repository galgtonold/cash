"""What the condition of an ``if`` changes reaches the cells that read it.

``if opts.pop('debug', False):`` and ``if stack.pop() > 7:`` in a loop
change the dict and the list whichever branch is taken. Only the branches'
changes moved a lineage, so a statement reading the value after the ``if``
was served the result from before the change. The expected values are what
the cells print without cash.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

D = "import time\ndef describe(d):\n    time.sleep(0.3)\n    return sorted(d)\n"
S = "import time\ndef stat(v):\n    time.sleep(0.3)\n    return sum(v)\n"


def test_the_same_statement_below_a_popping_if_sees_the_key_gone(nb_runner):
    nb_runner.create_notebook(
        [
            D + "opts = {'lr': 0.1, 'epochs': 3, 'debug': True}",
            "keys = describe(opts)\nprint(keys)",
            "if opts.pop('debug', False):\n    print('debug mode')",
            "keys = describe(opts)\nprint('after', keys)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "['debug', 'epochs', 'lr']"
    assert nb_runner.get_output(4).strip() == "after ['epochs', 'lr']"


def test_a_list_rebuilt_after_a_loop_that_pops_in_its_condition(nb_runner):
    nb_runner.create_notebook(
        [
            S + "stack = list(range(10))",
            "for i in range(5):\n    if stack.pop() > 7:\n        print('big')",
            "s1 = stat(stack)\nprint(s1)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "10"
    nb_runner.run_cell(1)
    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).strip() == "45"
