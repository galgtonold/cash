"""A branch first taken in the part of an iterator loop run as one unit moves the list it fills.

``for n in lines: if n > 300: errors.append(n)`` over a generator: the
first passes run one by one, the rest as one unit. The branch first fires in
that unit, which reports no branch it ran, and the loop end kept ``errors``'s
lineage from before the loop: the summary below was restored from before an
edit of the data, and on the first Run All a statement read the same way
above the loop handed its empty-list result to the one below. The expected
values are what the cells print without cash.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

S = "import time\ndef summarize(v):\n    time.sleep(0.3)\n    return (len(v), sum(v))\n"
DATA = S + "lines = (n for n in range({n}))\nerrors = []"
LOOP = "for n in lines:\n    if n > 300:\n        errors.append(n)"


def test_the_summary_follows_an_edit_of_the_data(nb_runner):
    nb_runner.create_notebook([DATA.format(n=400), LOOP, "summary = summarize(errors)\nprint(summary)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "(99, 34650)"

    nb_runner.set_cell_source(1, DATA.format(n=500))
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "(199, 79600)"


def test_the_same_statement_above_the_loop_is_not_reused_below_it(nb_runner):
    nb_runner.create_notebook(
        [
            DATA.format(n=400),
            "summary = summarize(errors)\nprint(summary)",
            LOOP,
            "summary = summarize(errors)\nprint('after', summary)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "(0, 0)"
    assert nb_runner.get_output(4).strip() == "after (99, 34650)"
