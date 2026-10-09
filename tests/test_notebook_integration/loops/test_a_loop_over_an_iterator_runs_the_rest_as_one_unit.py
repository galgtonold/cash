"""A long loop over an iterator runs the rest as one unit, drawing as Python does.

``parsed = filter(...)`` then ``for (name, act) in parsed:`` over 87,464 items
ran every pass through the per-statement machinery: 21.9 s against 0.19 s
plain. The loop runs its first passes one by one and, once it is long enough,
the rest as one unit that draws from the same iterator. A generator's own
work still interleaves with the body's, and a re-run over new data is never
served from before.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
COUNT = (
    "from cash.notebook.statement import StatementProcessor as SP\n"
    "if not hasattr(SP, 'real_process'):\n"
    "    SP.real_process = SP.process_statement\n"
    "    SP.statements = []\n"
    "    def counting(self, *a, SP=SP, **k):\n"
    "        SP.statements.append(1)\n"
    "        return SP.real_process(self, *a, **k)\n"
    "    SP.process_statement = counting\n"
)
DATA = (
    "N = {n}\ndata = [('u%d' % i, [('a', i)] * (i % 3)) for i in range(N)]\nlog = []\n"
    "def gen(rows):\n    for r in rows:\n        log.append('g')\n        yield r"
)
STATEMENTS = "len(__import__('cash.notebook.statement').notebook.statement.StatementProcessor.statements)"


def test_a_filter_and_a_generator_loop(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            DATA.format(n=5000),
            "parsed = filter(lambda x: True, data)\nempty = 0\nfor (name, act) in parsed:\n"
            "    if (len(act) == 0):\n        empty += 1",
            "g = gen(data)\nfor row in g:\n    log.append('b')",
            "print('R', empty, name, log[:4], len(log), log == ['g', 'b'] * N)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cells([2, 3, 4, 5])
    assert "R 1667 u4999 ['g', 'b', 'g', 'b'] 10000 True" in nb_runner.get_output(5)
    assert int(nb_runner.peek(STATEMENTS)) < 600  # 6,676 with one pass at a time

    nb_runner.set_cell_source(2, DATA.format(n=300))
    nb_runner.run_cells([2, 3, 4, 5])
    assert "R 100 u299 ['g', 'b', 'g', 'b'] 600 True" in nb_runner.get_output(5)
