"""A restored statement that draws through a helper from a generator held in
a variable leaves that generator where a run would.

Seen in a notebook (E4 of the correctness set)::

    rng = np.random.RandomState(42)
    x = 10 * rng.rand(200)
    def model(x, sigma=0.3):
        ...
        noise = sigma * rng.randn(len(x))
        ...
    y = model(x)

After a restart ``y = model(x)`` was restored, but ``rng`` stayed where
``x = ...`` left it: the hit put back the generators the statement names,
and ``rng`` is read only inside ``model``. Every draw from ``rng`` after it
then repeated numbers a run had already drawn. The oracle is the same cells
run without cash.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

DRAWS = textwrap.dedent("""\
    import time
    import numpy as np
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(200)

    def model(x, sigma=0.3):
        noise = sigma * rng.randn(len(x))
        time.sleep(0.3)  # worth storing
        return np.sin(5 * x) + np.sin(0.5 * x) + noise

    y = model(x)
""")
CURVE = "ytrue = model(np.linspace(0, 10, 1000), sigma=0)"
NEXT = "z = rng.randn(3)"

#: Where the generator stands, and what the cells made of it.
POSITION = "(int(rng.get_state()[2]), rng.get_state()[1].tolist(), float(y.sum()))"
STATUSES = (
    "[str(r['status']) for r in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)


def _plain(cells):
    ns = {}
    for cell in cells:
        exec(cell, ns)
    return repr(eval(POSITION, ns))


def test_the_generator_stands_where_a_run_leaves_it_after_a_restart(nb_runner):
    cells = [DRAWS, CURVE, NEXT]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()

    for upto in range(1, len(cells) + 1):
        nb_runner.run_cell(upto)
        if upto < len(cells):
            assert "RESTORED" in nb_runner.peek(STATUSES), upto
        assert nb_runner.peek(POSITION) == _plain(cells[:upto]), upto
