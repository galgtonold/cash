"""A call that only reads a memo on a class of the user's runs after a restart.

``a = fit(1)`` fills ``Model.lookup``'s ``functools.cache``; ``r = fit(1)``
then only reads it, and is stored when the notebook runs again. After a
restart ``a`` was served, so the memo stayed
empty, and ``r`` was served too: the cell reading the memo printed
``LOOKUP 1`` where plain Python prints it in ``a``. Now ``a`` runs after a
restart, filling the memo as plain Python does, with a trace and without one.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

FIT = """
import time

def fit(n):
    import memo_read_model  # only here: the notebook does not import it
    m = memo_read_model.Model()
    m.lookup(n)
    time.sleep(0.6)
    return m
"""

MODEL = """
import functools

class Model:
    @staticmethod
    @functools.cache
    def lookup(n):
        print("LOOKUP", n)
        return n * 2
"""


def _printed(nb_runner):
    return ["LOOKUP" in nb_runner.get_output(i) for i in range(2, 6)]


@pytest.mark.parametrize("trace", ["on", "off"])
def test_the_memo_is_filled_where_plain_python_fills_it(nb_runner, monkeypatch, trace):
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
    cells = [
        "%cash_badge print",
        "import memo_read_fit",
        "a = memo_read_fit.fit(1)",
        "r = memo_read_fit.fit(1)",
        "y = (type(r).lookup(1), type(r).lookup.cache_info().currsize)\nprint('Y', y)",
    ]
    nb_runner.create_notebook(cells)
    (nb_runner.work_dir / "memo_read_fit.py").write_text(FIT, encoding="utf-8")
    (nb_runner.work_dir / "memo_read_model.py").write_text(MODEL, encoding="utf-8")
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _printed(nb_runner) == [False, True, False, False]
    nb_runner.run_all()
    assert _printed(nb_runner) == [False] * 4
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _printed(nb_runner) == [False, True, False, False], [nb_runner.get_output(i) for i in range(2, 6)]
    assert "(2, 1)" in nb_runner.get_output(5)
