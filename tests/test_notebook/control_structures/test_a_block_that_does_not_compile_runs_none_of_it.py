"""A block that does not compile as a whole runs none of its statements.

IPython compiles a cell's top-level statements one by one, so an ``if``
holding a ``return`` (or a ``break`` outside a loop) raises SyntaxError
before its test or body runs; the statements above it have run. cash runs a
block's test and body statements one by one, each compiling on its own, so
the body ran up to the ``return`` and bound what it bound.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ("if y > 0:\n    z = 1\n    print('body ran')\n    return 1", "'return' outside function"),
        ("for i in range(2):\n    z = i\n    print('body ran')\nelse:\n    yield z", "'yield' outside function"),
        ("with open(__file__) as f:\n    z = 1\n    print('body ran')\n    break", "'break' outside loop"),
    ],
)
def test_the_block_fails_before_it_starts(cash_magics, mock_shell, capsys, block, message):
    with pytest.raises(SyntaxError, match=message) as raised:
        run_cash_cell(cash_magics, "y = 5\n" + block)

    assert raised.value.lineno == block.count("\n") + 2  # the line in the cell, as IPython reports it
    assert mock_shell.user_ns["y"] == 5  # the statement above it ran, as in plain
    assert "z" not in mock_shell.user_ns
    assert "body ran" not in capsys.readouterr().out


def test_a_block_that_compiles_still_runs(cash_magics, mock_shell, capsys):
    run_cash_cell(cash_magics, "y = 5\nif y > 0:\n    z = 1\n    print('body ran')")

    assert mock_shell.user_ns["z"] == 1
    assert "body ran" in capsys.readouterr().out
