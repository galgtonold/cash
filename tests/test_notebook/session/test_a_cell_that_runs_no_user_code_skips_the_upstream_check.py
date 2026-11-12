"""Which cells cash proves run none of the user's Python, so the upstream
check and its run-as-written fingerprints are skipped.

Only a cell the upstream check cannot read is a candidate. It is proved when
IPython's own transform of it does not parse (IPython shows the SyntaxError
and runs nothing), or when it is one automagic line of a magic that runs no
Python, with no ``{expr}`` or ``$name`` for IPython to expand. Anything else
keeps the old path.
"""

from __future__ import annotations

import pytest
from IPython.testing.globalipapp import get_ipython

from cash.notebook.ipython.ipython_cell import runs_no_user_code


@pytest.fixture(scope="module")
def transform():
    shell = get_ipython()
    shell.automagic = True
    return shell.transform_cell


@pytest.mark.parametrize(
    "cell",
    [
        "x = 1 2",
        "y = data[0] +",
        "for i in range(3):\nprint(i)",
        "pip install some-package",
        "env NAME=value",
        "pip install x\nimport y",  # automagic is single-line only: a SyntaxError
    ],
)
def test_proved(cell, transform):
    assert runs_no_user_code(cell, transform)


@pytest.mark.parametrize(
    "cell",
    [
        "x = 1",
        "%cd /tmp",  # the upstream check reads it
        "%pip install x\nx = 1",
        'run -i "script.py"',  # runs Python
        "time x = f() + 1 2",  # the magic's own argument; IPython compiles it
        "pip install {pick()}",  # IPython evaluates the braces
        "whos",
        "await f()",
    ],
)
def test_not_proved(cell, transform):
    assert not runs_no_user_code(cell, transform)


def test_a_transform_that_raises_a_syntax_error_is_proved():
    def refuse(_cell: str) -> str:
        raise SyntaxError("refused")

    assert runs_no_user_code("x = 1 2", refuse)


def test_a_transform_that_fails_otherwise_is_not_proved():
    def broken(_cell: str) -> str:
        raise RuntimeError("broken")

    assert not runs_no_user_code("x = 1 2", broken)
