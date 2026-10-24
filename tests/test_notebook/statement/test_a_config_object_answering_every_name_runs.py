"""A config object whose ``__getattr__`` answers every name runs as Python runs it.

``def __getattr__(self, name): return self._d.get(name, 0)`` is a common
config pattern. Asked for ``__code__``, such an object hands back ``0``, and
cash's walk of what a statement may call took that for code and failed the
user's cell with ``AttributeError: 'int' object has no attribute
'co_consts'``, at ``v = cfg``. Only a code object counts as code.
"""

from __future__ import annotations

import pytest

CFG = (
    "class Cfg:\n"
    "    def __init__(self):\n"
    "        self._d = {}\n"
    "    def __getattr__(self, name):\n"
    "        return self._d.get(name, 0)"
)


@pytest.mark.parametrize(
    ("code", "name", "expected"),
    [
        ("v = cfg", "v", None),
        ("w = cfg.alpha + 1", "w", 1),
        ("x = read(cfg)", "x", 2),
        ("y = [cfg, cfg]", "y", None),
        ("z = functools.partial(read, cfg)()", "z", 2),
    ],
    ids=["alias", "attribute", "handed_to_a_function", "in_a_list", "in_a_partial"],
)
def test_the_statement_runs(statement_processor, mock_shell, code, name, expected):
    for setup in (CFG, "cfg = Cfg()", "import functools", "def read(c):\n    return c.beta + 2"):
        statement_processor.process_statement(setup)
    statement_processor.process_statement(code)
    assert name in mock_shell.user_ns
    if expected is not None:
        assert mock_shell.user_ns[name] == expected


def test_a_setting_changed_is_read_again(statement_processor, mock_shell):
    for setup in (CFG, "cfg = Cfg()", "def read(c):\n    return c.beta + 2", "x = read(cfg)"):
        statement_processor.process_statement(setup)
    statement_processor.process_statement("cfg._d['beta'] = 5")
    statement_processor.process_statement("x = read(cfg)")
    assert mock_shell.user_ns["x"] == 7
