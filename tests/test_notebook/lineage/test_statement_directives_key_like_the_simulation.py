"""An upstream repair finds a statement's directives under the key it runs as.

The simulation plans a repair with ``statement_code``, which keeps the ``;``
that hides an expression's repr; the directive map must key it the same way.
"""

import ast

from cash.analysis.code_analyzer import statement_code
from cash.notebook.upstream.replay import StatementReplay

CELL = "log_metrics(df);  # @cash:no-cache\nx = 1  # @cash:no-cache"


def test_the_directive_map_uses_the_planned_statement_keys():
    planned = [statement_code(node, CELL) for node in ast.parse(CELL).body]
    found = StatementReplay.statement_directives([CELL])
    assert planned == ["log_metrics(df);", "x = 1"]
    for code in planned:
        assert found[code].no_cache, code


WAIVED = (
    "import time\n"
    "def slow(v):\n"
    "    open('calls.log', 'a').write('x')  # @cash:assume-safe (a counter)\n"
    "    return v\n"
    "def plain(v):\n"
    "    return v\n"
)


def test_a_repair_runs_a_waived_helper_from_its_own_text():
    """A repair runs a statement from its unparsed code, which has no
    comments: a helper it defined lost the waiver on its counter, and every
    statement calling it ran every time after a restart."""
    sources = StatementReplay.statement_exec_sources([WAIVED])
    tree = ast.parse(WAIVED)
    slow, plain = (statement_code(node, WAIVED) for node in tree.body[1:])
    assert "@cash:assume-safe" in sources[slow]
    assert plain not in sources
    assert StatementReplay.statement_exec_sources(["x = 1"]) == {}
