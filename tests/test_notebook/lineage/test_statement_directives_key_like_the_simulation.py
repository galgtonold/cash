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
