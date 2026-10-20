"""The order of the names on an import line does not change what reads them.

cash-bench finding: agents wrote ``import pandas as pd, json`` in one script
and ``import json, pandas as pd`` in the next, and every ``pd.read_csv``
after the second re-ran (351 times in the benchmark). Both bind the same
``pd``; its lineage now hangs on the clause that bound it. What decides the
binding still does: another module under the same alias re-runs.
"""

import pytest

from cash.notebook.cache_key import statement_source_hash
from cash.notebook.lineage_formula import binding_source_hash

pytestmark = [pytest.mark.integration, pytest.mark.modules, pytest.mark.timeout(300)]

#: An installed module's names: a module of the user's own is keyed by its
#: file, so its import line never mattered. The statement's own work, not
#: the ``dumps`` call, is what keeps it on disk: with a big ``dumps`` the call
#: cache kept the call, the statement's own cost was ~0.1 s, and under load
#: it fell below the disk floor about one run in five and was gone after
#: the restart.
READ = "\nx = len(ml.dumps([i * i for i in range(4_000_000)][-10:]))\nprint('X', x)"


def _run(nb_runner, import_line: str) -> str:
    nb_runner.set_cell_source(2, import_line + READ)
    nb_runner.run_all()
    return nb_runner.get_raw_output(2)


def test_reordered_import_names_serve_the_reader(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on\n%cash_badge print", ""])
    nb_runner.start_kernel()
    first = _run(nb_runner, "import os, json as ml")
    assert "EXECUTED: x = len(ml.dumps" in first, first
    size = first.split("X ")[1].split()[0]

    nb_runner.restart()
    again = _run(nb_runner, "import json as ml, os")
    assert f"X {size}" in again and "CACHED: x = len(ml.dumps" in again, again

    # Control: another module under the same alias is another input.
    nb_runner.restart()
    other = _run(nb_runner, "import os, pickle as ml")
    assert "EXECUTED: x = len(ml.dumps" in other and f"X {size}" not in other, other


@pytest.mark.parametrize(
    ("code", "name", "clause"),
    [
        ("import json, pandas as pd", "pd", "import pandas as pd"),
        ("import pandas as pd, json", "pd", "import pandas as pd"),
        ("import os.path, sys", "os", "import os.path"),
        ("from .pkg import a, b as c", "c", "from .pkg import b as c"),
        ("import pandas as pd\nimport numpy as pd", "pd", "import numpy as pd"),
    ],
)
def test_a_name_hangs_on_the_clause_that_bound_it(code, name, clause):
    assert binding_source_hash(statement_source_hash(code), code, name) == statement_source_hash(clause)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("x = 1", "x"),  # not an import
        ("import json\nx = json.loads('1')", "x"),  # not only imports
        ("from sympy import *", "factor"),  # bound by no clause by name
    ],
)
def test_other_statements_keep_their_whole_hash(code, name):
    own = statement_source_hash(code)
    assert binding_source_hash(own, code, name) == own
