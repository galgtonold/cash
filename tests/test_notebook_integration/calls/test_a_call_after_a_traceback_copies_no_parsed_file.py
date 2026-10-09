"""A statement that calls a function after an error copies no parsed file.

IPython's traceback display sets ``.parent`` on every node of each file it
shows, the shared ``ast.Load()`` node included. Cash copied each call's
syntax tree with ``copy.deepcopy``, which followed that link: after one
error inside pandas, every call in a loop body cost 0.1-0.6 s for the rest of
the session. The copy now follows only the syntax fields.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
COUNT = (
    "import ast, copy\n"
    "if not hasattr(copy, 'tree_copies'):\n"
    "    copy.tree_copies = []\n"
    "    copy.real_deepcopy = copy.deepcopy\n"
    "    def counting(x, memo=None, copy=copy, ast=ast):\n"
    "        if isinstance(x, ast.AST):\n"
    "            copy.tree_copies.append(type(x).__name__)\n"
    "        return copy.real_deepcopy(x, memo)\n"
    "    copy.deepcopy = counting\n"
)


def test_a_loop_of_calls_after_an_error_copies_no_tree(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "import json, re\ndef tidy(s):\n    return s.strip().lower()\nline = ' A1 -> B2 -> C3 '",
            "json.loads('{')  # an error raised inside a module: its traceback reads json/decoder.py",
            "out = []\nfor part in line.split('->'):\n    out.append(re.findall(r'[A-Z]\\d', tidy(part).upper())[0])\nr = tidy(line)",
            "print('R', out, repr(r))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(3)
    # what a traceback leaves behind: the shared Load node now links into a parsed file
    assert nb_runner.peek("hasattr(__import__('ast').parse('x').body[0].value.ctx, 'parent')") == "True"
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cells([4, 5])
    assert "R ['A1', 'B2', 'C3'] 'a1 -> b2 -> c3'" in nb_runner.get_output(5)
    assert nb_runner.peek("len(__import__('copy').tree_copies)") == "0"
