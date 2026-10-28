"""A statement reading a module made as the notebook ran keeps the value it
read when a cell below sets the module again.

``m = types.ModuleType('m')``, then ``m.v = next(it)``, ``u = m.v + 1``,
``m.v = next(it)`` and ``print(u, u2)``: having no file, ``m`` was not the
user's module, so ``u``'s key held nothing of ``m.v``. After a restart the
check of the last cell found ``u`` built otherwise than its entry said and
ran ``u = m.v + 1`` again, after the cell below had moved ``m.v`` on: 9
where plain Python gives 8.
"""

import types

import pytest

from cash.notebook.callee_reach import reached_user_code
from tools.plain_checker import check, notebook

MAKE = "import itertools, types\nm = types.ModuleType('m')\nit = itertools.count({})"


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(600)
def test_a_restart_and_run_all_give_what_plain_python_gives(tmp_path):
    shape = notebook(
        "made-module-read-above-its-next-write",
        [MAKE.format(""), "m.v = next(it)", "u = m.v + 1", "m.v = next(it)", "u2 = m.v + 1\nprint(u, u2)"],
        names=["u", "u2"],
        edits={1: {0: MAKE.format(7)}},
    )
    check(shape, tmp_path, ["floors0"]).assert_ok()


def test_what_a_made_module_holds_is_read_as_data():
    m = types.ModuleType("m_made_here")
    m.v = 3
    # By the notebook's name for it: its own tells nothing apart.
    assert reached_user_code("u = m.v + 1", {"__name__": "__main__", "m": m}).data == (("@m.v", 3),)
    # A library module's names are not the notebook's data.
    assert reached_user_code("u = t.pi + 1", {"t": __import__("math")}).data == ()


#: Made modules a name tells nothing apart: made twice by one function, two
#: of one name, and one named like a module that is loaded.
TOLD_APART = {
    "remade_by_a_function": [
        "import itertools, types\nit = itertools.count({})\ndef fresh():\n    mm = types.ModuleType('m')\n"
        "    mm.v = next(it)\n    return mm",
        "m = fresh()",
        "u = m.v + 1",
        "m = fresh()",
        "u2 = m.v + 1\nprint(u, u2)",
    ],
    "two_of_one_name": [
        "import itertools, types\nm = types.ModuleType('m')\nn = types.ModuleType('m')\nit = itertools.count({})",
        "m.v = next(it)\nn.v = 100 + next(it)",
        "u = m.v + 1",
        "w = n.v + 1",
        "m.v = next(it)\nn.v = 100 + next(it)",
        "u2 = m.v + w\nprint(u, w, u2)",
    ],
    "named_like_a_loaded_module": [
        MAKE.replace("'m'", "'textwrap'"),
        "m.v = next(it)",
        "u = m.v + 1",
        "m.v = next(it)",
        "u2 = m.v + 1\nprint(u, u2)",
    ],
}


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(600)
@pytest.mark.parametrize("shape", list(TOLD_APART))
def test_a_made_module_is_read_by_the_name_the_notebook_gives_it(tmp_path, shape):
    """Looked up by its own name, the label of what ``u`` read led to the
    last module of that name, or to the loaded one: ``u`` ran again against
    another module's ``v``, 2 where plain Python gives 1, from the first
    Run All."""
    first, *rest = TOLD_APART[shape]
    names = ["u", "w", "u2"] if shape == "two_of_one_name" else ["u", "u2"]
    shape = notebook(f"made-module-{shape}", [first.format(""), *rest], names=names, edits={1: {0: first.format(7)}})
    check(shape, tmp_path, ["floors0"]).assert_ok()
