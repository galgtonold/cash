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
    assert reached_user_code("u = m.v + 1", {"m": m}).data == (("m_made_here.v", 3),)
    # A library module's names are not the notebook's data.
    assert reached_user_code("u = t.pi + 1", {"t": __import__("math")}).data == ()
