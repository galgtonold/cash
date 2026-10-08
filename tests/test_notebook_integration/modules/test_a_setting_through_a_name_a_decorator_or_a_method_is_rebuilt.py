"""A setting on a module made without naming the module survives a restart or an edit.

``from mylib import CONFIG; CONFIG['k'] = 5``, ``cfg = mylib.CONFIG; cfg['k'] =
5``, ``@mylib.register`` on a def, ``import plugin`` where the plugin registers
itself, and ``mylib.Cfg.tune(5)`` doing ``cls.k = k``: after a kernel restart,
or an edit of ``mylib.py``, running only the reader below them computed with
the file's values, with no warning, where Restart & Run All sees the setting.
"""

import os

import pytest

from tests._nbharness.runner import CASH_TEST_PIN_THRESHOLDS

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = (
    "import time\n"
    "CONFIG = {'k': 1}\n"
    "REG = {}\n"
    "class Cfg:\n"
    "    k = 1\n"
    "    @classmethod\n    def tune(cls, k):\n        cls.k = k\n"
    "    def setk(self, k):\n        type(self).k = k\n"
    "def register(f):\n    REG[f.__name__] = f\n    return f\n"
    "def from_k(n):\n    time.sleep(0.3)\n    return CONFIG['k'] * Cfg.k * n\n"
    "def names():\n    time.sleep(0.3)\n    return sorted(REG)\n"
)
PLUGIN = "import mylib\n@mylib.register\ndef plug():\n    return 1\n"
SETUP = "import cash\n" + CASH_TEST_PIN_THRESHOLDS
READ_K = "x = mylib.from_k(2)\nprint('X', x)"
READ_NAMES = "x = mylib.names()\nprint('X', x)"

CASES = {
    "a_from_imported_dict": (["import mylib\nfrom mylib import CONFIG", "CONFIG['k'] = 5"], READ_K, "X 10"),
    "an_alias_of_a_dict": (["import mylib", "cfg = mylib.CONFIG", "cfg['k'] = 5"], READ_K, "X 10"),
    "a_bare_decorator": (["import mylib", "@mylib.register\ndef alpha():\n    return 1"], READ_NAMES, "X ['alpha']"),
    "a_class_method_named_anything": (["import mylib", "mylib.Cfg.tune(5)"], READ_K, "X 10"),
    "a_method_setting_its_class": (["import mylib", "mylib.Cfg().setk(5)"], READ_K, "X 10"),
}


def _start(nb_runner, tmp_path, cells):
    (tmp_path / "mylib.py").write_text(LIB, encoding="utf-8")
    (tmp_path / "plugin.py").write_text(PLUGIN, encoding="utf-8")
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_setting(nb_runner, tmp_path, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    _start(nb_runner, tmp_path, cells)
    reader_cell = len(cells) + 1
    assert expected in nb_runner.get_output(reader_cell), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(reader_cell)

    assert expected in nb_runner.get_output(reader_cell), nb_runner.get_raw_output(reader_cell)


@pytest.mark.fresh_kernel
def test_an_import_that_registers_itself_is_rebuilt_after_a_restart(nb_runner, tmp_path):
    cells = ["import mylib", "import plugin", READ_NAMES]
    _start(nb_runner, tmp_path, cells)
    assert "X ['plug']" in nb_runner.get_output(4), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(4)

    assert "X ['plug']" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


@pytest.mark.parametrize("case", ["a_from_imported_dict", "an_alias_of_a_dict", "a_class_method_named_anything"])
def test_a_reader_run_alone_after_an_edit_of_the_module_sees_the_setting(nb_runner, tmp_path, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    _start(nb_runner, tmp_path, cells)
    reader_cell = len(cells) + 1
    assert expected in nb_runner.get_output(reader_cell), "control: the top-to-bottom run"

    path = tmp_path / "mylib.py"
    before = path.stat().st_mtime
    path.write_text(LIB + "def unrelated():\n    return 0\n", encoding="utf-8")
    os.utime(path, (before + 2, before + 2))
    nb_runner.run_cell(reader_cell)

    assert expected in nb_runner.get_output(reader_cell), nb_runner.get_raw_output(reader_cell)
