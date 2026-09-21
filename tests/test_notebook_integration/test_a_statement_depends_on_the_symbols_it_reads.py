"""Editing one function in a module re-runs only what reads that function.

Round 27, r27s2 (ANNOYING, measured 2.8x then 9.1x): editing one helper in
their project module re-read all 10,000 ticket files after a restart, because
the load statement -- `corpus = tl.load_corpus(ROOT, MONTH)` -- was keyed on
the WHOLE module, and the module had changed. The load function had not.

Per-symbol keying: a statement that reads `lib.load` depends on `load`, on
everything `load` reaches inside the module (helpers, constants, module-level
code), and on nothing else in it. That makes the cases where it must still
re-run as important as the one where it must not, and this file pins both.

Every expensive statement below does a large matmul so it is unambiguously
worth caching; at a few milliseconds the cost floor would decide instead.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

HEAD = "import cash\n%cash_on\n%cash_persist on\n%cash_badge print\nimport numpy as np"

WORK = "float((np.ones((1600, 1600)) @ np.ones((1600, 1600))).sum())"


def _module(load_extra="", report="'v1:'", helper="1", const="3", table_fn="2"):
    """A module with an independent function and several that `load` reaches."""
    return (
        "import numpy as np\n"
        "THRESHOLD = " + const + "\n"
        "\n"
        "def build_table():\n"
        "    return " + table_fn + "\n"
        "\n"
        "TABLE = build_table()\n"
        "\n"
        "def _helper(n):\n"
        "    return n + " + helper + "\n"
        "\n"
        "def load(n):\n"
        "    total = " + WORK + "\n"
        "    return [_helper(i) * THRESHOLD * TABLE for i in range(n)]" + load_extra + "\n"
        "\n"
        "def report(rows):\n"
        "    return " + report + " + str(len(rows))\n"
    )


def _notebook(nb_runner, tmp_path, name, import_line, prefix):
    (tmp_path / (name + ".py")).write_text(_module(), encoding="utf-8")
    nb_runner.create_notebook([
        HEAD,
        import_line,
        "DATA = " + prefix + ".load(6)",
        "BIG = sum(DATA) + " + WORK,
        "OUT = " + prefix + ".report(DATA)\nprint('R', OUT, sum(DATA), BIG)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return tmp_path / (name + ".py")


def _edit(path, **changes):
    path.write_text(_module(**changes), encoding="utf-8")


def _status(raw, stmt):
    """'CACHED' / 'EXECUTED' for the badge row whose code starts with *stmt*."""
    for line in raw.splitlines():
        s = line.strip().lstrip("^~ ")
        for tag in ("CACHED", "EXECUTED", "NOT CACHED"):
            if s.startswith(tag + ": " + stmt):
                return tag
    return None


class TestAnUnrelatedEditIsFree:
    """The finding itself."""

    @LOAD_SENSITIVE
    @pytest.mark.parametrize("import_line,prefix", [
        ("import symlib", "symlib"),
        ("import symlib as sl", "sl"),
    ])
    def test_editing_another_function_does_not_re_run_this_one(
            self, nb_runner, tmp_path, import_line, prefix):
        path = _notebook(nb_runner, tmp_path, "symlib", import_line, prefix)

        _edit(path, report="'v2:'")          # `report` only; `load` untouched
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "DATA = ") == "CACHED", (
            "only `report` changed and the statement reading `load` re-ran:\n" + raw)

    @LOAD_SENSITIVE
    def test_the_same_through_a_from_import(self, nb_runner, tmp_path):
        """`from lib import load` is the other common spelling."""
        (tmp_path / "symfrom.py").write_text(_module(), encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "from symfrom import load, report",
            "DATA = load(6)",
            "OUT = report(DATA)\nprint('R', OUT, sum(DATA))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _edit(tmp_path / "symfrom.py", report="'v2:'")
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "DATA = ") == "CACHED", (
            "only `report` changed and the statement calling the from-imported "
            "`load` re-ran:\n" + raw)

    @LOAD_SENSITIVE
    def test_a_from_imported_constant_that_did_not_change(self, nb_runner, tmp_path):
        (tmp_path / "symconst2.py").write_text(_module(), encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "from symconst2 import THRESHOLD",
            "X = THRESHOLD + " + WORK + "\nprint('R', X)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _edit(tmp_path / "symconst2.py", report="'v2:'")
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "X = ") == "CACHED", (
            "only `report` changed and the statement reading THRESHOLD re-ran:\n" + raw)

    @LOAD_SENSITIVE
    def test_and_nothing_built_on_it_re_runs_either(self, nb_runner, tmp_path):
        """The key AND the lineage: otherwise DATA hits and BIG still misses."""
        path = _notebook(nb_runner, tmp_path, "symdown", "import symdown as sd", "sd")

        _edit(path, report="'v2:'")
        nb_runner.run_cell(4)
        raw = nb_runner.get_raw_output(4)
        assert _status(raw, "BIG = ") == "CACHED", (
            "`load` is unchanged, so DATA's lineage is too, and BIG re-ran:\n" + raw)


class TestWhatMustStillReRun:
    """Every way `load`'s behaviour can change without `load` changing."""

    @pytest.mark.parametrize("change,label", [
        ({"load_extra": " + [0]"}, "load itself"),
        ({"helper": "2"}, "a helper load calls"),
        ({"const": "4"}, "a module constant load reads"),
        ({"table_fn": "5"}, "a function run at import time to build a value load reads"),
    ])
    def test_a_change_load_depends_on_re_runs_it(self, nb_runner, tmp_path, change, label):
        path = _notebook(nb_runner, tmp_path, "symdep", "import symdep as sy", "sy")

        _edit(path, **change)
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "DATA = ") != "CACHED", (
            "%s changed and DATA was served from cache:\n%s" % (label, raw))

    def test_a_from_imported_function_still_sees_its_own_helper(self, nb_runner, tmp_path):
        (tmp_path / "symfromdep.py").write_text(_module(), encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "from symfromdep import load",
            "DATA = load(6)\nprint('R', sum(DATA))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _edit(tmp_path / "symfromdep.py", helper="2")
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "DATA = ") != "CACHED", (
            "a helper the from-imported `load` calls changed and DATA was "
            "served from cache:\n" + raw)

    def test_a_from_imported_constant_that_changed(self, nb_runner, tmp_path):
        (tmp_path / "symconst.py").write_text(_module(), encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "from symconst import THRESHOLD",
            "X = THRESHOLD + " + WORK + "\nprint('R', X)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _edit(tmp_path / "symconst.py", const="4")
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "X = ") != "CACHED", (
            "the from-imported constant changed and X was served from cache:\n" + raw)

    def test_a_value_import_time_code_computes_differently_every_run(self, nb_runner, tmp_path):
        """The hole narrowing would open, closed.

        An edit anywhere in the file reloads the module, and a reload runs
        `STAMP = time.time()` again. Keying on the whole module never served
        the old STAMP's result under the new one, because the edit re-keyed
        everything; narrowing must not start.
        """
        src = ("import time\nSTAMP = time.time()\n"
               "def load(n):\n    return [STAMP] * n\n"
               "def report(rows):\n    return 'v1'\n")
        path = tmp_path / "symstamp.py"
        path.write_text(src, encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "import symstamp as ss",
            "DATA = ss.load(3) + [" + WORK + "]\nprint('R', DATA[0])",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        path.write_text(src.replace("'v1'", "'v2'"), encoding="utf-8")
        nb_runner.run_cell(3)
        raw = nb_runner.get_raw_output(3)
        assert _status(raw, "DATA = ") != "CACHED", (
            "the reload recomputed STAMP and DATA was served from before it:\n" + raw)

    def test_passing_the_module_itself_depends_on_all_of_it(self, nb_runner, tmp_path):
        """The fallback: a use cash cannot bound keys on the whole module."""
        (tmp_path / "symbare.py").write_text(_module(), encoding="utf-8")
        nb_runner.create_notebook([
            HEAD,
            "import symbare",
            "def run(mod):\n    return mod.load(6)",
            "DATA = run(symbare)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _edit(tmp_path / "symbare.py", report="'v2:'")
        nb_runner.run_cell(4)
        raw = nb_runner.get_raw_output(4)
        assert _status(raw, "DATA = ") != "CACHED", (
            "the module was passed whole, so any edit to it must count:\n" + raw)


class TestAcrossARestart:
    """The path that would make this worse if runtime and simulation disagreed."""

    @LOAD_SENSITIVE
    def test_a_restart_then_a_jump_still_restores_it(self, nb_runner, tmp_path):
        _notebook(nb_runner, tmp_path, "symrs", "import symrs as rs", "rs")

        nb_runner.restart()
        nb_runner.run_cell(1)
        nb_runner.run_cell(4)
        raw = nb_runner.get_raw_output(4)
        assert "EXECUTED: DATA = " not in raw, (
            "after a restart the jump re-ran the load it should have "
            "restored:\n" + raw)

    @LOAD_SENSITIVE
    def test_an_unrelated_edit_across_a_restart_is_still_free(self, nb_runner, tmp_path):
        """r27s2's actual sequence: edit a helper, next session, jump down."""
        path = _notebook(nb_runner, tmp_path, "symrs2", "import symrs2 as r2", "r2")

        nb_runner.restart()
        _edit(path, report="'v2:'")
        nb_runner.run_cell(1)
        nb_runner.run_cell(4)
        raw = nb_runner.get_raw_output(4)
        assert "EXECUTED: DATA = " not in raw, (
            "only `report` changed between sessions and the load re-ran:\n" + raw)
