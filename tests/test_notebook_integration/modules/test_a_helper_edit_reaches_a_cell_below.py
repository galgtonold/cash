"""Editing a helper module reaches a cell that never names it.

A wrong answer, 2/2 in a real board pack and 5/5 in a minimal repro: edit a
function in the project's own helper module, then run a cell BELOW the one
that calls it -- the ordinary way to work. The badge said MODULE RELOADED, and the cell printed,
and exported, the value built by the pre-edit helper.

The alias fix (9785293) and its tests all ran the cell that CALLS the
helper, which does recompute. Here the reader only sees ``tbl``; whether
``tbl`` is stale is the upstream check's call, and it has to know that the
statement that built it read a module that changed.
"""

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLOW = "    _ = sum(i * i for i in range(2_000_000))\n"


def _module(op):
    return "def summary(rows):\n" + SLOW + "    return " + op + "(rows)\n"


def _cells(import_line, call):
    return [
        "import cash\n%cash_on\n%cash_badge print",
        import_line + "\nROWS = [1, 2, 3, 4]",
        "tbl = " + call + "(ROWS)",
        "print('R', tbl)",
    ]


def _play(nb_runner, tmp_path, name, import_line, call):
    mod = tmp_path / (name + ".py")
    mod.write_text(_module("sum"), encoding="utf-8")
    nb_runner.create_notebook(_cells(import_line, call))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(_module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    return nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_a_plain_import(nb_runner, tmp_path):
    out, raw = _play(nb_runner, tmp_path, "helperplain", "import helperplain", "helperplain.summary")
    assert "R 4" in out, "the helper was edited and the cell below its caller printed the pre-edit value:\n" + raw


def test_an_aliased_import(nb_runner, tmp_path):
    out, raw = _play(nb_runner, tmp_path, "helperalias", "import helperalias as hm", "hm.summary")
    assert "R 4" in out, raw


def test_a_from_import(nb_runner, tmp_path):
    out, raw = _play(nb_runner, tmp_path, "helperfrom", "from helperfrom import summary", "summary")
    assert "R 4" in out, raw


def test_two_cells_below_through_a_value_built_from_it(nb_runner, tmp_path):
    """The real notebook: the exported commentary was built from the
    helper's output one more step down, not read from it directly."""
    mod = tmp_path / "helpertwo.py"
    mod.write_text(_module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            "import helpertwo as hm\nROWS = [1, 2, 3, 4]",
            "tbl = hm.summary(ROWS)",
            "note = 'total ' + str(tbl)",
            "print('R', note)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total 10" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    mod.write_text(_module("max"), encoding="utf-8")
    nb_runner.run_cell(5)
    assert "R total 4" in nb_runner.get_output(5), (
        "the helper was edited and a value two cells below its caller kept "
        "the pre-edit result:\n" + nb_runner.get_raw_output(5)
    )


def test_a_module_that_cannot_be_narrowed(nb_runner, tmp_path):
    """A helper whose closure reaches dynamic code (`globals()`) is keyed on
    the whole module, and then the invalidator DROPS the lineage of what was
    built from it. A dropped lineage compares with nothing, so this needs the
    re-run repair (TrackingState.rerun_bindings), not only a fresh simulation.
    """

    def module(op):
        return "def summary(rows):\n" + SLOW + "    assert 'summary' in globals()\n" + "    return " + op + "(rows)\n"

    mod = tmp_path / "helperdyn.py"
    mod.write_text(module("sum"), encoding="utf-8")
    nb_runner.create_notebook(_cells("import helperdyn as hm", "hm.summary"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    assert "R 4" in nb_runner.get_output(4), (
        "the helper was edited and the cell below its caller printed the "
        "pre-edit value:\n" + nb_runner.get_raw_output(4)
    )


def test_after_a_restart_and_a_jump(nb_runner, tmp_path):
    """A wrong answer, 2/2 in a real fleet notebook and 4/4 in a repro
    (``module`` mode): restart, jump to
    the last cell (everything restores), edit the helper, run the last cell
    again. The badge said MODULE RELOADED and ``by_road_class.csv`` was
    exported with the pre-edit numbers. The same edit WITHOUT a restart was
    right, which is why the first-session tests above never saw it.
    """
    mod = tmp_path / "helperrst.py"
    mod.write_text(_module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_persist on\n%cash_badge print",
            "import helperrst as hm\nROWS = [1, 2, 3, 4]",
            "tbl = hm.summary(ROWS)",
            "note = 'total ' + str(tbl)\nprint('R', note)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    assert "R total 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(_module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    assert "R total 4" in nb_runner.get_output(4), (
        "after a restart, the helper was edited and the last cell kept the "
        "pre-edit result:\n" + nb_runner.get_raw_output(4)
    )


def test_the_text_badge_for_a_reload_is_ascii(nb_runner, tmp_path):
    """The module-reload row carried a U+1F504 glyph, and their cp1252
    console client crashed reading the badge. `%cash_badge print` is for
    exactly that reader, and the docs promise it plain ASCII."""
    _out, raw = _play(nb_runner, tmp_path, "helperascii", "import helperascii", "helperascii.summary")
    badge = raw[raw.find("[Cash]") :]
    assert "reloaded" in badge.lower(), raw
    assert badge.isascii(), [c for c in badge if not c.isascii()]


@pytest.mark.parametrize(
    "import_line,prefix",
    [
        ("import helperloop as hm", "hm."),
        ("import helperloop", "helperloop."),
    ],
)
def test_a_loop_that_calls_the_helper(nb_runner, tmp_path, import_line, prefix):
    """The board pack builds its regional table in a loop. A loop's
    recorded outcome is reused when what it read still matches, and after the
    edit the module name still carried its pre-edit lineage in the simulation,
    so the stale table was adopted (their repro, 2/2 with the loop variants)."""

    def module(op):
        return "def summary(rows, g):\n" + SLOW + "    return " + op + "(rows) + g\n"

    mod = tmp_path / "helperloop.py"
    mod.write_text(module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            import_line + "\nROWS = [1, 2, 3, 4]",
            "blocks = {}\nfor g in [0, 100]:\n    blocks[g] = " + prefix + "summary(ROWS, g)\n"
            "tbl = sorted(blocks.values())",
            "print('R', tbl)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R [10, 110]" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    assert "R [4, 104]" in nb_runner.get_output(4), (
        "the helper was edited and the table the loop builds kept the pre-edit values:\n" + nb_runner.get_raw_output(4)
    )


def test_a_loop_through_a_helper_that_cannot_be_narrowed(nb_runner, tmp_path):
    """The repro exactly: the helper draws from a seeded generator, which
    keeps it from being narrowed to its symbols, and a loop builds the dict a
    later statement turns into the table. Both lose their lineage on the edit,
    and re-running only the table's statement rebuilt it from the stale dict."""

    def module(op):
        return (
            "import random\n"
            "def summary(rows, g):\n" + SLOW + "    random.Random(0).random()\n" + "    return " + op + "(rows) + g\n"
        )

    mod = tmp_path / "helperrng.py"
    mod.write_text(module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            "import helperrng as hm\nROWS = [1, 2, 3, 4]",
            "blocks = {}\nfor g in [0, 100]:\n    blocks[g] = hm.summary(ROWS, g)\ntbl = sorted(blocks.values())",
            "print('R', tbl)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R [10, 110]" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    assert "R [4, 104]" in nb_runner.get_output(4), (
        "the helper was edited and the table the loop builds kept the pre-edit values:\n" + nb_runner.get_raw_output(4)
    )


@pytest.mark.parametrize("restart", [False, True])
def test_a_helper_imported_in_the_cash_on_cell(nb_runner, tmp_path, restart):
    """A wrong answer, 1/1 real + 3/3 alias + 3/3 plain, and it survived
    Restart & Run All.

    Their first cell is `import cash`, `%cash_on`, then their imports -- the
    layout the quickstart now says is fine. Cash tracks a local module when a
    cell it processes imports it, and it does not process the cell that turns
    it on. So the helper was never tracked: an edit reloaded nothing, and the
    helper's source reached no key, so `z = hm.norm(y)` kept its pre-edit
    entry even after a restart, while `y` itself was recomputed.
    """
    mod = tmp_path / ("helpercashon" + ("r" if restart else "") + ".py")
    name = mod.stem
    mod.write_text(_module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_persist on\n%cash_badge print\nimport " + name + " as hm",
            "ROWS = [1, 2, 3, 4]",
            "tbl = hm.summary(ROWS)",
            "note = 'total ' + str(tbl) + ' ' + str(sum(i * i for i in range(2_000_000)) % 1)",
            "print('R', note)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total 10 0" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    mod.write_text(_module("max"), encoding="utf-8")
    if restart:
        nb_runner.restart()
        nb_runner.run_all()
    else:
        nb_runner.run_cell(5)
    assert "R total 4 0" in nb_runner.get_output(5), (
        "the helper imported in the %cash_on cell was edited and the cell "
        "below kept the pre-edit result:\n" + nb_runner.get_raw_output(5)
    )


def test_a_same_size_edit_in_the_second_of_the_import_reaches_a_restart(nb_runner, tmp_path):
    """The case above failed intermittently in a full run: on a warm kernel
    the edit lands in the same second as the first import, and ``sum`` ->
    ``max`` keeps the size, so the restarted kernel imported the helper from
    the ``.pyc`` of the first save -- Python checks only whole-second mtime
    and size. Cash keyed the helper's readers by the edited file while the
    old code ran, and persisted the old value under the new key. Here the
    edit keeps the first save's mtime outright, and the ``.pyc`` is dated to
    that same instant, so it is not left to how fast the kernel started.
    """
    mod = tmp_path / "helpersamesec.py"
    mod.write_text(_module("sum"), encoding="utf-8")
    first = os.stat(mod)
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_persist on\n%cash_badge print\nimport helpersamesec as hm",
            "ROWS = [1, 2, 3, 4]",
            "tbl = hm.summary(ROWS)",
            "note = 'total ' + str(tbl) + ' ' + str(sum(i * i for i in range(2_000_000)) % 1)",
            "print('R', note)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total 10 0" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    pyc = tmp_path / "__pycache__"
    assert any(p.name.startswith("helpersamesec.") for p in pyc.iterdir()), (
        "no .pyc: the restart cannot load stale bytecode"
    )

    mod.write_text(_module("max"), encoding="utf-8")
    os.utime(mod, ns=(first.st_atime_ns, first.st_mtime_ns))
    # The first import compiled the .pyc in the same instant as the first save.
    # A slow kernel start (coverage, a loaded machine) would otherwise date it
    # seconds later, and a .pyc written well after its source is trusted.
    for p in pyc.iterdir():
        if p.name.startswith("helpersamesec."):
            os.utime(p, ns=(first.st_atime_ns, first.st_mtime_ns))
    nb_runner.restart()
    nb_runner.run_all()
    assert "R total 4 0" in nb_runner.get_output(5), (
        "a same-size edit in the second of the import was not what ran after "
        "the restart:\n" + nb_runner.get_raw_output(5)
    )
