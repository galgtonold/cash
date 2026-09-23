"""A helper that is slow to import is imported again after an edit, not restored.

The intermittent ``test_a_helper_edit_reaches_a_cell_below::test_a_from_import``
failure (sweep9, 2026-09-21; once in round 28), caught with a trace while the
machine was out of memory: the simulation found ``tbl`` stale after the helper
edit, exactly as in a passing run, but the plan RESTORED
``from helperfrom import summary`` from the cache instead of re-running it. A
restored import brings ``summary`` back with the lineage it had before the
edit, so ``tbl = summary(ROWS)`` was keyed as before and served the pre-edit
value, under a MODULE RELOADED badge. Normally an import is too cheap to be
stored, so it re-runs; under load that one took 0.17 s. A helper that does real
work when imported takes that path every time.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLOW_IMPORT = "_warm = sum(i * i for i in range(3_000_000))\n"
SLOW_CALL = "    _ = sum(i * i for i in range(2_000_000))\n"


def _module(op):
    return SLOW_IMPORT + "def summary(rows):\n" + SLOW_CALL + "    return " + op + "(rows)\n"


@pytest.mark.parametrize(
    "import_line, call",
    [
        ("from helperslow import summary", "summary"),
        ("import helperslow as hm", "hm.summary"),
    ],
    ids=["from_import", "aliased"],
)
def test_a_cell_below_sees_the_edit(nb_runner, tmp_path, import_line, call):
    mod = tmp_path / "helperslow.py"
    mod.write_text(_module("sum"), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            import_line + "\nROWS = [1, 2, 3, 4]",
            "tbl = " + call + "(ROWS)",
            "print('R', tbl)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 10" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    # an import runs every time and is never stored, however long it took
    assert "-> RAM" not in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)

    mod.write_text(_module("max"), encoding="utf-8")
    nb_runner.run_cell(4)
    assert "R 4" in nb_runner.get_output(4), (
        "the helper was edited and the cell below printed the pre-edit value:\n"
        + nb_runner.get_raw_output(2)
        + nb_runner.get_raw_output(4)
    )
