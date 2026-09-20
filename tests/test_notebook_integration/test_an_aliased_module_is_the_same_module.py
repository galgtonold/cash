"""Editing a module invalidates work done with it, whatever the cell called it.

Round 27, r27s2, reported 3/3 from an empty cache: a fix to a function in a
project module had no effect in a live kernel -- the cell re-ran and returned
the PRE-fix answer, with the badge saying

    EXECUTED: parsed = tl.parse_headers(corpus) (0.00s, saved 0.68s by cached calls)

The tester exported `per_category.csv` and `confusion_matrix.png` from that
value. Only a kernel restart fixed it. Their notebook says

    import tickets_lib as tl

and that alias is the whole finding.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLOW = "    _ = sum(i * i for i in range(3_000_000))\n"


def _module(marker):
    return ("def parse_headers(rows):\n"
            + SLOW +
            "    return [r + '" + marker + "' for r in rows]\n")


def _cells(import_line, callee):
    return [
        "import cash\n%cash_on\n%cash_badge print",
        import_line + "\nROWS = ['a', 'b', 'c']",
        "parsed = " + callee + "(ROWS)\nprint('R', parsed[0])",
    ]


def _play(nb_runner, tmp_path, name, import_line, callee):
    mod = tmp_path / (name + ".py")
    mod.write_text(_module("_OLD"), encoding="utf-8")

    nb_runner.create_notebook(_cells(import_line, callee))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R a_OLD" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    mod.write_text(_module("_NEW"), encoding="utf-8")
    nb_runner.run_cell(3)
    return nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_an_aliased_module_edit_reaches_the_call(nb_runner, tmp_path):
    """r27s2's shape: `import lib as x`, then `x.f(...)`."""
    out, raw = _play(nb_runner, tmp_path, "aliaslib",
                     "import aliaslib as al", "al.parse_headers")
    assert "R a_NEW" in out, (
        "the module was edited and the call returned the pre-edit value; "
        "only the alias differs from the passing case below:\n" + raw
    )


def test_the_same_module_without_an_alias(nb_runner, tmp_path):
    """The control, and why three minimal repros missed this.

    All three of the tester's minimisations wrote `import mylib`, and so does
    every module-reload test in this suite.
    """
    out, raw = _play(nb_runner, tmp_path, "plainlib",
                     "import plainlib", "plainlib.parse_headers")
    assert "R a_NEW" in out, raw


def test_a_from_import_of_the_function(nb_runner, tmp_path):
    """The third spelling, for completeness."""
    out, raw = _play(nb_runner, tmp_path, "fromlib",
                     "from fromlib import parse_headers", "parse_headers")
    assert "R a_NEW" in out, raw
