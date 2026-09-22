"""A comment added to a module does not re-run the work built on it.

The module's digest is in the key of every statement that reads a name from
it, so when the digest moves, they all miss. It used to be ``sha256`` of the
file's bytes: a comment, a blank line or a reformat re-ran the lot.

Measured on this notebook, 2026-09-21: adding one comment re-executed the
1.2 s call, for ``import lib`` and ``import lib as x`` alike. Round 27 r27s2
reported it at scale -- editing one helper re-read all 10,000 of their ticket
files, 48.7 s against a 17.3 s control, later 9.1x. The unit twin is
``tests/test_notebook/test_a_module_is_its_code_not_its_formatting.py`` and
pins what still counts: ``@cash:`` directives. Docstrings do not.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Same marker and the same `only_rerun` filter as
# `test_loop_edit_rerun_matrix.py`, whose comment block is the canonical
# explanation: these assert that work came back from cache, and under heavy
# enough load cash correctly decides it should not.
LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

HEAD = "import cash\n%cash_on\n%cash_persist on\n%cash_badge print"

WORK = "sum(i * i for i in range(2_000_000))"


def _module(extra=""):
    return ("def load(n):\n"
            "    return [str(i) + '-' + str(" + WORK + " % 7) for i in range(n)]\n"
            "\n"
            "def report(rows):\n"
            "    return 'v1:' + str(len(rows))\n" + extra)


def _run(nb_runner, tmp_path, name, import_line, prefix, edit):
    mod = tmp_path / (name + ".py")
    mod.write_text(_module(), encoding="utf-8")

    nb_runner.create_notebook([
        HEAD,
        import_line,
        "DATA = " + prefix + ".load(8)",
        "OUT = " + prefix + ".report(DATA)\nprint('R', OUT)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R v1:8" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    mod.write_text(edit, encoding="utf-8")
    nb_runner.run_cell(3)
    return nb_runner.get_raw_output(3)


@LOAD_SENSITIVE
def test_a_comment_added_to_a_module_does_not_re_run_its_callers(
        nb_runner, tmp_path):
    raw = _run(nb_runner, tmp_path, "fmtlib", "import fmtlib", "fmtlib",
               _module("# a note to myself\n"))
    assert "CACHED: DATA" in raw, (
        "a comment was added to the module and the call re-ran; nothing it "
        "does changed:\n" + raw
    )


@LOAD_SENSITIVE
def test_the_same_through_an_alias(nb_runner, tmp_path):
    raw = _run(nb_runner, tmp_path, "fmtalias", "import fmtalias as fa", "fa",
               _module("# a note to myself\n"))
    assert "CACHED: DATA" in raw, (
        "a comment was added and the call re-ran, reached through an "
        "alias:\n" + raw
    )


@LOAD_SENSITIVE
def test_a_docstring_reworded_in_a_module_does_not_re_run_its_callers(
        nb_runner, tmp_path):
    documented = _module().replace(
        "def load(n):\n", 'def load(n):\n    """Load n rows."""\n')
    mod = tmp_path / "fmtdoc.py"
    mod.write_text(documented, encoding="utf-8")
    nb_runner.create_notebook([
        HEAD,
        "import fmtdoc",
        "DATA = fmtdoc.load(8)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    mod.write_text(documented.replace("Load n rows.", "Return n rows, as strings."),
                   encoding="utf-8")
    nb_runner.run_cell(3)
    raw = nb_runner.get_raw_output(3)
    assert "CACHED: DATA" in raw, (
        "only a docstring in the module changed and the call re-ran:\n" + raw
    )


@LOAD_SENSITIVE
def test_a_docstring_reworded_in_a_cell_does_not_re_run_its_callers(nb_runner):
    """The notebook twin: the function is defined in a cell, not a file."""
    define = ('def load(n):\n'
              '    """Load n rows."""\n'
              '    return [str(i) + "-" + str(' + WORK + ' % 7) for i in range(n)]')
    nb_runner.create_notebook([HEAD, define, "DATA = load(8)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, define.replace("Load n rows.", "Return n rows."))
    nb_runner.run_cells([2, 3])
    raw = nb_runner.get_raw_output(3)
    assert "CACHED: DATA" in raw, (
        "only the docstring of the function changed and its caller re-ran:\n"
        + raw
    )


def test_an_edit_to_the_function_it_calls_still_re_runs_it(nb_runner, tmp_path):
    """The control that keeps the fix honest.

    Not load-marked: this asserts work DID happen, which load cannot fake.
    """
    changed = _module().replace("range(n)]", "range(n + 1)]")
    raw = _run(nb_runner, tmp_path, "fmtreal", "import fmtreal", "fmtreal",
               changed)
    assert "CACHED: DATA" not in raw, (
        "the function the statement calls was edited and it was served from "
        "cache anyway:\n" + raw
    )


def test_a_cash_directive_added_to_a_module_still_re_runs_it(nb_runner, tmp_path):
    """The other control, and the reason directives survive the AST.

    ``# @cash:assume-safe`` waives a purity check on the line it sits on, so
    it changes what cash does. A comment that instructs cash is not
    commentary.
    """
    annotated = _module().replace("def load(n):", "def load(n):  # @cash:assume-safe")
    raw = _run(nb_runner, tmp_path, "fmtdirective", "import fmtdirective",
               "fmtdirective", annotated)
    assert "CACHED: DATA" not in raw, (
        "a @cash: directive was added to the function and the statement was "
        "served from cache, so the directive had no effect:\n" + raw
    )
