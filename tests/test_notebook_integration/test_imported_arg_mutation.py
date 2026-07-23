"""A function IMPORTED from a .py file that mutates its argument must be tracked.

The realistic workflow: a notebook does ``from mymod import build`` and calls
``build(obj)`` where ``build`` mutates ``obj`` in place; a later cell reads the
mutated object. The mutation must be attributed to ``obj`` so a downstream cell
restores the MUTATED object, not its pre-mutation constructor.

Regression: the function-arg-mutation detector resolved callee source only from
notebook CELLS, so an IMPORTED ``build`` (defined in a file, not a cell) -- and
especially one whose body calls a module-level HELPER (interprocedural) -- was
invisible. The runtime captured the mutation but the simulation did not, so the
cross-cell restore reverted it: the reader cell errored (missing column) or saw
a stale value, even on the first (cold) run.

Run through ``scripts/fails_first.py`` to confirm these fail without the fix.
"""
import pytest

pytestmark = pytest.mark.core

N = 200000
EXPECTED = 3 * sum(range(N))


def _num(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("RESULT"):
            return line.split(None, 1)[1].strip()
    return f"<no RESULT: {out[:80]!r}>"


def _run(nb_runner, tmp_path, module_body):
    (tmp_path / "mutmod.py").write_text(module_body, encoding="utf-8")
    p = str(tmp_path).replace("\\", "/")
    nb_runner.create_notebook([
        f"import sys\nsys.path.insert(0, '{p}')\nimport pandas as pd\n"
        f"from mutmod import build\nclass Box:\n    def __init__(self, df): self.df = df",
        f"box = Box(pd.DataFrame({{'a': range({N})}}))\nbuild(box)",   # create + mutate via import
        "result = int(box.df['x'].sum())\nprint('RESULT', result)",   # read in a LATER cell
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return _num(nb_runner.get_output(3))


def test_imported_fn_arg_mutation_survives_cross_cell(nb_runner, tmp_path):
    v = _run(nb_runner, tmp_path, "def build(box):\n    box.df['x'] = box.df['a'] * 3\n")
    assert v.isdigit(), f"imported-fn mutation reverted (reader lost the column): {v!r}"
    assert int(v) == EXPECTED, f"wrong value from imported-fn mutation: {v}"


def test_imported_interprocedural_arg_mutation_survives_cross_cell(nb_runner, tmp_path):
    # entry point calls a module-level helper -> needs interprocedural resolution
    # through the imported function's __globals__.
    v = _run(
        nb_runner, tmp_path,
        "def _apply(box):\n    box.df['x'] = box.df['a'] * 3\n"
        "def build(box):\n    _apply(box)\n",
    )
    assert v.isdigit(), f"interprocedural imported-fn mutation reverted: {v!r}"
    assert int(v) == EXPECTED, f"wrong value from interprocedural imported-fn mutation: {v}"
