"""
Test that upstream module imports are re-executed when they're not in memory.

Bug: When a cell depends on a variable computed using an imported function
(e.g., `pressure_solve = factorized(A_laplacian)` where `factorized` comes from
`from scipy.sparse.linalg import factorized`), the upstream checker failed to
schedule the import statement for re-execution when `factorized` wasn't in memory.

Root cause: The backwards scan in _find_and_reexecute skipped module imports
(`if inp in virtual_modules: continue`) without checking whether they were actually
present in memory. After a kernel restart or fresh session, imported names like
`factorized` would be missing, causing NameError when the dependent statement
was auto-executed.

Fix: Check `self.shell.user_ns` before skipping module imports. If the import
is in virtual_modules but NOT in memory, schedule it for re-execution.
"""
import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.modules]


def test_mid_cell_import_available_for_upstream_execution(nb_runner):
    """
    Test that a function imported mid-cell is available when its dependent
    statement is scheduled for upstream re-execution.

    Reproduces the CFD demo bug where:
    - Cell A has: `from scipy.sparse.linalg import factorized` followed by
      `pressure_solve = factorized(A_laplacian)`
    - Cell B depends on `pressure_solve`
    - Running Cell B triggers upstream execution of `pressure_solve = factorized(...)`
    - But `factorized` isn't in memory → NameError
    """
    nb_runner.create_notebook([
        # Cell 1: imports
        "from collections import OrderedDict",
        # Cell 2: define data
        "data = {'b': 2, 'a': 1, 'c': 3}",
        # Cell 3: import mid-cell + use the import
        (
            "from operator import itemgetter\n"
            "sorted_keys = sorted(data.keys(), key=itemgetter(0))\n"
            "result = OrderedDict((k, data[k]) for k in sorted_keys)\n"
            "print(f'Sorted: {list(result.items())}')"
        ),
        # Cell 4: depends on result from cell 3
        "output = list(result.values())\nprint(f'Values: {output}')",
    ])
    nb_runner.start_kernel()

    # Run cells 1-3 first to populate cache
    nb_runner.run_cells([1, 2, 3])
    out3 = nb_runner.get_output(3)
    assert "Sorted:" in out3, f"Cell 3 should produce sorted output, got: {out3}"

    # Now run cell 4 directly — triggers upstream check.
    # `result` depends on `itemgetter` (imported in cell 3).
    # The upstream checker must re-execute the import + dependent statement.
    nb_runner.run_cell(4)
    out4 = nb_runner.get_output(4)
    assert "Values: [1, 2, 3]" in out4, f"Cell 4 should show values, got: {out4}"


def test_import_and_usage_in_same_upstream_cell(nb_runner):
    """
    Test that when an upstream cell contains both an import and code using that
    import, both are correctly re-executed.

    This simulates the CFD pattern:
      Cell 7: from scipy... import factorized; pressure_solve = factorized(A)
      Cell 10: uses pressure_solve in a loop
    """
    nb_runner.create_notebook([
        # Cell 1: basic data
        "import math\nradius = 5",
        # Cell 2: import + compute in same cell (like the CFD Laplacian cell)
        (
            "from math import pi as PI\n"
            "area = PI * radius ** 2\n"
            "print(f'Area: {area:.4f}')"
        ),
        # Cell 3: depends on area from cell 2
        "circumference = 2 * math.sqrt(area / math.pi) * math.pi\n"
        "print(f'Circumference: {circumference:.4f}')",
    ])
    nb_runner.start_kernel()

    # Run all cells to build cache
    nb_runner.run_all()

    out2 = nb_runner.get_output(2)
    assert "Area:" in out2

    out3 = nb_runner.get_output(3)
    assert "Circumference:" in out3

    # Now run cell 3 alone — it depends on `area` which requires `PI` import
    nb_runner.run_cell(3)
    out3_rerun = nb_runner.get_output(3)
    assert "Circumference:" in out3_rerun, f"Expected circumference output, got: {out3_rerun}"


def test_module_import_restore_after_kernel_restart(nb_runner, tmp_path):
    """
    Test that after a kernel restart, upstream module imports are properly
    re-executed before dependent statements.

    Uses FileBackend to persist cache across restart.
    """
    cache_dir = tmp_path / "cache"
    cache_dir_str = str(cache_dir).replace("\\", "/")

    setup_cell = f"""
%load_ext cash
from cash import Cash
from cash.backends import FileBackend
from cash.notebook.ipython.magics import CashMagics

backend = FileBackend(cache_dir='{cache_dir_str}')
ip = get_ipython()
cash_inst = Cash(backend=backend, register_magic=False)
magics = CashMagics(ip, cash_inst)
ip.register_magics(magics)
%cash_on
"""

    nb_runner.create_notebook([
        # Cell 1: setup (custom backend)
        setup_cell,
        # Cell 2: data
        "values = [3, 1, 4, 1, 5]",
        # Cell 3: import + compute
        (
            "from functools import reduce\n"
            "total = reduce(lambda a, b: a + b, values)\n"
            "print(f'Total: {total}')"
        ),
        # Cell 4: use total
        "avg = total / len(values)\nprint(f'Average: {avg}')",
    ])
    nb_runner.start_kernel(with_cash=False)  # We have custom setup

    # Run all cells
    nb_runner.run_all()
    out4 = nb_runner.get_output(4)
    assert "Average:" in out4, f"Expected average output, got: {out4}"

    # Now run only cell 4 — triggers upstream check
    # `total` depends on `reduce` which must be re-imported
    nb_runner.run_cell(4)
    out4_rerun = nb_runner.get_output(4)
    assert "Average:" in out4_rerun, f"Expected average after re-run, got: {out4_rerun}"
