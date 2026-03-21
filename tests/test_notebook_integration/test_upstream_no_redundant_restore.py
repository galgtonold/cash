"""
Test that upstream variables are NOT redundantly restored when nothing has changed.

Bug: When re-executing a cell that uses `df`, the upstream checker restored ALL
upstream `df` statements every time — even though `df` hadn't changed and was
already current in memory.

Root cause: The UpstreamChecker didn't have access to function_tracker, so its
simulation produced different cache keys than the runtime for any statement
using user-defined functions. This caused forward propagation to fail, the
fallback lineage computation (without func/module components) to diverge,
and `df` to be incorrectly classified as "broken".

Fix: Share function_tracker from StatementProcessor to UpstreamChecker.
"""

import pytest

pytestmark = pytest.mark.upstream


def test_no_redundant_upstream_restore_with_functions(nb_runner):
    """
    When a cell uses df and upstream cells define functions used on df,
    re-running the cell should NOT restore upstream df statements.
    """
    nb_runner.create_notebook([
        # Cell 1: Define a transform function and apply it
        (
            "def double(x):\n"
            "    return x * 2\n"
            "data = [1, 2, 3, 4, 5]\n"
            "result = [double(x) for x in data]"
        ),
        # Cell 2: Use result in a loop
        (
            "total = 0\n"
            "for val in result:\n"
            "    total += val\n"
            "print(f'Total: {total}')"
        ),
    ])
    nb_runner.start_kernel()

    # First run: everything computed fresh
    nb_runner.run_all()
    output1 = nb_runner.get_output(2)
    assert "Total: 30" in output1

    # Second run of cell 2 only: result hasn't changed, should NOT restore
    nb_runner.run_cell(2)
    output2 = nb_runner.get_output(2)
    assert "Total: 30" in output2

    # Check that no upstream restoration badge appears
    raw_output = nb_runner.get_raw_output(2)
    assert "Restored" not in raw_output, (
        f"Unexpected upstream restoration on re-run. Output: {raw_output}"
    )


def test_no_redundant_restore_multi_transform(nb_runner):
    """
    Multiple chained transformations on the same variable should not
    trigger redundant upstream restorations when re-executing a downstream cell.
    """
    nb_runner.create_notebook([
        # Cell 1: Create data
        "data = list(range(10))",
        # Cell 2: Transform step 1
        "data = [x + 1 for x in data]",
        # Cell 3: Transform step 2 with user-defined function
        (
            "def square(x):\n"
            "    return x ** 2\n"
            "data = [square(x) for x in data]"
        ),
        # Cell 4: Use data
        "total = sum(data)\nprint(f'Sum: {total}')",
    ])
    nb_runner.start_kernel()

    # First run: everything computed
    nb_runner.run_all()
    output1 = nb_runner.get_output(4)
    assert "Sum:" in output1

    # Re-run cell 4: should NOT restore upstream data transformations
    nb_runner.run_cell(4)
    output2 = nb_runner.get_output(4)
    assert "Sum:" in output2

    raw_output = nb_runner.get_raw_output(4)
    assert "Restored" not in raw_output, (
        f"Unexpected upstream restoration. Output: {raw_output}"
    )


def test_no_redundant_restore_third_reexecution(nb_runner):
    """
    The simulation cache must be synced after the first execution so that
    subsequent re-runs don't see stale virtual lineages.

    Scenario: Cell 1 defines data, Cell 2 transforms it with a function,
    Cell 3 uses it in a loop. On the 3rd execution of Cell 3 (i.e., first
    run, first re-run, second re-run), there should be zero upstream
    restoration.

    This specifically tests the fix for poisoned simulation cache:
    the first run may compute fallback lineages (different from runtime),
    but after _sync_simulation_cache_lineages(), subsequent runs should
    match.
    """
    nb_runner.create_notebook([
        # Cell 1: Create data
        "data = list(range(20))",
        # Cell 2: Transform with user-defined function
        (
            "def transform(x):\n"
            "    return x ** 2 + 1\n"
            "data = [transform(x) for x in data]"
        ),
        # Cell 3: Use data in a loop
        (
            "total = 0\n"
            "for val in data:\n"
            "    total += val\n"
            "print(f'Total: {total}')"
        ),
    ])
    nb_runner.start_kernel()

    # First run
    nb_runner.run_all()
    output1 = nb_runner.get_output(3)
    assert "Total:" in output1

    # Second run of cell 3
    nb_runner.run_cell(3)
    output2 = nb_runner.get_output(3)
    assert "Total:" in output2

    # Third run of cell 3 — this is the critical test case.
    # Without the simulation cache sync fix, the 3rd run would still see
    # stale virtual lineages and trigger restoration.
    nb_runner.run_cell(3)
    output3 = nb_runner.get_output(3)
    assert "Total:" in output3

    raw_output = nb_runner.get_raw_output(3)
    assert "Restored" not in raw_output, (
        f"Unexpected upstream restoration on 3rd run. Output: {raw_output}"
    )


def test_no_redundant_restore_control_structure_upstream(nb_runner, tmp_path):
    """
    When an upstream cell contains a control structure (if/else), the simulation
    treats it as a single unit. This can cause cache key mismatch with the runtime
    (which processes body statements individually). The simulation cache sync
    should prevent this from causing repeated restoration.
    """
    # Create a data file
    data_file = tmp_path / "test_data.csv"
    data_file.write_text("a,b\n1,2\n3,4\n5,6\n")
    data_path_str = str(data_file).replace('\\', '/')

    nb_runner.create_notebook([
        # Cell 1: Conditional data loading (control structure)
        (
            "import os\n"
            f"path = '{data_path_str}'\n"
            "if os.path.exists(path):\n"
            "    with open(path) as f:\n"
            "        lines = f.readlines()\n"
            "    data = [line.strip() for line in lines[1:]]\n"
            "else:\n"
            "    data = []\n"
            "print(f'Loaded {len(data)} rows')"
        ),
        # Cell 2: Process data
        "result = len(data)\nprint(f'Result: {result}')",
    ])
    nb_runner.start_kernel()

    # First run
    nb_runner.run_all()
    output1 = nb_runner.get_output(2)
    assert "Result: 3" in output1

    # Re-run cell 2
    nb_runner.run_cell(2)
    output2 = nb_runner.get_output(2)
    assert "Result: 3" in output2

    # Third re-run
    nb_runner.run_cell(2)
    raw_output = nb_runner.get_raw_output(2)
    assert "Restored" not in raw_output, (
        f"Unexpected upstream restoration. Output: {raw_output}"
    )
