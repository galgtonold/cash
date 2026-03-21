"""
Integration tests for real-time output streaming from control structures.

These tests verify that print() statements inside for loops produce output
that appears in the cell output as each iteration runs, rather than being
batched until the entire loop completes.
"""
import pytest

pytestmark = [pytest.mark.loops, pytest.mark.control]


def test_for_loop_print_appears_in_output(nb_runner):
    """Print statements inside a for loop should appear in the cell output."""
    nb_runner.create_notebook([
        """for i in range(5):
    print(f'Step {i}: processing')
print('Done')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    for i in range(5):
        assert f"Step {i}: processing" in output, f"Missing output for step {i}"
    assert "Done" in output


def test_for_loop_print_with_status_updates(nb_runner):
    """
    Simulate a long-running loop with status updates (like the CFD demo).
    Output should contain all status prints.
    """
    nb_runner.create_notebook([
        """import time
residuals = []
for step in range(10):
    residual = 1.0 / (step + 1)
    residuals.append(residual)
    print(f'  Step {step:4d}/10: residual={residual:.2e}')

print(f'\\nSimulation complete')
print(f'Final residual: {residuals[-1]:.2e}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    # Verify all step outputs appear
    for step in range(10):
        1.0 / (step + 1)
        assert f"Step {step:4d}/10" in output, f"Missing output for step {step}"
    assert "Simulation complete" in output
    assert "Final residual" in output


def test_for_loop_cached_print_replays(nb_runner):
    """
    On second run with identical code, cached stdout should be replayed
    from the cache (not re-executed) and still appear in output.
    """
    nb_runner.create_notebook([
        """results = []
for i in range(3):
    results.append(i * 10)
    print(f'Computed: {i * 10}')
print(f'Total: {sum(results)}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(1)
    assert "Computed: 0" in output1
    assert "Computed: 10" in output1
    assert "Computed: 20" in output1
    assert "Total: 30" in output1

    # Re-run the same cell — output should still appear (from cache)
    nb_runner.run_cell(1)
    output2 = nb_runner.get_output(1)
    assert "Computed: 0" in output2
    assert "Computed: 10" in output2
    assert "Computed: 20" in output2
    assert "Total: 30" in output2


def test_nested_loop_print_output(nb_runner):
    """Print inside nested loops should all appear in output."""
    nb_runner.create_notebook([
        """for i in range(3):
    for j in range(2):
        print(f'({i},{j})')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    for i in range(3):
        for j in range(2):
            assert f"({i},{j})" in output, f"Missing output for ({i},{j})"


def test_if_statement_print_in_output(nb_runner):
    """Print statements in if/else branches should appear in output."""
    nb_runner.create_notebook([
        """x = 42
if x > 0:
    print(f'x={x} is positive')
    print('All good')
else:
    print('x is non-positive')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    assert "x=42 is positive" in output
    assert "All good" in output


def test_for_loop_with_conditional_print(nb_runner):
    """For loop with conditional printing should show all expected outputs."""
    nb_runner.create_notebook([
        """for i in range(10):
    if i % 3 == 0:
        print(f'Step {i}: milestone')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    assert "Step 0: milestone" in output
    assert "Step 3: milestone" in output
    assert "Step 6: milestone" in output
    assert "Step 9: milestone" in output


def test_while_loop_single_unit_prints(nb_runner):
    """While loops execute as single unit — print output should still appear."""
    nb_runner.create_notebook([
        """i = 0
while i < 5:
    print(f'while iteration {i}')
    i += 1
print('while done')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    for i in range(5):
        assert f"while iteration {i}" in output, f"Missing output for while iteration {i}"
    assert "while done" in output


def test_for_loop_with_break_single_unit_prints(nb_runner):
    """For loop with break executes as single unit — prints should stream."""
    nb_runner.create_notebook([
        """for i in range(100):
    print(f'processing {i}')
    if i >= 4:
        print('stopping early')
        break
print('loop exited')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    for i in range(5):
        assert f"processing {i}" in output, f"Missing output for processing {i}"
    assert "stopping early" in output
    assert "loop exited" in output


def test_for_loop_with_continue_single_unit_prints(nb_runner):
    """For loop with continue executes as single unit — prints should still appear."""
    nb_runner.create_notebook([
        """for i in range(6):
    if i % 2 == 0:
        continue
    print(f'odd: {i}')
print('done')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(1)
    assert "odd: 1" in output
    assert "odd: 3" in output
    assert "odd: 5" in output
    assert "done" in output


def test_single_unit_cached_replay(nb_runner):
    """Single-unit loop output should replay correctly from cache on second run."""
    nb_runner.create_notebook([
        """total = 0
for i in range(5):
    total += i
    if i == 3:
        break
    print(f'added {i}, total={total}')
print(f'final total={total}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(1)
    assert "added 0, total=0" in output1
    assert "added 1, total=1" in output1
    assert "added 2, total=3" in output1
    assert "final total=6" in output1

    # Re-run — should replay from cache
    nb_runner.run_cell(1)
    output2 = nb_runner.get_output(1)
    assert "added 0, total=0" in output2
    assert "final total=6" in output2
