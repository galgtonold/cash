"""
Test for loop append mutation bug.

Issue: a.append(...) inside a loop was being cached despite mutating 'a',
causing 'a' to be empty after cache restore.

Root cause: Purity check was preventing mutation detection from running.
The condition was:
    if not skip_cache and not all_calls_pure:
        mutated_vars_early = MutationDetector.get_top_level_mutated_variables(code)

This meant that if all function calls were pure (e.g., range(), .mean()),
mutation detection was skipped entirely, even though the statement had
in-place mutations (a.append()).

Fix: Always run mutation detection regardless of purity:
    if not skip_cache:
        mutated_vars_early = MutationDetector.get_top_level_mutated_variables(code)
"""
import pytest


@pytest.mark.loops
def test_loop_append_not_cached(nb_runner):
    """
    Test that a.append(...) inside a loop is NOT cached.
    
    The mutation to 'a' should cause the statement to be marked as
    uncacheable and re-executed every time, even if it calls pure functions.
    """
    nb_runner.create_notebook([
        "a = []",
        """for i in range(3):
    a.append(i * 2)""",
        "a"
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # Check output shows correct list
    output = nb_runner.get_output(3)
    assert "[0, 2, 4]" in output or "0, 2, 4" in output
    
    # Re-run the loop cell
    nb_runner.run_cell(2)
    
    # The list should now have 6 elements (appended twice)
    # This proves a.append() is NOT being cached
    nb_runner.get_output(3)
    nb_runner.run_cell(3)
    output3 = nb_runner.get_output(3)
    
    # After running twice, if append was NOT cached, we'd have [0,2,4,0,2,4]
    # If append WAS cached (bug), we'd still have [0,2,4]
    assert "0, 2, 4, 0, 2, 4" in output3 or "[0, 2, 4, 0, 2, 4]" in output3


@pytest.mark.loops  
def test_loop_append_complex_expr_not_cached(nb_runner):
    """
    Test that a.append(...) with complex pure expressions is NOT cached.
    
    This reproduces the actual bug: a.append() calling pure functions like
    range() and .mean() was being cached because purity check prevented
    mutation detection.
    """
    nb_runner.create_notebook([
        "data = [1, 2, 3, 4, 5]",
        "results = []",
        """for i in range(3):
    # Complex expression with pure functions
    value = sum([x * 2 for x in range(i + 1)])
    results.append(value)""",
        "results"
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # First run: results = [0, 2, 6]
    output = nb_runner.get_output(4)
    assert "0" in output and "2" in output and "6" in output
    
    # Clear results and run loop again
    nb_runner.run_cell(2)  # results = []
    nb_runner.run_cell(3)  # for loop
    nb_runner.run_cell(4)  # results
    
    output2 = nb_runner.get_output(4)
    
    # Should still be [0, 2, 6], not empty (which would happen if append was incorrectly cached and restored)
    assert "0" in output2 and "2" in output2 and "6" in output2


@pytest.mark.loops
def test_loop_append_after_restart(nb_runner):
    """
    Test that after kernel restart, loop with a.append() works correctly.
    
    This is the scenario from the user's bug report: after restart, 'a' becomes empty.
    """
    nb_runner.create_notebook([
        "a = []",
        """for ticker in ["A", "B", "C"]:
    a.append({"name": ticker, "value": len(ticker)})""",
        "a"
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # First run: a should have 3 dicts
    output1 = nb_runner.get_output(3)
    assert "A" in output1 and "B" in output1 and "C" in output1
    
    # Simulate kernel restart
    nb_runner.reset_cash_state()
    
    # Run all cells again (should restore from cache)
    nb_runner.run_all()
    
    output2 = nb_runner.get_output(3)
    
    # After restart, 'a' should still have the 3 dicts, not be empty
    assert "A" in output2 and "B" in output2 and "C" in output2
