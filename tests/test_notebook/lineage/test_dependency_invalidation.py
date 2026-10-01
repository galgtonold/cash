"""
Test for dependency invalidation when upstream cells change.
"""

from tests._cell_driver import run_cash_cell


def test_upstream_cell_change_invalidates_cache(cash_magics, mock_shell):
    """
    Test scenario:
    1. Cell 1: selected_region = 'South'
    2. Cell 2: result = f"Region: {selected_region}"
    3. Run both - Cell 2 caches result
    4. Change Cell 1 to selected_region = 'North'
    5. Run Cell 2 - should detect Cell 1 changed and re-execute it first
    """

    # Step 1: Set variable
    cell1_v1 = "selected_region = 'South'"
    run_cash_cell(cash_magics, cell1_v1)

    assert mock_shell.user_ns["selected_region"] == "South", "Step 1: selected_region should be South"

    # Step 2: Use variable (will be cached)
    cell2 = "result = f'Region: {selected_region}'"
    run_cash_cell(cash_magics, cell2)

    assert mock_shell.user_ns["result"] == "Region: South", 'Step 2: result should be "Region: South"'

    # Step 3: Run Cell 2 again (should get cache hit)
    run_cash_cell(cash_magics, cell2)

    assert mock_shell.user_ns["result"] == "Region: South", (
        'Step 3: result should still be "Region: South" (from cache)'
    )

    # Step 4: Change Cell 1 code (simulate user editing notebook)
    # In real scenario, Cell 1's code in notebook file changes but hasn't been executed
    # selected_region is still 'South' in memory
    assert mock_shell.user_ns["selected_region"] == "South", "Step 4: selected_region should still be South in memory"

    # Step 5: Run Cell 2 - should detect Cell 1 changed and re-execute it
    # The system should:
    # 1. See that Cell 2 depends on selected_region
    # 2. Check if any earlier cells define selected_region
    # 3. Check if those cells code has changed
    # 4. Re-execute changed cells first
    # 5. Then execute Cell 2 with new value

    # For now, manually simulate what should happen:
    # Execute the NEW version of Cell 1
    cell1_v2 = "selected_region = 'North'"
    run_cash_cell(cash_magics, cell1_v2)

    # Now run Cell 2
    run_cash_cell(cash_magics, cell2)

    # Should have new value
    assert mock_shell.user_ns["result"] == "Region: North", (
        'Step 5: result should be "Region: North" after dependency change'
    )
