"""
Test for accumulator pattern when adding a new item to loop list.

This tests the scenario where:
1. A loop builds up an accumulator dict over several iterations
2. All iterations are cached
3. A new item is added to the loop list
4. Running a downstream cell should compute the new iteration AND keep the old values
"""
import pytest

pytestmark = [pytest.mark.mutations, pytest.mark.loops]


def test_accumulator_add_item_fresh_session(nb_runner):
    """
    Test that adding a new item to a loop list correctly shows ALL items,
    not just the new one.
    
    Scenario:
    1. Fresh session (no kernel state)
    2. Loop with 4 items is cached
    3. Add 5th item to the list
    4. Run cell that uses the accumulator
    5. Should see all 5 items
    """
    # First run: cache 4 iterations
    nb_runner.create_notebook([
        # Cell 1: Simple accumulator loop with 4 items
        """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
print(f"Run 1: {list(results.keys())}")""",
        # Cell 2: Use the accumulator
        """print(f"Keys: {list(results.keys())}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(1)
    assert "['A', 'B', 'C', 'D']" in output1 or "dict_keys(['A', 'B', 'C', 'D'])" in output1
    
    output2 = nb_runner.get_output(2)
    assert "'A'" in output2 and "'B'" in output2 and "'C'" in output2 and "'D'" in output2
    
    # Modify loop to have 5 items
    nb_runner.set_cell_source(1, """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
print(f"Run 2: {list(results.keys())}")""")
    
    nb_runner.run_all()
    
    output3 = nb_runner.get_output(1)
    print(f"After modification, Cell 1 output: {output3}")
    
    output4 = nb_runner.get_output(2)
    print(f"After modification, Cell 2 output: {output4}")
    
    # Critical assertion: ALL 5 items should be present
    assert "'A'" in output4, f"Missing 'A' in output: {output4}"
    assert "'B'" in output4, f"Missing 'B' in output: {output4}"
    assert "'C'" in output4, f"Missing 'C' in output: {output4}"
    assert "'D'" in output4, f"Missing 'D' in output: {output4}"
    assert "'E'" in output4, f"Missing 'E' in output: {output4}"


def test_accumulator_add_item_skip_loop_cell(nb_runner):
    """
    Test the exact user scenario: 
    1. Run loop cell (caches 4 iterations)
    2. Run keys cell 
    3. Modify loop to add 5th item
    4. Run ONLY the keys cell (not the loop cell)
    5. The upstream checker should handle restoration + new computation
    """
    nb_runner.create_notebook([
        # Cell 1: Loop with 4 items
        """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
print(f"Loop done: {list(results.keys())}")""",
        # Cell 2: Print keys
        """print(f"Keys: {list(results.keys())}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(2)
    print(f"Run 1 output: {output1}")
    assert "'D'" in output1
    
    # Modify loop to add 5th item
    nb_runner.set_cell_source(1, """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
print(f"Loop done: {list(results.keys())}")""")
    
    # Run ONLY cell 2 - upstream checker should detect the change
    nb_runner.run_cell(2)
    
    output2 = nb_runner.get_output(2)
    print(f"After modification, Cell 2 output: {output2}")
    
    # All 5 should be present
    for item in ["A", "B", "C", "D", "E"]:
        assert f"'{item}'" in output2, f"Missing '{item}' in: {output2}"


def test_accumulator_only_run_downstream_cell(nb_runner):
    """
    Test running only the downstream cell after adding a new item.
    The upstream checker should detect the change and re-execute.
    """
    nb_runner.create_notebook([
        # Cell 1: Loop with 3 items
        """data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(f"After loop: {data}")""",
        # Cell 2: Use data
        """print(f"Downstream: {data}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(2)
    assert "1: 10" in out1 and "2: 20" in out1 and "3: 30" in out1
    
    # Add 4th item
    nb_runner.set_cell_source(1, """data = {}
for x in [1, 2, 3, 4]:
    data[x] = x * 10
print(f"After loop: {data}")""")
    
    # Run only downstream cell
    nb_runner.run_cell(2)
    
    out2 = nb_runner.get_output(2)
    print(f"After modification: {out2}")
    
    for x in [1, 2, 3, 4]:
        assert f"{x}: {x*10}" in out2, f"Missing {x} in: {out2}"


def test_accumulator_fresh_session_only_downstream(nb_runner):
    """
    Test with fresh session, run only downstream cell.
    """
    nb_runner.create_notebook([
        # Cell 1: Define data
        """data = {'a': 1, 'b': 2}""",
        # Cell 2: Use data
        """print(f"Keys: {list(data.keys())}")"""
    ])
    nb_runner.start_kernel()
    
    # Run only cell 2 - should trigger upstream execution of cell 1
    nb_runner.run_cell(2)
    
    out = nb_runner.get_output(2)
    assert "'a'" in out and "'b'" in out


def test_accumulator_upstream_triggered(nb_runner):
    """
    Test that running downstream cell triggers upstream execution.
    """
    nb_runner.create_notebook([
        # Cell 1: Define x
        """x = 10
print(f"x = {x}")""",
        # Cell 2: Use x
        """y = x * 2
print(f"y = {y}")""",
        # Cell 3: Use y
        """z = y + 5
print(f"z = {z}")"""
    ])
    nb_runner.start_kernel()
    
    # Run only cell 3 - should trigger cells 1 and 2
    nb_runner.run_cell(3)
    
    out = nb_runner.get_output(3)
    assert "z = 25" in out  # 10 * 2 + 5 = 25
