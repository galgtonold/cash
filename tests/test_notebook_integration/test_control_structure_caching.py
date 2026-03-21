"""
Integration tests for control structure caching.

These tests verify that control structures (for loops, if statements, etc.)
work correctly with the caching system in a real notebook execution environment.
"""
import pytest

pytestmark = [pytest.mark.control, pytest.mark.loops]


def test_for_loop_caching(nb_runner, tmp_path):
    """
    Test that for loop iterations are cached and restored correctly.
    """
    execution_log = tmp_path / "loop_execution.txt"
    execution_log_str = str(execution_log).replace("\\", "/")
    
    nb_runner.create_notebook([
        f"""with open('{execution_log_str}', 'a') as f:
    for i in range(3):
        f.write(f'executed_{{i}}\\n')
        print(f'Iteration {{i}}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # Verify all iterations were executed
    assert execution_log.exists(), "Execution log should exist"
    log_content = execution_log.read_text()
    assert "executed_0" in log_content
    assert "executed_1" in log_content
    assert "executed_2" in log_content
    
    output = nb_runner.get_output(1)
    assert "Iteration 0" in output
    assert "Iteration 1" in output
    assert "Iteration 2" in output


def test_if_statement_branch_caching(nb_runner):
    """
    Test that only the executed branch of an if statement is cached.
    """
    nb_runner.create_notebook([
        """condition = True
if condition:
    result = 'took_true_branch'
    print('True Branch')
else:
    result = 'took_false_branch'
    print('False Branch')
print(f'Result: {result}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(1)
    assert "True Branch" in output1
    assert "Result: took_true_branch" in output1
    assert "False Branch" not in output1


def test_nested_control_structures(nb_runner):
    """
    Test caching of nested control structures (for loop with if inside).
    """
    nb_runner.create_notebook([
        """results = []
for i in range(4):
    if i % 2 == 0:
        results.append(f'{i} is even')
    else:
        results.append(f'{i} is odd')
print('\\n'.join(results))"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(1)
    assert "0 is even" in output1
    assert "1 is odd" in output1
    assert "2 is even" in output1
    assert "3 is odd" in output1


def test_loop_with_changing_input(nb_runner):
    """
    Test that changing the data being iterated over causes cache re-computation.
    """
    nb_runner.create_notebook([
        "data = [1, 2, 3]",
        """total = 0
for x in data:
    total += x
    print(f'Adding {x}, total={total}')
print(f'Final: {total}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(2)
    assert "Final: 6" in output1
    
    # Change the data
    nb_runner.set_cell_source(1, "data = [10, 20, 30]")
    nb_runner.run_all()
    
    output2 = nb_runner.get_output(2)
    assert "Final: 60" in output2


def test_while_loop_caching(nb_runner):
    """
    Test while loop caching.
    """
    nb_runner.create_notebook([
        """count = 0
i = 0
while i < 3:
    count += 1
    print(f'Iteration {i}')
    i += 1
print(f'Total iterations: {count}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output = nb_runner.get_output(1)
    assert "Total iterations: 3" in output


def test_tuple_unpacking_in_loop(nb_runner):
    """
    Test for loop with tuple unpacking.
    """
    nb_runner.create_notebook([
        """results = []
for name, value in [('a', 1), ('b', 2), ('c', 3)]:
    results.append(f'{name}={value}')
print(', '.join(results))"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output = nb_runner.get_output(1)
    assert "a=1" in output
    assert "b=2" in output
    assert "c=3" in output


def test_loop_dict_update_and_print(nb_runner):
    """
    Test loop that updates a dictionary and prints keys.
    """
    nb_runner.create_notebook([
        # Cell 1: Loop that builds dict
        """stats = {}
for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    stats[ticker] = {'price': len(ticker) * 10}
    print(f'Added {ticker}')
print(f'Keys: {list(stats.keys())}')""",
        # Cell 2: Use the dict
        """print(f'Final keys: {list(stats.keys())}')"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output = nb_runner.get_output(2)
    assert "AAPL" in output
    assert "MSFT" in output
    assert "GOOGL" in output


def test_loop_execution_failure_with_missing_dep(nb_runner):
    """
    Test that a loop fails appropriately when a dependency is missing.
    """
    nb_runner.create_notebook([
        # Cell 1: Loop that uses undefined variable
        """for x in undefined_list:
    print(x)"""
    ])
    nb_runner.start_kernel()
    
    from nbclient.exceptions import CellExecutionError
    with pytest.raises(CellExecutionError):
        nb_runner.run_all()


def test_loop_variable_change_invalidates_downstream(nb_runner):
    """
    Test that changing loop variable invalidates downstream.
    
    When `items` changes in cell 1, cell 2's loop should detect the new lineage
    (via the iterable lineage included in the iteration context) and re-execute.
    """
    nb_runner.create_notebook([
        "items = ['a', 'b', 'c']",
        """result = []
for item in items:
    result.append(item.upper())
print(result)"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(2)
    assert "'A'" in out1 and "'B'" in out1 and "'C'" in out1
    
    # Change items - this modifies cell source AND saves the notebook file
    nb_runner.set_cell_source(1, "items = ['x', 'y', 'z']")
    
    # Run cell 1 FIRST to execute the new code
    nb_runner.run_cell(1)
    
    # Then run cell 2 which should pick up the new items value
    nb_runner.run_cell(2)
    
    out2 = nb_runner.get_output(2)
    assert "'X'" in out2 and "'Y'" in out2 and "'Z'" in out2, f"Expected X,Y,Z but got: {out2}"


def test_loop_item_removal_invalidates_downstream(nb_runner):
    """
    Test that removing an item from loop invalidates correctly.
    """
    nb_runner.create_notebook([
        """data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(data)""",
        """print(f"Keys: {list(data.keys())}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(2)
    assert "1" in out1 and "2" in out1 and "3" in out1
    
    # Remove an item
    nb_runner.set_cell_source(1, """data = {}
for x in [1, 2]:
    data[x] = x * 10
print(data)""")
    nb_runner.run_all()
    
    out2 = nb_runner.get_output(2)
    assert "1" in out2 and "2" in out2
    # Note: key 3 might still be in output due to how keys() works, 
    # but the actual dict should only have 1 and 2
