"""
Stress Test Batch 3: Loops, Mutations, Control Structures (Scenarios 66-95)

Tests loop caching (empty, single, break, continue, nested), mutation detection,
while loops, if/else branches, and side effect handling.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.loops, pytest.mark.mutations]


# =============================================================================
# Scenario 66-85: Loop & Control Structure Edge Cases
# =============================================================================


class TestLoopEdgeCases:
    """Tests for loop caching edge cases."""

    def test_66_empty_loop(self, nb_runner):
        """Scenario 66: Empty loop — no iterations."""
        nb_runner.create_notebook([
            "items = []",
            "results = []\nfor x in items:\n    results.append(x * 2)\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[]" in nb_runner.get_output(2)


    def test_68_loop_with_break(self, nb_runner):
        """Scenario 68: Loop with break — fallback to single-unit."""
        nb_runner.create_notebook([
            "result = 0\nfor i in range(10):\n    if i > 2:\n        break\n    result += i\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # 0 + 1 + 2 = 3
        assert "result=3" in nb_runner.get_output(1)

    def test_69_loop_with_continue(self, nb_runner):
        """Scenario 69: Loop with continue — fallback to single-unit."""
        nb_runner.create_notebook([
            "result = 0\nfor i in range(5):\n    if i == 2:\n        continue\n    result += i\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # 0 + 1 + 3 + 4 = 8
        assert "result=8" in nb_runner.get_output(1)

    def test_70_nested_loop(self, nb_runner):
        """Scenario 70: Nested loop."""
        nb_runner.create_notebook([
            "results = {}\nfor i in ['a', 'b']:\n    results[i] = {}\n    for j in [1, 2]:\n        results[i][j] = f'{i}{j}'\nprint(results)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        out = nb_runner.get_output(1)
        assert "'a'" in out and "'b'" in out

    def test_71_loop_modifying_external_accumulator(self, nb_runner):
        """Scenario 71: Loop appending to external list."""
        nb_runner.create_notebook([
            "data = [10, 20, 30]",
            "total = 0\nfor x in data:\n    total += x\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)
        # Change data
        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40]")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "total=100" in nb_runner.get_output(2)

    def test_72_loop_with_function_call(self, nb_runner):
        """Scenario 72: Loop calling function defined in another cell."""
        nb_runner.create_notebook([
            "def process(x):\n    return x ** 2",
            "data = [1, 2, 3]",
            "results = []\nfor x in data:\n    results.append(process(x))\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 4, 9]" in nb_runner.get_output(3)
        # Change function
        nb_runner.set_cell_source(1, "def process(x):\n    return x ** 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "results=[1, 8, 27]" in nb_runner.get_output(3)

    def test_73_while_loop_basic(self, nb_runner):
        """Scenario 73: Basic while loop."""
        nb_runner.create_notebook([
            "x = 0\nwhile x < 5:\n    x += 1\nprint(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "x=5" in nb_runner.get_output(1)

    def test_74_if_else_branch_caching(self, nb_runner):
        """Scenario 75: If/else — change condition, verify correct branch."""
        nb_runner.create_notebook([
            "n = 10",
            "if n > 5:\n    result = 'big'\nelse:\n    result = 'small'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=big" in nb_runner.get_output(2)
        # Change n to switch branch
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "result=small" in nb_runner.get_output(2)

    def test_75_if_elif_else_chain(self, nb_runner):
        """Scenario 76: If/elif/else chain."""
        nb_runner.create_notebook([
            "score = 85",
            "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelif score >= 70:\n    grade = 'C'\nelse:\n    grade = 'F'\nprint(f'grade={grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade=B" in nb_runner.get_output(2)
        # Change to A range
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "grade=A" in nb_runner.get_output(2)

    def test_76_loop_with_tuple_unpacking(self, nb_runner):
        """Scenario 77: Loop with tuple unpacking."""
        nb_runner.create_notebook([
            "pairs = [(1, 'a'), (2, 'b'), (3, 'c')]",
            "results = {}\nfor num, letter in pairs:\n    results[letter] = num * 10\nprint(results)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 10" in out and "'b': 20" in out

    def test_77_loop_variable_escapes(self, nb_runner):
        """Scenario 81: After loop, loop variable has last value."""
        nb_runner.create_notebook([
            "data = [10, 20, 30]",
            "for val in data:\n    pass\nprint(f'last_val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "last_val=30" in nb_runner.get_output(2)

    def test_78_loop_then_downstream_uses_loop_var(self, nb_runner):
        """Loop variable used in downstream cell after loop."""
        nb_runner.create_notebook([
            "for i in range(5):\n    pass",
            "print(f'i={i}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "i=4" in nb_runner.get_output(2)

    def test_79_loop_body_multi_statement(self, nb_runner):
        """Loop body with multiple statements."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "sums = []\nprods = []\nfor x in data:\n    s = x + 10\n    p = x * 10\n    sums.append(s)\n    prods.append(p)\nprint(f'sums={sums}')\nprint(f'prods={prods}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sums=[11, 12, 13]" in out
        assert "prods=[10, 20, 30]" in out

    def test_80_loop_change_iterations_downstream_correct(self, nb_runner):
        """Change loop iterations, downstream cell gets correct result."""
        nb_runner.create_notebook([
            "d = {}\nfor k in ['a', 'b']:\n    d[k] = len(k)",
            "print(f'keys={sorted(d.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['a', 'b']" in nb_runner.get_output(2)
        # Add an iteration
        nb_runner.set_cell_source(1, "d = {}\nfor k in ['a', 'b', 'cc']:\n    d[k] = len(k)")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "keys=['a', 'b', 'cc']" in nb_runner.get_output(2)

    def test_81_nested_if_inside_loop(self, nb_runner):
        """Scenario 82: Nested if inside loop."""
        nb_runner.create_notebook([
            "data = [1, -2, 3, -4, 5]",
            "pos = []\nneg = []\nfor x in data:\n    if x > 0:\n        pos.append(x)\n    else:\n        neg.append(x)\nprint(f'pos={pos}, neg={neg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "pos=[1, 3, 5]" in out
        assert "neg=[-2, -4]" in out

    def test_82_loop_over_dict_items(self, nb_runner):
        """Scenario 84: Loop over dict.items()."""
        nb_runner.create_notebook([
            "config = {'a': 1, 'b': 2, 'c': 3}",
            "total = 0\nfor key, val in config.items():\n    total += val\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=6" in nb_runner.get_output(2)


# =============================================================================
# Scenario 86-95: Mutation Detection & Side Effects
# =============================================================================


class TestMutationDetection:
    """Tests for mutation detection and side effect handling."""

    def test_83_list_append_in_loop(self, nb_runner):
        """Scenario 86: List append in loop — mutation tracked."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "results = []\nfor x in data:\n    results.append(x * 2)\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[2, 4, 6]" in nb_runner.get_output(2)
        # Change data
        nb_runner.set_cell_source(1, "data = [10, 20]")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "results=[20, 40]" in nb_runner.get_output(2)


    def test_85_augmented_assignment_on_list(self, nb_runner):
        """Scenario 89: lst += [...] — augmented assignment on collection."""
        nb_runner.create_notebook([
            "lst = [1, 2]",
            "lst += [3, 4]\nprint(f'lst={lst}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lst=[1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_86_chained_method_not_mutation(self, nb_runner):
        """Scenario 90: df.drop().reset_index() — NOT a mutation of df."""
        nb_runner.create_notebook([
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2], 'b': [3,4]})",
            "result = df.drop(columns=['b']).reset_index(drop=True)\nprint(result.columns.tolist())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a']" in nb_runner.get_output(2)
        # Re-run — df is unchanged, should skip/cache
        nb_runner.run_cell(2)
        assert "['a']" in nb_runner.get_output(2)

    def test_87_attribute_assignment_mutation(self, nb_runner):
        """Scenario 93: obj.attr = value — mutation detected."""
        nb_runner.create_notebook([
            "class Obj:\n    def __init__(self):\n        self.x = 1\nobj = Obj()",
            "obj.x = 42\nprint(f'obj.x={obj.x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "obj.x=42" in nb_runner.get_output(2)

    def test_88_mutation_then_downstream(self, nb_runner):
        """Mutation in one cell affects downstream correctly."""
        nb_runner.create_notebook([
            "lst = [1, 2, 3]",
            "lst.append(4)",
            "total = sum(lst)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=10" in nb_runner.get_output(3)

    def test_89_inplace_pandas_mutation(self, nb_runner):
        """Scenario 88: DataFrame inplace operation."""
        nb_runner.create_notebook([
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2], 'b': [3,4]})",
            "df.drop(columns=['b'], inplace=True)\nprint(df.columns.tolist())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a']" in nb_runner.get_output(2)

    def test_90_nested_dict_mutation(self, nb_runner):
        """Scenario 92: d['key']['subkey'] = value."""
        nb_runner.create_notebook([
            "d = {'outer': {}}",
            "d['outer']['inner'] = 42\nprint(f\"d={d}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'inner': 42" in out

    def test_91_loop_accumulator_correct_after_change(self, nb_runner):
        """Loop accumulator reflects changes after loop code modification."""
        nb_runner.create_notebook([
            "nums = [1, 2, 3]",
            "acc = 0\nfor n in nums:\n    acc += n\nprint(f'acc={acc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "acc=6" in nb_runner.get_output(2)
        # Change accumulator operation
        nb_runner.set_cell_source(2, "acc = 1\nfor n in nums:\n    acc *= n\nprint(f'acc={acc}')")
        nb_runner.run_cell(2)
        assert "acc=6" in nb_runner.get_output(2)

    def test_92_dataframe_column_add_mutation(self, nb_runner):
        """Adding column via subscript is a mutation."""
        nb_runner.create_notebook([
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})",
            "df['b'] = df['a'] * 2\nprint(df.columns.tolist())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a', 'b']" in nb_runner.get_output(2)

    def test_93_multiple_mutations_same_cell(self, nb_runner):
        """Multiple mutations in same cell — all tracked."""
        nb_runner.create_notebook([
            "lst = []\nd = {}",
            "lst.append(1)\nlst.append(2)\nd['a'] = 10\nd['b'] = 20\nprint(f'lst={lst}, d={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lst=[1, 2]" in out
        assert "'a': 10" in out

    def test_94_mutation_no_output_variable(self, nb_runner):
        """Mutation with no new output variable — just modifying existing."""
        nb_runner.create_notebook([
            "data = {'count': 0}",
            "data['count'] += 1\nprint(f\"count={data['count']}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1" in nb_runner.get_output(2)
        # Re-run — data['count'] is now 1, += 1 = 2
        nb_runner.run_cell(2)
        assert "count=2" in nb_runner.get_output(2)

    def test_95_set_operations_mutation(self, nb_runner):
        """Set.add(), set.update() — mutations detected."""
        nb_runner.create_notebook([
            "s = set()",
            "s.add(1)\ns.add(2)\ns.update([3, 4])\nprint(f's={sorted(s)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s=[1, 2, 3, 4]" in nb_runner.get_output(2)
