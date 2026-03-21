"""
Round 3 - Batch 11: Combined complex patterns stressing multiple subsystems.

Tests focusing on:
1. File deps + upstream simulation combined
2. Module reload + function tracking + from-import combined
3. Kernel restart + file modification + upstream invalidation
4. Multi-output cells + dependency chain + cache invalidation
5. Complex real-world simulation: data loading → processing → analysis → visualization data
6. Nested function definitions with closures across cells
7. Exception handling interleaved with caching
8. Type conversion chains (str→int→float→list→dict)
"""

import pytest
import time



pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestFileDepsWithUpstream:
    """Combined file dependency and upstream simulation patterns."""

    @pytest.mark.files
    @pytest.mark.upstream
    def test_file_dep_change_propagates_through_chain(self, nb_runner, tmp_path):
        """File change should propagate through dependent cells."""
        import pandas as pd

        csv_path = str(tmp_path / "input.csv").replace('\\', '/')
        pd.DataFrame({'val': [1, 2, 3]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{csv_path}')",
            "total = df['val'].sum()",
            "doubled = total * 2",
            "print(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "doubled=12" in output

        # Modify file
        time.sleep(0.1)
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "doubled=120" in output2

    @pytest.mark.files
    @pytest.mark.upstream
    def test_multiple_file_deps_one_changes(self, nb_runner, tmp_path):
        """Multiple file deps, only one changes - verify partial invalidation."""
        import pandas as pd

        file_a = str(tmp_path / "a.csv").replace('\\', '/')
        file_b = str(tmp_path / "b.csv").replace('\\', '/')

        pd.DataFrame({'x': [1]}).to_csv(file_a, index=False)
        pd.DataFrame({'y': [100]}).to_csv(file_b, index=False)

        nb_runner.create_notebook([
            "import pandas as pd",
            f"a = pd.read_csv('{file_a}')['x'].iloc[0]",
            f"b = pd.read_csv('{file_b}')['y'].iloc[0]",
            "result = a + b",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=101" in output

        # Change only file_a
        time.sleep(0.1)
        pd.DataFrame({'x': [50]}).to_csv(file_a, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "result=150" in output2


class TestModuleReloadCombined:
    """Module reload combined with various other features."""

    @pytest.mark.modules
    @pytest.mark.upstream
    def test_module_function_change_propagates_downstream(self, nb_runner, tmp_path):
        """Changing a module function should invalidate all downstream users."""
        mod_path = str(tmp_path / "mymod.py").replace('\\', '/')

        with open(mod_path, 'w') as f:
            f.write("def transform(x): return x * 2\n")

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
            "import mymod",
            "a = mymod.transform(5)",
            "b = a + 10",
            "print(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "b=20" in output  # transform(5)=10, 10+10=20

        # Modify module
        time.sleep(0.1)
        with open(mod_path, 'w') as f:
            f.write("def transform(x): return x * 3\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "b=25" in output2  # transform(5)=15, 15+10=25

    @pytest.mark.modules
    def test_from_import_function_and_constant_mixed(self, nb_runner, tmp_path):
        """Module with both function and constant from-imports."""
        mod_path = str(tmp_path / "config_mod.py").replace('\\', '/')

        with open(mod_path, 'w') as f:
            f.write("VERSION = '1.0'\ndef greet(name): return f'Hello {name} v{VERSION}'\n")

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
            "from config_mod import VERSION, greet",
            "msg = greet('World')",
            "print(f'msg={msg} version={VERSION}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "Hello World v1.0" in output
        assert "version=1.0" in output

        # Update module
        time.sleep(0.1)
        with open(mod_path, 'w') as f:
            f.write("VERSION = '2.0'\ndef greet(name): return f'Hi {name} v{VERSION}'\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "Hi World v2.0" in output2
        assert "version=2.0" in output2


class TestRestartWithFileDeps:
    """Kernel restart combined with file dependency changes."""

    @pytest.mark.restore
    @pytest.mark.files
    def test_restart_after_file_modification(self, nb_runner, tmp_path):
        """Restart kernel after external file mod should recompute."""
        import pandas as pd

        csv_path = str(tmp_path / "restart_test.csv").replace('\\', '/')
        pd.DataFrame({'val': [5, 10, 15]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{csv_path}')",
            "result = df['val'].mean()",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=10.0" in output

        # Modify file and restart
        time.sleep(0.1)
        pd.DataFrame({'val': [100, 200, 300]}).to_csv(csv_path, index=False)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=200.0" in output2


class TestMultiOutputChains:
    """Complex multi-output cell chains."""

    @pytest.mark.core
    @pytest.mark.upstream
    def test_multi_output_diamond_invalidation(self, nb_runner):
        """Multi-output cell feeding into diamond pattern."""
        nb_runner.create_notebook([
            "base = 10",
            "x = base + 1\ny = base + 2\nz = base + 3",
            "left = x * y",     # depends on x, y
            "right = y * z",    # depends on y, z
            "final = left + right",
            "print(f'final={final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        # x=11, y=12, z=13 → left=132, right=156 → final=288
        assert "final=288" in output

        # Change base
        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        # x=101, y=102, z=103 → left=10302, right=10506 → final=20808
        assert "final=20808" in output2

    @pytest.mark.core
    def test_cell_with_many_outputs_selective_use(self, nb_runner):
        """Cell producing many outputs, only some used downstream."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3\nd = 4\ne = 5",
            "# Only use a and e\nresult = a + e",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=6" in output


class TestRealWorldSimulation:
    """Simulates a realistic data analysis workflow."""

    @pytest.mark.stress
    @pytest.mark.files
    def test_full_data_pipeline(self, nb_runner, tmp_path):
        """Complete data pipeline: load → clean → transform → analyze → report."""
        import pandas as pd
        import numpy as np

        # Create test data
        csv_path = str(tmp_path / "sales.csv").replace('\\', '/')
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D').astype(str),
            'product': np.random.choice(['A', 'B', 'C'], 100),
            'quantity': np.random.randint(1, 50, 100),
            'price': np.round(np.random.uniform(10, 100, 100), 2)
        })
        df.to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            f"raw = pd.read_csv('{csv_path}')",
            "raw['date'] = pd.to_datetime(raw['date'])\nraw['revenue'] = raw['quantity'] * raw['price']",
            "by_product = raw.groupby('product')['revenue'].sum().to_dict()",
            "total_revenue = sum(by_product.values())\ntop_product = max(by_product, key=by_product.get)",
            "print(f'total={total_revenue:.0f} top={top_product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "total=" in output
        assert "top=" in output

        # Re-run should cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(6)
        assert output2.strip() == output.strip()

    @pytest.mark.stress
    def test_iterative_model_tuning(self, nb_runner):
        """Simulate iterative parameter tuning."""
        nb_runner.create_notebook([
            "import numpy as np\nnp.random.seed(42)",
            "data = np.random.randn(1000)",
            "threshold = 1.0",
            "above = np.sum(data > threshold)\nbelow = np.sum(data < -threshold)\nwithin = len(data) - above - below",
            "print(f'above={above} below={below} within={within}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "above=" in output1

        # Tune threshold
        nb_runner.set_cell_source(3, "threshold = 2.0")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        # With higher threshold, fewer above/below
        above2 = int(output2.split("above=")[1].split()[0])
        above1 = int(output1.split("above=")[1].split()[0])
        assert above2 < above1  # Higher threshold → fewer above


class TestNestedFunctionClosures:
    """Tests for nested functions and closure patterns across cells."""

    @pytest.mark.core
    def test_closure_over_mutable_state(self, nb_runner):
        """Closure over a list (mutable state)."""
        nb_runner.create_notebook([
            "history = []",
            "def log(msg):\n    history.append(msg)\n    return len(history)",
            "count1 = log('first')\ncount2 = log('second')",
            "print(f'count1={count1} count2={count2} history={history}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "count1=1" in output
        assert "count2=2" in output

    @pytest.mark.core
    def test_higher_order_function_composition(self, nb_runner):
        """Function composition with higher-order functions."""
        nb_runner.create_notebook([
            "def compose(f, g):\n    def composed(x):\n        return f(g(x))\n    return composed",
            "double = lambda x: x * 2\nadd_one = lambda x: x + 1",
            "double_then_add = compose(add_one, double)",
            "result = double_then_add(5)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=11" in output  # double(5)=10, add_one(10)=11

        # Change composition order
        nb_runner.set_cell_source(3, "add_then_double = compose(double, add_one)")
        nb_runner.set_cell_source(4, "result = add_then_double(5)")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "result=12" in output2  # add_one(5)=6, double(6)=12


class TestExceptionInterleaving:
    """Tests for exception handling interleaved with caching."""

    @pytest.mark.core
    def test_try_except_caches_success_path(self, nb_runner):
        """Try/except block should cache the successful result."""
        nb_runner.create_notebook([
            "data = [1, 2, 0, 4]",
            "results = []\nfor d in data:\n    try:\n        results.append(10 / d)\n    except ZeroDivisionError:\n        results.append(None)",
            "valid = [r for r in results if r is not None]",
            "avg = sum(valid) / len(valid)",
            "print(f'avg={avg:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "avg=" in output

    @pytest.mark.core
    def test_assertion_in_notebook_cell(self, nb_runner):
        """Assert statements should work within caching."""
        nb_runner.create_notebook([
            "x = 42",
            "assert x > 0, 'x must be positive'\nresult = x * 2",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=84" in output


class TestTypeConversionChains:
    """Tests for type conversion chains across cells."""

    @pytest.mark.core
    def test_str_to_int_to_float_chain(self, nb_runner):
        """String → int → float conversion chain."""
        nb_runner.create_notebook([
            "raw = '42'",
            "as_int = int(raw)",
            "as_float = float(as_int) / 10",
            "print(f'result={as_float}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=4.2" in output

        # Change input
        nb_runner.set_cell_source(1, "raw = '100'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=10.0" in output2

    @pytest.mark.core
    def test_list_to_dict_to_dataframe(self, nb_runner):
        """List → dict → DataFrame conversion chain."""
        nb_runner.create_notebook([
            "import pandas as pd",
            "names = ['Alice', 'Bob', 'Charlie']",
            "scores = [90, 85, 95]",
            "data_dict = dict(zip(names, scores))",
            "df = pd.DataFrame(list(data_dict.items()), columns=['name', 'score'])",
            "top = df.loc[df['score'].idxmax(), 'name']",
            "print(f'top={top}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(7)
        assert "top=Charlie" in output

        # Change scores
        nb_runner.set_cell_source(3, "scores = [95, 100, 85]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(7)
        assert "top=Bob" in output2

    @pytest.mark.core
    def test_json_roundtrip(self, nb_runner):
        """JSON serialization roundtrip should preserve data."""
        nb_runner.create_notebook([
            "import json",
            "original = {'key': [1, 2, 3], 'nested': {'a': True, 'b': None}}",
            "serialized = json.dumps(original)",
            "restored = json.loads(serialized)",
            "match = original == restored",
            "print(f'match={match} type={type(restored).__name__}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "match=True" in output
        assert "type=dict" in output


class TestComplexLoopPatterns:
    """Tests for complex loop patterns that interact with caching."""

    @pytest.mark.loops
    def test_nested_loop_with_accumulator(self, nb_runner):
        """Nested loops with accumulator pattern."""
        nb_runner.create_notebook([
            "n = 5",
            "matrix = []\nfor i in range(n):\n    row = []\n    for j in range(n):\n        row.append(i * n + j)\n    matrix.append(row)",
            "flat = [x for row in matrix for x in row]",
            "total = sum(flat)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "total=300" in output  # sum(0..24) = 300

        # Change n
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "total=36" in output2  # sum(0..8) = 36

    @pytest.mark.loops
    def test_while_loop_with_convergence(self, nb_runner):
        """While loop that converges to a result."""
        nb_runner.create_notebook([
            "target = 100\ntolerance = 0.01",
            "guess = 1.0\niterations = 0\nwhile abs(guess * guess - target) > tolerance:\n    guess = (guess + target / guess) / 2\n    iterations += 1",
            "print(f'sqrt={guess:.4f} iterations={iterations}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "sqrt=10.0" in output

        # Change target
        nb_runner.set_cell_source(1, "target = 225\ntolerance = 0.01")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "sqrt=15.0" in output2
