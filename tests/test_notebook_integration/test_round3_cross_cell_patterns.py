"""
Round 3 Batch 3: Cross-cell patterns, class instances, generators, 
exception recovery, partial re-runs, multi-module cascades.

Tests focus on complex real-world usage patterns that span multiple cells
and exercise the caching framework's ability to track dependencies across
execution boundaries.
"""

import pytest
import time


@pytest.mark.core
class TestClassInstanceCrossCells:
    """Test class instance creation, method calls, and mutation across cells."""

    def test_class_defined_in_one_cell_used_in_another(self, nb_runner):
        """Define a class in cell 1, instantiate in cell 2, use in cell 3."""
        nb_runner.create_notebook([
            "class Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n        return self.n",
            "c = Counter()",
            "r1 = c.inc()\nr2 = c.inc()\nprint(f'r1={r1}, r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "r1=1" in out, f"Got: {out}"
        assert "r2=2" in out, f"Got: {out}"

    def test_class_redefinition_invalidates_instances(self, nb_runner):
        """Redefining a class should invalidate cells using instances of it."""
        nb_runner.create_notebook([
            "class Greeter:\n    def greet(self):\n        return 'hello'",
            "g = Greeter()\nprint(g.greet())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "hello" in nb_runner.get_output(2)

        # Change the class
        nb_runner.set_cell_source(1, "class Greeter:\n    def greet(self):\n        return 'hi there'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "hi there" in out, f"Expected 'hi there', got: {out}"

    def test_inheritance_chain_across_cells(self, nb_runner):
        """Base class in cell 1, child in cell 2, usage in cell 3."""
        nb_runner.create_notebook([
            "class Animal:\n    def speak(self):\n        return 'generic sound'",
            "class Dog(Animal):\n    def speak(self):\n        return 'woof'",
            "d = Dog()\nprint(d.speak())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "woof" in nb_runner.get_output(3)

        # Modify base class — doesn't change Dog.speak(), but tests revalidation
        nb_runner.set_cell_source(1, "class Animal:\n    def speak(self):\n        return 'roar'\n    def name(self):\n        return 'animal'")
        nb_runner.run_all()

        # Dog.speak still returns woof
        assert "woof" in nb_runner.get_output(3)


@pytest.mark.core
class TestGeneratorAndIteratorCaching:
    """Test caching behavior with generators and iterators."""

    def test_list_comprehension_from_range(self, nb_runner):
        """List comprehension result should be cached."""
        nb_runner.create_notebook([
            "n = 5",
            "squares = [x**2 for x in range(n)]\nprint(squares)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run — should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_dict_comprehension_caching(self, nb_runner):
        """Dict comprehension results cached properly."""
        nb_runner.create_notebook([
            "names = ['alice', 'bob', 'charlie']",
            "name_lens = {n: len(n) for n in names}\nprint(name_lens)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "'alice': 5" in out, f"Got: {out}"
        assert "'bob': 3" in out, f"Got: {out}"

    def test_enumerate_zip_patterns(self, nb_runner):
        """Test common functional patterns."""
        nb_runner.create_notebook([
            "xs = [10, 20, 30]\nys = ['a', 'b', 'c']",
            "pairs = list(zip(xs, ys))\nindexed = list(enumerate(xs))\nprint(f'pairs={pairs}, indexed={indexed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "(10, 'a')" in out, f"Got: {out}"
        assert "(0, 10)" in out, f"Got: {out}"


@pytest.mark.core
class TestExceptionRecovery:
    """Test that errors in one cell don't break caching in subsequent cells."""

    def test_error_cell_doesnt_break_next_cell(self, nb_runner):
        """An error in cell 2 shouldn't prevent cell 3 from running."""
        from nbclient.exceptions import CellExecutionError
        
        nb_runner.create_notebook([
            "x = 42",
            "y = 1/0  # ZeroDivisionError",
            "z = x * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib
        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # This will error
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 84" in out, f"Got: {out}"

    def test_fix_error_and_rerun(self, nb_runner):
        """Fix a broken cell and re-run — should work correctly."""
        from nbclient.exceptions import CellExecutionError
        
        nb_runner.create_notebook([
            "x = 10",
            "y = x + undefined_var",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib
        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # NameError
        
        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x + 5")
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "y = 15" in out, f"Got: {out}"


@pytest.mark.core
class TestPartialReExecution:
    """Test re-running only some cells while others use cache."""

    def test_rerun_middle_cell_only(self, nb_runner):
        """Run all, then only re-run cell 2 — cell 3 should still have correct output."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 3",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 31" in nb_runner.get_output(3)

        # Only re-run cell 3 (without running cells 1 and 2 again)
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "z = 31" in out, f"Expected z=31, got: {out}"

    def test_skip_to_last_cell(self, nb_runner):
        """Run only the last cell — upstream system should restore deps."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 2",
            "c = b + 3\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        # Run all to populate cache
        nb_runner.run_all()
        assert "c = 6" in nb_runner.get_output(3)

        # Now only run cell 3 — should restore a and b from cache/upstream
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "c = 6" in out, f"Expected c=6, got: {out}"



@pytest.mark.modules
class TestFromImportCascadeChain:
    """Test from-import with multi-cell cascade chains."""

    def test_from_import_used_in_three_cells(self, nb_runner, tmp_path):
        """
        from X import Y in cell 1, used in cells 2 and 3.
        Module change should invalidate both downstream cells.
        """
        mod_path = tmp_path / "calc.py"
        mod_path.write_text("def double(x):\n    return x * 2\n")

        nb_runner.create_notebook([
            "from calc import double",
            "a = double(5)\nprint(f'a = {a}')",
            "b = double(a)\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 20" in nb_runner.get_output(3)

        # Change double to triple
        mod_path.write_text("def double(x):\n    return x * 3\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        assert "a = 15" in out2, f"Cell 2: expected a=15, got: {out2}"
        assert "b = 45" in out3, f"Cell 3: expected b=45, got: {out3}"

    def test_from_import_multiple_names(self, nb_runner, tmp_path):
        """
        from X import Y, Z — both names should update when module changes.
        """
        mod_path = tmp_path / "ops.py"
        mod_path.write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def sub(a, b):\n    return a - b\n"
        )

        nb_runner.create_notebook([
            "from ops import add, sub",
            "r1 = add(10, 3)\nr2 = sub(10, 3)\nprint(f'add={r1}, sub={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "add=13" in out, f"Got: {out}"
        assert "sub=7" in out, f"Got: {out}"

        # Change add to multiply
        mod_path.write_text(
            "def add(a, b):\n    return a * b\n\n"
            "def sub(a, b):\n    return a - b\n"
        )
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "add=30" in out2, f"Expected add=30, got: {out2}"
        assert "sub=7" in out2, f"sub should still be 7, got: {out2}"


@pytest.mark.modules
class TestMultiModuleDependencies:
    """Test scenarios with multiple interdependent modules."""

    def test_two_modules_one_cell(self, nb_runner, tmp_path):
        """Import two different modules and use both."""
        mod_a = tmp_path / "mod_a.py"
        mod_a.write_text("def fa(x):\n    return x + 1\n")
        mod_b = tmp_path / "mod_b.py"
        mod_b.write_text("def fb(x):\n    return x * 2\n")

        nb_runner.create_notebook([
            "import mod_a\nimport mod_b",
            "r = mod_a.fa(mod_b.fb(5))\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r = 11" in nb_runner.get_output(2)

        # Change mod_b
        mod_b.write_text("def fb(x):\n    return x * 3\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "r = 16" in out, f"Expected r=16, got: {out}"

    def test_module_importing_module(self, nb_runner, tmp_path):
        """
        Module A imports Module B. Change B, both should update.
        """
        mod_b = tmp_path / "helper.py"
        mod_b.write_text("FACTOR = 10\n")
        mod_a = tmp_path / "processor.py"
        mod_a.write_text("from helper import FACTOR\ndef process(x):\n    return x * FACTOR\n")

        nb_runner.create_notebook([
            "import processor",
            "result = processor.process(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 50" in nb_runner.get_output(2)

        # Change the helper module's FACTOR
        mod_b.write_text("FACTOR = 100\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        # This tests transitive dependency detection
        assert "result = 500" in out, f"Expected result=500 after helper change, got: {out}"


@pytest.mark.files
class TestFileOperationPatterns:
    """Test various file operation patterns and their caching behavior."""

    def test_write_then_read_same_cell(self, nb_runner, tmp_path):
        """Write and read a file in the same cell."""
        nb_runner.create_notebook([
            f"import json\ndata = {{'key': 'value'}}\nwith open(r'{(tmp_path / 'test.json').as_posix()}', 'w') as f:\n    json.dump(data, f)",
            f"import json\nwith open(r'{(tmp_path / 'test.json').as_posix()}') as f:\n    loaded = json.load(f)\nprint(loaded['key'])",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "value" in nb_runner.get_output(2)

    def test_csv_with_different_separators(self, nb_runner, tmp_path):
        """Read CSV with semicolons."""
        csv_path = tmp_path / "semi.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}', sep=';')",
            "print(df.sum().to_string())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "5" in out, f"Expected sum of column a=5, got: {out}"


@pytest.mark.upstream
class TestComplexUpstreamPatterns:
    """Test complex upstream dependency resolution patterns."""

    def test_diamond_dependency(self, nb_runner):
        """
        Cell 1 → Cell 2 and Cell 3 → Cell 4 (diamond).
        Modify cell 1, run cell 4 — should cascade through both paths.
        """
        nb_runner.create_notebook([
            "x = 10",
            "a = x + 1",
            "b = x + 2",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 23" in nb_runner.get_output(4)

        # Modify the root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(4)

        out = nb_runner.get_output(4)
        assert "c = 203" in out, f"Expected c=203, got: {out}"

    def test_long_chain_six_cells(self, nb_runner):
        """Six-cell chain: each transforms the previous."""
        nb_runner.create_notebook([
            "x = 1",
            "x2 = x + 1",
            "x3 = x2 + 1",
            "x4 = x3 + 1",
            "x5 = x4 + 1",
            "x6 = x5 + 1\nprint(f'x6 = {x6}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x6 = 6" in nb_runner.get_output(6)

        # Modify root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(6)

        out = nb_runner.get_output(6)
        assert "x6 = 105" in out, f"Expected x6=105, got: {out}"

    def test_independent_branches_no_interference(self, nb_runner):
        """
        Two independent branches should not interfere with each other.
        Cell 1: x = 10
        Cell 2: a = x + 1
        Cell 3: y = 20 (independent)
        Cell 4: b = y + 1 (depends only on y)
        Modifying x should not re-execute cell 4.
        """
        nb_runner.create_notebook([
            "x = 10",
            "a = x + 1\nprint(f'a = {a}')",
            "y = 20",
            "b = y + 1\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(4)

        # Modify x — only cell 2 should change, cell 4 stays
        nb_runner.set_cell_source(1, "x = 99")
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out4 = nb_runner.get_output(4)
        assert "a = 100" in out2, f"Expected a=100, got: {out2}"
        assert "b = 21" in out4, f"Expected b=21 unchanged, got: {out4}"


@pytest.mark.core
class TestStringAndFormattingPatterns:
    """Test caching with various string operations."""

    def test_fstring_interpolation(self, nb_runner):
        """f-strings with complex expressions."""
        nb_runner.create_notebook([
            "name = 'World'\ncount = 3",
            "msg = f'{name}! ' * count\nprint(msg.strip())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "World! World! World!" in nb_runner.get_output(2)

    def test_multiline_string_processing(self, nb_runner):
        """Multiline string split/join operations."""
        nb_runner.create_notebook([
            "text = '''line1\nline2\nline3'''",
            "lines = text.split('\\n')\nresult = ' | '.join(lines)\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "line1 | line2 | line3" in nb_runner.get_output(2)


@pytest.mark.core
class TestCollectionPatterns:
    """Test caching with various collection manipulations."""


    def test_set_operations_across_cells(self, nb_runner):
        """Set operations using variables from different cells."""
        nb_runner.create_notebook([
            "s1 = {1, 2, 3, 4, 5}",
            "s2 = {3, 4, 5, 6, 7}",
            "inter = s1 & s2\nunion = s1 | s2\ndiff = s1 - s2\nprint(f'inter={sorted(inter)}, union={sorted(union)}, diff={sorted(diff)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "inter=[3, 4, 5]" in out, f"Got: {out}"
        assert "union=[1, 2, 3, 4, 5, 6, 7]" in out, f"Got: {out}"
        assert "diff=[1, 2]" in out, f"Got: {out}"

    def test_defaultdict_pattern(self, nb_runner):
        """defaultdict accumulation across cells."""
        nb_runner.create_notebook([
            "from collections import defaultdict\nword_count = defaultdict(int)",
            "for w in ['hello', 'world', 'hello', 'python', 'world', 'hello']:\n    word_count[w] += 1",
            "print(dict(word_count))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "'hello': 3" in out, f"Got: {out}"
        assert "'world': 2" in out, f"Got: {out}"


@pytest.mark.core
class TestConditionalExecution:
    """Test conditional patterns across cells."""

    def test_conditional_variable_assignment(self, nb_runner):
        """Condition in cell 1 affects cell 2's computation."""
        nb_runner.create_notebook([
            "mode = 'double'",
            "x = 10\nresult = x * 2 if mode == 'double' else x * 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'triple'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 30" in out, f"Expected 30, got: {out}"

    def test_early_return_pattern_with_function(self, nb_runner):
        """Function with early return, called cross-cell."""
        nb_runner.create_notebook([
            "def process(val):\n    if val < 0:\n        return 'negative'\n    if val == 0:\n        return 'zero'\n    return f'positive: {val}'",
            "r1 = process(-5)\nr2 = process(0)\nr3 = process(42)\nprint(f'{r1}, {r2}, {r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "negative" in out, f"Got: {out}"
        assert "zero" in out, f"Got: {out}"
        assert "positive: 42" in out, f"Got: {out}"


@pytest.mark.core
class TestGlobalStateInteraction:
    """Test caching with global/module-level state modifications."""

    def test_counter_function_with_closure(self, nb_runner):
        """Closure-based counter — state changes across calls."""
        nb_runner.create_notebook([
            "def make_counter():\n    count = [0]\n    def inc():\n        count[0] += 1\n        return count[0]\n    return inc",
            "counter = make_counter()",
            "r1 = counter()\nr2 = counter()\nprint(f'r1={r1}, r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "r1=1" in out, f"Got: {out}"
        assert "r2=2" in out, f"Got: {out}"

    def test_memoization_pattern(self, nb_runner):
        """Test a memoized function pattern."""
        nb_runner.create_notebook([
            "def memoize(f):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = f(*args)\n        return cache[args]\n    return wrapper",
            "@memoize\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            "r = fib(10)\nprint(f'fib(10) = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "fib(10) = 55" in nb_runner.get_output(3)


@pytest.mark.core
class TestNumericPatterns:
    """Test caching with various numeric operations."""

    def test_complex_math_chain(self, nb_runner):
        """Chain of mathematical operations across cells."""
        nb_runner.create_notebook([
            "import math\nx = 2.0",
            "y = math.sqrt(x) + math.log(x)",
            "z = round(y ** 2, 4)\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        # sqrt(2) ≈ 1.4142, log(2) ≈ 0.6931, sum ≈ 2.1073, squared ≈ 4.4409
        assert "z = 4.44" in out, f"Got: {out}"

    def test_boolean_logic_chain(self, nb_runner):
        """Boolean operations across cells."""
        nb_runner.create_notebook([
            "a = True\nb = False\nc = True",
            "d = (a and c) or b\ne = not (a and b)\nf = a ^ c\nprint(f'd={d}, e={e}, f={f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "d=True" in out, f"Got: {out}"
        assert "e=True" in out, f"Got: {out}"
        assert "f=False" in out, f"Got: {out}"


@pytest.mark.core
class TestTryCatchPatterns:
    """Test try/except patterns in cached cells."""



    def test_try_except_switch_paths(self, nb_runner):
        """Change input to switch from success to error path."""
        nb_runner.create_notebook([
            "x = '42'",
            "try:\n    val = int(x)\nexcept ValueError:\n    val = -1\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change to trigger error path
        nb_runner.set_cell_source(1, "x = 'abc'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "val = -1" in out, f"Expected val=-1, got: {out}"
