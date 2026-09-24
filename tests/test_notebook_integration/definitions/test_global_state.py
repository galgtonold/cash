"""Globals, counters, registries, singletons, scopes and shadowing across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.core
@pytest.mark.timeout(30)
class TestGlobalStateEdits:
    """Global state manipulation + cell edits."""

    def test_global_dict_update(self, nb_runner):
        """Update a global dict across cells, edit one update."""
        nb_runner.create_notebook(
            [
                "config = {}",
                "config['a'] = 1",
                "config['b'] = 2",
                "total = config['a'] + config['b']\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "config['a'] = 100")
        nb_runner.run_all()
        assert "total = 102" in nb_runner.get_output(4)


@pytest.mark.mutations
@pytest.mark.timeout(90)
class TestGlobalCounterEdits:
    """Editing cells that use global counters."""

    def test_counter_increment_edit(self, nb_runner):
        """Edit a counter's increment value."""
        nb_runner.create_notebook(
            [
                "counter = 0  # init counter",
                "counter = counter + 1\nprint(f'counter = {counter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter = 1" in nb_runner.get_output(2)

        # Edit to increment by 5
        nb_runner.set_cell_source(2, "counter = counter + 5\nprint(f'counter = {counter}')")
        nb_runner.run_all()
        assert "counter = 5" in nb_runner.get_output(2)

    def test_counter_init_edit(self, nb_runner):
        """Edit the initial counter value."""
        nb_runner.create_notebook(
            [
                "total = 10  # starting value",
                "total = total * 2\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 20" in nb_runner.get_output(2)

        # Change starting value
        nb_runner.set_cell_source(1, "total = 100  # starting value changed")
        nb_runner.run_all()
        assert "total = 200" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestCounterPatterns:
    """Counter / running total patterns + edits."""

    def test_running_total(self, nb_runner):
        """Running total across cells, edit one addition."""
        nb_runner.create_notebook(
            [
                "total = 0",
                "total = total + 10  # first add",
                "total = total + 20  # second add",
                "total = total + 30  # third add",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "total = total + 200  # second add (boosted)")
        nb_runner.run_all()
        assert "total = 240" in nb_runner.get_output(5)

    def test_flag_based_flow(self, nb_runner):
        """Flag variable controls flow, edit the flag."""
        nb_runner.create_notebook(
            [
                "debug = True",
                "label = str(debug)\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "debug = False")
        nb_runner.run_all()
        assert "label = False" in nb_runner.get_output(2)


@pytest.mark.mutations
@pytest.mark.timeout(90)
class TestAccumulatorEdits:
    """Editing accumulator patterns."""

    def test_list_accumulator_edit_append(self, nb_runner):
        """Edit what gets appended to a list."""
        nb_runner.create_notebook(
            [
                "items = []  # fresh list",
                "for i in range(3):\n    items.append(i)\nprint(f'items = {items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(2)

        # Change to squared values
        nb_runner.set_cell_source(
            2,
            "for i in range(3):\n    items.append(i ** 2)\nprint(f'items = {items}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "0" in out and "1" in out and "4" in out

    def test_dict_accumulator_edit(self, nb_runner):
        """Edit dictionary accumulation logic."""
        nb_runner.create_notebook(
            [
                "data = {}  # fresh dict",
                "for k in ['a', 'b', 'c']:\n    data[k] = len(k)\nprint(f'data = {data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        # Change to uppercase keys
        nb_runner.set_cell_source(
            2,
            "for k in ['a', 'b', 'c']:\n    data[k.upper()] = len(k) * 10\nprint(f'data = {data}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': 10" in out


@pytest.mark.integration
@pytest.mark.timeout(90)
class TestGlobalRegistryInteraction:
    """Test global registry/config patterns with cache invalidation."""

    def test_global_registry_edit(self, nb_runner):
        """Editing a global list should propagate to downstream consumers."""
        nb_runner.create_notebook(
            [
                "REGISTRY = ['alpha', 'beta']",
                "count = len(REGISTRY)\nnames = ', '.join(REGISTRY)",
                "print(f'count={count},names={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=2" in out
        assert "names=alpha, beta" in out

        nb_runner.set_cell_source(1, "REGISTRY = ['alpha', 'beta', 'gamma', 'delta']")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=4" in out

    def test_config_class_edit(self, nb_runner):
        """Editing a config class should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Config:\n"
                    "    def __init__(self):\n"
                    "        self.debug = False\n"
                    "        self.level = 1\n"
                    "cfg = Config()"
                ),
                "mode = 'debug' if cfg.debug else 'prod'\nlvl = cfg.level",
                "info = f'{mode}:L{lvl}'",
                "print(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=prod:L1" in out

        nb_runner.set_cell_source(
            1,
            (
                "class Config:\n"
                "    def __init__(self):\n"
                "        self.debug = True\n"
                "        self.level = 5\n"
                "cfg = Config()"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=debug:L5" in out


@pytest.mark.integration
class TestSingletonPatterns:
    """Test singleton-like patterns across cells."""

    def test_registry_pattern(self, nb_runner):
        """Registry pattern: register handlers across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                _registry = {}
                def register(name):
                    def decorator(fn):
                        _registry[name] = fn
                        return fn
                    return decorator
            """),
                textwrap.dedent("""\
                @register('add')
                def add(a, b):
                    return a + b

                @register('mul')
                def mul(a, b):
                    return a * b
            """),
                textwrap.dedent("""\
                result = _registry['add'](3, 4)
                print(f"add={result} registered={sorted(_registry.keys())}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "add=7" in output
        assert "['add', 'mul']" in output


class TestGlobalNonlocalScope:
    """Scope-related interaction patterns."""

    @pytest.mark.timeout(90)
    def test_global_var_used_in_function(self, nb_runner):
        """Edit global variable, function using it re-evaluates."""
        nb_runner.create_notebook(
            [
                "RATE = 0.1",
                "def calc_tax(amount):\n    return amount * RATE",
                "result = calc_tax(1000)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "RATE = 0.25")
        nb_runner.run_all()
        assert "result = 250.0" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_closure_captured_variable(self, nb_runner):
        """Edit closure factory parameter, downstream reflects."""
        nb_runner.create_notebook(
            [
                "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
                "add5 = make_adder(5)",
                "result = add5(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "add5 = make_adder(100)")
        nb_runner.run_all()
        assert "result = 110" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_module_level_constant_edit(self, nb_runner):
        """Edit module-level constant used in multiple downstream cells."""
        nb_runner.create_notebook(
            [
                "PI = 3.14",
                "area = PI * 5 * 5\nprint(f'area = {area}')",
                "circumference = 2 * PI * 5\nprint(f'circumference = {circumference}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(2)
        assert "circumference = 31.4" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "PI = 3.14159")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        assert "78.539" in out2
        assert "31.415" in out3

    @pytest.mark.timeout(90)
    def test_nested_function_with_nonlocal_pattern(self, nb_runner):
        """Counter pattern with nested function."""
        nb_runner.create_notebook(
            [
                "def make_counter(start):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
                "counter = make_counter(0)\nval1 = counter()\nval2 = counter()\nprint(f'val1 = {val1}, val2 = {val2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val1 = 1, val2 = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "counter = make_counter(10)\nval1 = counter()\nval2 = counter()\nprint(f'val1 = {val1}, val2 = {val2}')"
        )
        nb_runner.run_all()
        assert "val1 = 11, val2 = 12" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_global_in_function(self, nb_runner):
        nb_runner.create_notebook(
            [
                "counter = 0\ndef increment(n):\n    global counter\n    counter += n\n    return counter",
                "r1 = increment(5)\nr2 = increment(3)\nprint(f'r1={r1} r2={r2} counter={counter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5 r2=8 counter=8" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_nonlocal_in_nested(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def outer():\n    total = 0\n    def inner(x):\n        nonlocal total\n        total += x\n        return total\n    return inner",
                "acc = outer()\nresults = [acc(i) for i in [10, 20, 30]]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, 30, 60]" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.timeout(90)
class TestModuleLocalScopeInteraction:
    """Test module vs local scope patterns with cache invalidation."""

    def test_module_constant_edit(self, nb_runner):
        """Editing a module-level constant used inside a function should propagate."""
        nb_runner.create_notebook(
            [
                "MULTIPLIER = 10",
                "def scale(x):\n    return x * MULTIPLIER",
                "result = scale(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=50" in out

        nb_runner.set_cell_source(1, "MULTIPLIER = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=500" in out

    def test_global_dict_edit(self, nb_runner):
        """Editing a global config dict used in functions should propagate."""
        nb_runner.create_notebook(
            [
                "CONFIG = {'tax_rate': 0.1, 'discount': 0.05}",
                "def compute_price(base):\n    tax = base * CONFIG['tax_rate']\n    disc = base * CONFIG['discount']\n    return base + tax - disc",
                "price = compute_price(100)",
                "print(f'price={price}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "price=105.0" in out

        nb_runner.set_cell_source(1, "CONFIG = {'tax_rate': 0.2, 'discount': 0.1}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "price=110.0" in out

    def test_nested_function_scope_edit(self, nb_runner):
        """Editing a closure variable captured from outer scope should propagate."""
        nb_runner.create_notebook(
            [
                "base_offset = 5",
                "def make_adder():\n    offset = base_offset\n    def add(x):\n        return x + offset\n    return add",
                "adder = make_adder()\nresult = adder(10)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(1, "base_offset = 50")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=60" in out


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestVariableShadowingEdits:
    """Editing code with variable shadowing."""

    def test_shadow_with_function_local(self, nb_runner):
        """Edit function that shadows a global variable."""
        nb_runner.create_notebook(
            [
                "x = 'global'  # shadow source",
                "def show_x():\n    x = 'local'\n    return x",
                "result = show_x()\nprint(f'result = {result}')\nprint(f'global_x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result = local" in out
        assert "global_x = global" in out

        # Change function to use global
        nb_runner.set_cell_source(2, "def show_x():\n    return x  # use global")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result = global" in out2

    def test_shadow_in_comprehension(self, nb_runner):
        """Edit code that shadows variables in comprehensions."""
        nb_runner.create_notebook(
            [
                "x = 100  # comprehension shadow source",
                "result = [x for x in range(5)]\nprint(f'result = {result}')\nprint(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result = [0, 1, 2, 3, 4]" in out
        # In Python 3, comprehension has its own scope
        assert "x = 100" in out

        # Change comprehension range
        nb_runner.set_cell_source(2, "result = [x for x in range(3)]\nprint(f'result = {result}')\nprint(f'x = {x}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "result = [0, 1, 2]" in out2
        assert "x = 100" in out2

    def test_edit_global_affects_function(self, nb_runner):
        """Edit a global variable and verify functions see the change."""
        nb_runner.create_notebook(
            [
                "FACTOR = 10  # global factor",
                "def compute(val):\n    return val * FACTOR",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

        # Change global
        nb_runner.set_cell_source(1, "FACTOR = 100  # global factor v2")
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(3)

    def test_nested_function_shadowing(self, nb_runner):
        """Edit nested functions that shadow variables."""
        nb_runner.create_notebook(
            [
                "val = 'outer'  # nested shadow source",
                "def outer():\n    val = 'middle'\n    def inner():\n        val = 'inner'\n        return val\n    return f'{val}-{inner()}'",
                "result = outer()\nprint(f'result = {result}')\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result = middle-inner" in out
        assert "val = outer" in out

        # Edit inner to use nonlocal
        nb_runner.set_cell_source(
            2,
            "def outer():\n    val = 'middle'\n    def inner():\n        nonlocal val\n        val = 'changed'\n        return val\n    return f'{inner()}-{val}'",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result = changed-changed" in out2


@pytest.mark.mutations
@pytest.mark.timeout(90)
class TestSideEffectEdits:
    """Editing cells with file I/O side effects."""

    def test_file_write_edit_content(self, nb_runner, tmp_path):
        """Edit what gets written to a file."""
        out_path = str(tmp_path / "output.txt").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"path = '{out_path}'",
                "with open(path, 'w') as f:\n    f.write('hello')",
                "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = hello" in nb_runner.get_output(3)

        # Edit what we write
        nb_runner.set_cell_source(2, "with open(path, 'w') as f:\n    f.write('world')")
        nb_runner.run_all()
        assert "content = world" in nb_runner.get_output(3)

    def test_file_write_edit_path(self, nb_runner, tmp_path):
        """Edit the output file path."""
        path1 = str(tmp_path / "out1.txt").replace("\\", "/")
        path2 = str(tmp_path / "out2.txt").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"path = '{path1}'",
                "with open(path, 'w') as f:\n    f.write('data1')",
                "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = data1" in nb_runner.get_output(3)

        # Switch path
        nb_runner.set_cell_source(1, f"path = '{path2}'")
        nb_runner.set_cell_source(2, "with open(path, 'w') as f:\n    f.write('data2')")
        nb_runner.set_cell_source(
            3,
            "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
        )
        nb_runner.run_all()
        assert "content = data2" in nb_runner.get_output(3)
