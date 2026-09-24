"""Closures and nonlocal state across cells."""

import textwrap

import pytest


# Closure and scope interaction tests.
#
# Tests editing closures, nonlocal variables, nested function scopes,
# and variable capture patterns.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestClosureEdits:
    """Editing closure definitions and captured variables."""

    def test_edit_closure_body(self, nb_runner):
        """Edit the body of a closure."""
        nb_runner.create_notebook(
            [
                "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
                "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change closure to multiply
        nb_runner.set_cell_source(
            1,
            "def make_adder(n):\n    def adder(x):\n        return x * n\n    return adder",
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)

    def test_edit_captured_value(self, nb_runner):
        """Edit the value passed to create a closure."""
        nb_runner.create_notebook(
            [
                "def make_greeter(name):\n    def greet():\n        return f'Hello, {name}!'\n    return greet",
                "g = make_greeter('Alice')\nresult = g()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Hello, Alice!" in nb_runner.get_output(2)

        # Change captured value
        nb_runner.set_cell_source(
            2,
            "g = make_greeter('Bob')\nresult = g()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = Hello, Bob!" in nb_runner.get_output(2)

    def test_nested_closure_edit(self, nb_runner):
        """Two levels of closure nesting, edit inner."""
        nb_runner.create_notebook(
            [
                "def outer(a):\n    def middle(b):\n        def inner(c):\n            return a + b + c\n        return inner\n    return middle",
                "fn = outer(1)(2)\nresult = fn(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        # Change inner to multiply
        nb_runner.set_cell_source(
            1,
            "def outer(a):\n    def middle(b):\n        def inner(c):\n            return a * b * c\n        return inner\n    return middle",
        )
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)  # 1*2*3=6, same value!

        # Use different values to see the difference
        nb_runner.set_cell_source(2, "fn = outer(2)(3)\nresult = fn(4)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)


# Closure factory and scope capture interaction tests.
#
# Tests editing cells with closure factories, captured variables,
# and function composition patterns.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestClosureFactoryEdits:
    """Editing closure and function factory patterns."""

    def test_edit_closure_captured_argument(self, nb_runner):
        """Edit the argument passed to a closure factory."""
        nb_runner.create_notebook(
            [
                "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
                "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change to add 20
        nb_runner.set_cell_source(2, "add5 = make_adder(20)\nresult = add5(10)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

    def test_edit_lambda_factory(self, nb_runner):
        """Edit a function that returns a lambda."""
        nb_runner.create_notebook(
            [
                "def multiplier(factor):\n    return lambda x: x * factor",
                "double = multiplier(2)\nval = double(7)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 14" in nb_runner.get_output(2)

        # Edit factory to add offset
        nb_runner.set_cell_source(1, "def multiplier(factor):\n    return lambda x: x * factor + 1")
        nb_runner.run_all()
        assert "val = 15" in nb_runner.get_output(2)

    def test_edit_counter_start_value(self, nb_runner):
        """Edit counter closure's starting value."""
        nb_runner.create_notebook(
            [
                "def make_counter(start=0):\n    count = [start]\n    def inc():\n        count[0] += 1\n        return count[0]\n    return inc",
                "counter = make_counter(0)\na = counter()\nb = counter()\nprint(f'a={a} b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=2" in nb_runner.get_output(2)

        # Change start value
        nb_runner.set_cell_source(2, "counter = make_counter(10)\na = counter()\nb = counter()\nprint(f'a={a} b={b}')")
        nb_runner.run_all()
        assert "a=11 b=12" in nb_runner.get_output(2)

    def test_edit_function_composition(self, nb_runner):
        """Edit composed functions."""
        nb_runner.create_notebook(
            [
                "def compose(f, g):\n    return lambda x: f(g(x))",
                "inc = lambda x: x + 1\ndbl = lambda x: x * 2\ninc_then_dbl = compose(dbl, inc)\nresult = inc_then_dbl(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 8" in nb_runner.get_output(2)

        # Change composition order
        nb_runner.set_cell_source(
            2,
            "inc = lambda x: x + 1\ndbl = lambda x: x * 2\ndbl_then_inc = compose(inc, dbl)\nresult = dbl_then_inc(3)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)


# Closure & scope edge cases — closures, nonlocal, late binding.
@pytest.mark.stress
class TestClosureBasics:
    """Test closure patterns across cells."""

    def test_closure_factory_change(self, nb_runner):
        """Changing closure factory propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def make_greeter(greeting):
                    def greet(name):
                        return f"{greeting}, {name}!"
                    return greet

                hello = make_greeter("Hello")
            """),
                textwrap.dedent("""\
                msg = hello("World")
                print(f"msg={msg}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)

        # Change factory
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def make_greeter(greeting):
                def greet(name):
                    return f"{greeting} to {name}!!"
                return greet

            hello = make_greeter("Welcome")
        """),
        )
        nb_runner.run_all()
        assert "msg=Welcome to World!!" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClosureScopeCapture:
    """closure scope and nonlocal variable capture."""

    def test_closure_nonlocal(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def make_counter():\n    count = 0\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
                "counter = make_counter()\nr1 = counter()\nr2 = counter()\nr3 = counter()\nprint(f'r1={r1} r2={r2} r3={r3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=1 r2=2 r3=3" in nb_runner.get_output(2)

    def test_closure_captures(self, nb_runner):
        nb_runner.create_notebook(
            [
                "multiplier = 10",
                "def scale(x):\n    return x * multiplier\nresult = scale(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(2)

    def test_closure_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def make_adder(n):\n    def add(x):\n        return x + n\n    return add",
                "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "add100 = make_adder(100)\nresult = add100(10)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(2)


# Closure and nonlocal interaction tests.
# Tests that editing closures with nonlocal variables properly
# invalidates downstream computations.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClosureNonlocalInteraction:
    """Test closure/nonlocal patterns with cache invalidation."""

    def test_closure_capture_edit(self, nb_runner):
        """Editing closure-captured variable should propagate."""
        nb_runner.create_notebook(
            [
                "factor = 3",
                "def make_multiplier():\n    f = factor\n    def mul(x):\n        return x * f\n    return mul",
                "m = make_multiplier()\nresult = m(7)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=21" in out

        nb_runner.set_cell_source(1, "factor = 10")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=70" in out

    def test_counter_closure_edit(self, nb_runner):
        """Editing a counter closure's starting value should propagate."""
        nb_runner.create_notebook(
            [
                "start = 0",
                (
                    "def make_counter():\n"
                    "    count = start\n"
                    "    def increment():\n"
                    "        nonlocal count\n"
                    "        count += 1\n"
                    "        return count\n"
                    "    return increment"
                ),
                "counter = make_counter()\nvals = [counter() for _ in range(3)]",
                "print(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "vals=[1, 2, 3]" in out

        nb_runner.set_cell_source(1, "start = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "vals=[101, 102, 103]" in out

    def test_accumulator_closure_edit(self, nb_runner):
        """Editing accumulator initial value should propagate."""
        nb_runner.create_notebook(
            [
                "initial = 10",
                (
                    "def make_accumulator():\n"
                    "    total = initial\n"
                    "    def add(n):\n"
                    "        nonlocal total\n"
                    "        total += n\n"
                    "        return total\n"
                    "    return add"
                ),
                "acc = make_accumulator()\nresults = [acc(1), acc(2), acc(3)]",
                "print(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "results=[11, 13, 16]" in out

        nb_runner.set_cell_source(1, "initial = 0")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "results=[1, 3, 6]" in out


# advanced closure, nonlocal, and scope edge cases.
@pytest.mark.stress
@pytest.mark.integration
class TestAdvancedClosures:
    """Advanced closure and nonlocal patterns."""

    def test_closure_memoize(self, nb_runner):
        """Closure-based memoization."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def memoize(fn):
                    cache = {}
                    call_count = [0]
                    def wrapper(*args):
                        call_count[0] += 1
                        if args not in cache:
                            cache[args] = fn(*args)
                        return cache[args]
                    wrapper.calls = call_count
                    wrapper.cache = cache
                    return wrapper

                @memoize
                def fib(n):
                    if n < 2: return n
                    return fib(n - 1) + fib(n - 2)

                result = fib(20)
                cache_size = len(fib.cache)
            """),
                "print(f'fib20={result} cache_size={cache_size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "fib20=6765" in out
        assert "cache_size=21" in out

    def test_closure_chain(self, nb_runner):
        """Chain of closures composing functions."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def compose(*funcs):
                    def composed(x):
                        result = x
                        for f in reversed(funcs):
                            result = f(result)
                        return result
                    return composed

                add1 = lambda x: x + 1
                double = lambda x: x * 2
                square = lambda x: x ** 2

                pipe = compose(square, double, add1)
                results = [pipe(i) for i in range(5)]
            """),
                "print(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # pipe(0) = square(double(add1(0))) = square(double(1)) = square(2) = 4
        # pipe(1) = square(double(2)) = square(4) = 16
        assert "4" in out
        assert "16" in out

    def test_closure_propagation_memoize(self, nb_runner):
        """Memoized closure with upstream change propagation."""
        nb_runner.create_notebook(
            [
                "scale = 2",
                textwrap.dedent("""\
                def make_scaler(s):
                    cache = {}
                    def scaled(x):
                        if x not in cache:
                            cache[x] = x * s
                        return cache[x]
                    return scaled

                scaler = make_scaler(scale)
                vals = [scaler(i) for i in range(5)]
            """),
                "print(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 2, 4, 6, 8]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 5")
        nb_runner.run_cells([1, 2, 3])
        assert "[0, 5, 10, 15, 20]" in nb_runner.get_output(3)


@pytest.mark.stress
class TestLatebinding:
    """Test late binding in closures — a classic Python gotcha."""

    def test_late_binding_gotcha(self, nb_runner):
        """Late binding in loop closures."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                # Classic late-binding gotcha: all funcs see final i
                funcs_late = []
                for i in range(5):
                    funcs_late.append(lambda: i)

                # Fix with default argument
                funcs_early = []
                for i in range(5):
                    funcs_early.append(lambda i=i: i)
            """),
                textwrap.dedent("""\
                late_results = [f() for f in funcs_late]
                early_results = [f() for f in funcs_early]
                print(f"late={late_results}")
                print(f"early={early_results}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "late=[4, 4, 4, 4, 4]" in out
        assert "early=[0, 1, 2, 3, 4]" in out


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNonlocalEdits:
    """Nonlocal variable patterns."""

    def test_nonlocal_counter(self, nb_runner):
        """Edit a nonlocal counter closure."""
        nb_runner.create_notebook(
            [
                "def make_counter(start=0):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
                "c = make_counter()\nvals = [c() for _ in range(3)]\nprint(f'vals = {vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 2, 3]" in nb_runner.get_output(2)

        # Change start value
        nb_runner.set_cell_source(
            2,
            "c = make_counter(10)\nvals = [c() for _ in range(3)]\nprint(f'vals = {vals}')",
        )
        nb_runner.run_all()
        assert "vals = [11, 12, 13]" in nb_runner.get_output(2)

    def test_nonlocal_accumulator_edit(self, nb_runner):
        """Edit the accumulation logic in a nonlocal pattern."""
        nb_runner.create_notebook(
            [
                "def make_acc():\n    total = 0\n    def add(x):\n        nonlocal total\n        total += x\n        return total\n    return add",
                "acc = make_acc()\nresults = [acc(i) for i in [10, 20, 30]]\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [10, 30, 60]" in nb_runner.get_output(2)

        # Change to multiply accumulator
        nb_runner.set_cell_source(
            1,
            "def make_acc():\n    total = 1\n    def add(x):\n        nonlocal total\n        total *= x\n        return total\n    return add",
        )
        nb_runner.run_all()
        assert "results = [10, 200, 6000]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestScopeInteraction:
    """Variable scope interactions between cells."""

    def test_same_name_different_scope(self, nb_runner):
        """Same variable name in different scopes."""
        nb_runner.create_notebook(
            [
                "x = 'global_x'  # global scope",
                "def show_x():\n    x = 'local_x'\n    return x",
                "local_val = show_x()\nprint(f'global={x} local={local_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "global=global_x" in nb_runner.get_output(3)
        assert "local=local_x" in nb_runner.get_output(3)

        # Edit global
        nb_runner.set_cell_source(1, "x = 'new_global'  # global scope changed")
        nb_runner.run_all()
        assert "global=new_global" in nb_runner.get_output(3)
        assert "local=local_x" in nb_runner.get_output(3)


@pytest.mark.stress
class TestNestedScopes:
    """Test deeply nested scope patterns."""

    def test_closure_with_class(self, nb_runner):
        """Closure inside a class method."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class EventBus:
                    def __init__(self):
                        self.handlers = {}

                    def on(self, event):
                        def decorator(fn):
                            self.handlers.setdefault(event, []).append(fn)
                            return fn
                        return decorator

                    def emit(self, event, *args):
                        results = []
                        for handler in self.handlers.get(event, []):
                            results.append(handler(*args))
                        return results

                bus = EventBus()

                @bus.on('click')
                def handle_click(x):
                    return f"clicked:{x}"

                @bus.on('click')
                def handle_click2(x):
                    return f"also:{x}"
            """),
                textwrap.dedent("""\
                results = bus.emit('click', 'button')
                print(f"results={results}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['clicked:button', 'also:button']" in nb_runner.get_output(2)
