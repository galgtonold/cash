"""Batch 49: Closure & scope edge cases — closures, nonlocal, late binding."""
import textwrap
import pytest


@pytest.mark.stress
class TestClosureBasics:
    """Test closure patterns across cells."""

    def test_closure_captures_value(self, nb_runner):
        """Closure capturing variable from outer scope."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_multiplier(factor):
                    def multiply(x):
                        return x * factor
                    return multiply

                double = make_multiplier(2)
                triple = make_multiplier(3)
            """),
            textwrap.dedent("""\
                r1 = double(10)
                r2 = triple(10)
                print(f"r1={r1} r2={r2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=20 r2=30" in nb_runner.get_output(2)

    def test_closure_over_mutable_state(self, nb_runner):
        """Closure with nonlocal modifying mutable state."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_counter(start=0):
                    count = start
                    def increment():
                        nonlocal count
                        count += 1
                        return count
                    def get():
                        return count
                    return increment, get

                inc, get = make_counter(10)
            """),
            textwrap.dedent("""\
                inc()
                inc()
                inc()
                val = get()
                print(f"val={val}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=13" in nb_runner.get_output(2)

    def test_closure_factory_change(self, nb_runner):
        """Changing closure factory propagates."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)

        # Change factory
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def make_greeter(greeting):
                def greet(name):
                    return f"{greeting} to {name}!!"
                return greet

            hello = make_greeter("Welcome")
        """))
        nb_runner.run_all()
        assert "msg=Welcome to World!!" in nb_runner.get_output(2)


@pytest.mark.stress
class TestLatebinding:
    """Test late binding in closures — a classic Python gotcha."""

    def test_late_binding_gotcha(self, nb_runner):
        """Late binding in loop closures."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "late=[4, 4, 4, 4, 4]" in out
        assert "early=[0, 1, 2, 3, 4]" in out

    def test_functools_partial_vs_closure(self, nb_runner):
        """Comparing partial vs closure across cells."""
        nb_runner.create_notebook([
            "from functools import partial",
            textwrap.dedent("""\
                def power(base, exp):
                    return base ** exp

                square_closure = lambda x: power(x, 2)
                square_partial = partial(power, exp=2)
                cube_partial = partial(power, exp=3)
            """),
            textwrap.dedent("""\
                r1 = square_closure(5)
                r2 = square_partial(5)
                r3 = cube_partial(5)
                print(f"closure={r1} partial_sq={r2} partial_cu={r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "closure=25 partial_sq=25 partial_cu=125" in nb_runner.get_output(3)


@pytest.mark.stress
class TestNestedScopes:
    """Test deeply nested scope patterns."""

    def test_triple_nested_closure(self, nb_runner):
        """Three levels of closure nesting."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def outer(a):
                    def middle(b):
                        def inner(c):
                            return a + b + c
                        return inner
                    return middle

                fn = outer(100)(20)
            """),
            textwrap.dedent("""\
                result = fn(3)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=123" in nb_runner.get_output(2)

    def test_closure_with_class(self, nb_runner):
        """Closure inside a class method."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['clicked:button', 'also:button']" in nb_runner.get_output(2)

    def test_nonlocal_accumulator(self, nb_runner):
        """Nonlocal variable accumulation across calls."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_accumulator():
                    total = 0
                    items = []
                    def add(value):
                        nonlocal total
                        total += value
                        items.append(value)
                        return total
                    def summary():
                        return {'total': total, 'count': len(items), 'items': items[:]}
                    return add, summary

                add, summary = make_accumulator()
            """),
            textwrap.dedent("""\
                add(10)
                add(20)
                add(30)
                s = summary()
                print(f"total={s['total']} count={s['count']} items={s['items']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60 count=3 items=[10, 20, 30]" in nb_runner.get_output(2)
