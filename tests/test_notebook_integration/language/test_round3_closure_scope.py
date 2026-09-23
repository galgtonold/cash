"""Closure & scope edge cases — closures, nonlocal, late binding."""

import textwrap

import pytest


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
