"""
Interaction test: functools.wraps and update_wrapper for decorator metadata.
Tests preserving __name__, __doc__, __module__ across decorated functions,
and cross-cell decorator composition with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsWrapsUpdate:
    """Test functools.wraps decorator metadata across cells."""

    def test_wraps_metadata(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create decorated function
            "import functools\ndef my_decorator(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n\n@my_decorator\ndef greet(name):\n    '''Say hello'''\n    return f'Hello, {name}!'\n\nprint(f'name={greet.__name__}')\nprint(f'doc={greet.__doc__}')",
            # Cell 2: use the decorated function
            "result = greet('Alice')\nprint(f'result={result}')",
            # Cell 3: check wrapped attribute
            "has_wrapped = hasattr(greet, '__wrapped__')\nprint(f'has_wrapped={has_wrapped}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "name=greet" in out1
        assert "doc=Say hello" in out1
        out2 = nb_runner.get_output(2)
        assert "result=Hello, Alice!" in out2
        out3 = nb_runner.get_output(3)
        assert "has_wrapped=True" in out3

    def test_wraps_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import functools\ndef logged(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@logged\ndef add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(f'result={result}')",
            "doubled = result * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=7" in nb_runner.get_output(1)
        assert "doubled=14" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(1, "import functools\ndef logged(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@logged\ndef add(a, b):\n    return a + b + 10\nresult = add(3, 4)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "result=17" in nb_runner.get_output(1)
        assert "doubled=34" in nb_runner.get_output(2)

    def test_wraps_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import functools\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@timer\ndef square(n):\n    return n * n\nval = square(5)\nprint(f'val={val}')",
            "name_ok = square.__name__ == 'square'\nprint(f'name_ok={name_ok}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=25" in nb_runner.get_output(1)
        assert "name_ok=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "name_ok=True" in nb_runner.get_output(2)
