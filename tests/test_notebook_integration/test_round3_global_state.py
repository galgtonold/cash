"""
Global state, singleton, registry, and mutable default argument patterns.
Tests tricky Python patterns that interact with caching in subtle ways.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


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


class TestMutableDefaultArguments:
    """Test caching with mutable default arguments."""

    def test_mutable_default_list(self, nb_runner):
        """Classic mutable default argument gotcha."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def append_to(item, lst=None):
                    if lst is None:
                        lst = []
                    lst.append(item)
                    return lst
            """),
                textwrap.dedent("""\
                r1 = append_to(1)
                r2 = append_to(2)
                r3 = append_to(3, [10, 20])
                print(f"r1={r1} r2={r2} r3={r3}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "r1=[1]" in output
        assert "r2=[2]" in output
        assert "r3=[10, 20, 3]" in output
