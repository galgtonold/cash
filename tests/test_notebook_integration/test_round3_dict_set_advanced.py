"""complex dict and set operations."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDictOperations:
    """Advanced dictionary patterns."""

    def test_nested_dict_update(self, nb_runner):
        """Deep merge of nested dicts."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def deep_merge(base, override):
                    result = base.copy()
                    for k, v in override.items():
                        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                            result[k] = deep_merge(result[k], v)
                        else:
                            result[k] = v
                    return result

                base = {'db': {'host': 'localhost', 'port': 5432}, 'debug': False}
                override = {'db': {'port': 3306}, 'debug': True, 'cache': True}
                merged = deep_merge(base, override)
            """),
                "print(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'host': 'localhost'" in out  # preserved from base
        assert "'port': 3306" in out  # overridden
        assert "'debug': True" in out
        assert "'cache': True" in out
