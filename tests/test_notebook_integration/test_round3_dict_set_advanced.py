"""Batch 93 – complex dict and set operations."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDictOperations:
    """Advanced dictionary patterns."""

    def test_dict_merge_operators(self, nb_runner):
        """Dict merge (|) and update (|=) operators (Python 3.9+)."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                defaults = {'a': 1, 'b': 2, 'c': 3}
                overrides = {'b': 20, 'd': 40}
                merged = defaults | overrides
                copy = defaults.copy()
                copy |= {'c': 30, 'e': 50}
            """),
            "print(f'merged={merged}')\nprint(f'copy={copy}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'b': 20" in out  # overrides win
        assert "'d': 40" in out
        assert "'c': 30" in out

    def test_dict_comprehension_filtering(self, nb_runner):
        """Dict comprehension with complex filtering."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                inventory = {
                    'apple': {'price': 1.5, 'stock': 100, 'category': 'fruit'},
                    'bread': {'price': 2.5, 'stock': 50, 'category': 'bakery'},
                    'milk': {'price': 3.0, 'stock': 0, 'category': 'dairy'},
                    'cheese': {'price': 5.0, 'stock': 30, 'category': 'dairy'},
                    'cake': {'price': 8.0, 'stock': 10, 'category': 'bakery'},
                }
                in_stock = {k: v for k, v in inventory.items() if v['stock'] > 0}
                by_category = {}
                for k, v in in_stock.items():
                    by_category.setdefault(v['category'], []).append(k)
                total_value = sum(v['price'] * v['stock'] for v in in_stock.values())
            """),
            "print(f'in_stock={len(in_stock)}')\nprint(f'categories={by_category}')\nprint(f'total_value={total_value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "in_stock=4" in out  # milk is out of stock
        assert "fruit" in out
        assert "total_value=" in out

    def test_nested_dict_update(self, nb_runner):
        """Deep merge of nested dicts."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'host': 'localhost'" in out  # preserved from base
        assert "'port': 3306" in out         # overridden
        assert "'debug': True" in out
        assert "'cache': True" in out


class TestSetOperations:
    """Advanced set operations."""

    def test_set_algebra(self, nb_runner):
        """Set union, intersection, difference, symmetric difference."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                python_devs = {'Alice', 'Bob', 'Charlie', 'Diana'}
                js_devs = {'Bob', 'Diana', 'Eve', 'Frank'}
                go_devs = {'Charlie', 'Eve', 'Grace'}

                all_devs = python_devs | js_devs | go_devs
                multi_lang = (python_devs & js_devs) | (python_devs & go_devs) | (js_devs & go_devs)
                python_only = python_devs - js_devs - go_devs
            """),
            "print(f'all={sorted(all_devs)}')\nprint(f'multi={sorted(multi_lang)}')\nprint(f'py_only={sorted(python_only)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "Bob" in out
        assert "Eve" in out

    def test_frozenset_dict_key(self, nb_runner):
        """Frozenset as dictionary keys."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                edges = [
                    frozenset({'A', 'B'}),
                    frozenset({'B', 'C'}),
                    frozenset({'A', 'C'}),
                    frozenset({'A', 'B'}),  # duplicate
                ]
                weights = {}
                for i, edge in enumerate(edges):
                    weights[edge] = i + 1  # last wins for duplicates
                unique_edges = len(weights)
            """),
            "print(f'unique={unique_edges} weights={weights}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "unique=3" in out
