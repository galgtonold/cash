"""Batch 77: Complex data transformations — cash caching with multi-step reshaping."""
import textwrap
import pytest


@pytest.mark.stress
class TestDictTransforms:
    """Test complex dictionary transformation patterns."""

    def test_nested_dict_flatten(self, nb_runner):
        """Flatten nested dict across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def flatten_dict(d, parent_key='', sep='.'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)

                nested = {
                    'a': 1,
                    'b': {'c': 2, 'd': {'e': 3, 'f': 4}},
                    'g': 5,
                }
                flat = flatten_dict(nested)
                print(f"flat={flat}")
            """),
            textwrap.dedent("""\
                keys = sorted(flat.keys())
                print(f"keys={keys}")
                total = sum(flat.values())
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "b.d.e" in out1
        out2 = nb_runner.get_output(2)
        assert "total=15" in out2

    def test_dict_merge_deep(self, nb_runner):
        """Deep merge dictionaries across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def deep_merge(base, override):
                    result = dict(base)
                    for k, v in override.items():
                        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                            result[k] = deep_merge(result[k], v)
                        else:
                            result[k] = v
                    return result

                defaults = {'db': {'host': 'localhost', 'port': 5432}, 'debug': False}
                overrides = {'db': {'port': 3306, 'name': 'mydb'}, 'debug': True}
                config = deep_merge(defaults, overrides)
                print(f"config={config}")
            """),
            textwrap.dedent("""\
                print(f"host={config['db']['host']}")
                print(f"port={config['db']['port']}")
                print(f"debug={config['debug']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "host=localhost" in out2
        assert "port=3306" in out2
        assert "debug=True" in out2


@pytest.mark.stress
class TestListTransforms:
    """Test complex list transformation patterns."""

    def test_group_by(self, nb_runner):
        """Group-by operation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import defaultdict

                records = [
                    ('Engineering', 'Alice', 90000),
                    ('Engineering', 'Bob', 85000),
                    ('Sales', 'Charlie', 70000),
                    ('Sales', 'Diana', 75000),
                    ('Engineering', 'Eve', 95000),
                    ('Marketing', 'Frank', 65000),
                ]

                grouped = defaultdict(list)
                for dept, name, salary in records:
                    grouped[dept].append((name, salary))
            """),
            textwrap.dedent("""\
                for dept in sorted(grouped.keys()):
                    members = grouped[dept]
                    avg_salary = sum(s for _, s in members) / len(members)
                    print(f"{dept}: {len(members)} employees, avg=${avg_salary:,.0f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Engineering: 3 employees" in out
        assert "Sales: 2 employees" in out
        assert "Marketing: 1 employees" in out

    def test_transpose_list_of_dicts(self, nb_runner):
        """Transpose list of dicts to dict of lists across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                rows = [
                    {'name': 'A', 'x': 1, 'y': 10},
                    {'name': 'B', 'x': 2, 'y': 20},
                    {'name': 'C', 'x': 3, 'y': 30},
                ]

                columns = {}
                for row in rows:
                    for k, v in row.items():
                        columns.setdefault(k, []).append(v)
                print(f"cols={sorted(columns.keys())}")
            """),
            textwrap.dedent("""\
                print(f"names={columns['name']}")
                print(f"x_sum={sum(columns['x'])}")
                print(f"y_sum={sum(columns['y'])}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "names=['A', 'B', 'C']" in out2
        assert "x_sum=6" in out2
        assert "y_sum=60" in out2

    def test_transform_propagation(self, nb_runner):
        """Transform propagates when input changes."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                raw = [1, 2, 3, 4, 5]
                step1 = [x * 2 for x in raw]
            """),
            textwrap.dedent("""\
                step2 = [x + 10 for x in step1]
                print(f"result={step2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[12, 14, 16, 18, 20]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            raw = [10, 20, 30]
            step1 = [x * 2 for x in raw]
        """))
        nb_runner.run_cells([1, 2])
        assert "result=[30, 50, 70]" in nb_runner.get_output(2)
