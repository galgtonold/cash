"""Complex data transformations — cash caching with multi-step reshaping."""

import textwrap

import pytest


@pytest.mark.stress
class TestListTransforms:
    """Test complex list transformation patterns."""

    def test_group_by(self, nb_runner):
        """Group-by operation across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Engineering: 3 employees" in out
        assert "Sales: 2 employees" in out
        assert "Marketing: 1 employees" in out

    def test_transform_propagation(self, nb_runner):
        """Transform propagates when input changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                raw = [1, 2, 3, 4, 5]
                step1 = [x * 2 for x in raw]
            """),
                textwrap.dedent("""\
                step2 = [x + 10 for x in step1]
                print(f"result={step2}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[12, 14, 16, 18, 20]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            raw = [10, 20, 30]
            step1 = [x * 2 for x in raw]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "result=[30, 50, 70]" in nb_runner.get_output(2)
