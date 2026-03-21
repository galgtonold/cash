"""Batch 98 – multiple assignment, unpacking, and star expressions."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestUnpacking:
    """Unpacking and multiple assignment patterns."""

    def test_star_unpacking(self, nb_runner):
        """Star (*) unpacking in assignments."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = [1, 2, 3, 4, 5, 6, 7]
                first, *middle, last = data
                a, b, *rest = data
                *init, x, y = data
            """),
            "print(f'first={first} middle={middle} last={last}')\n"
            "print(f'a={a} b={b} rest={rest}')\n"
            "print(f'init={init} x={x} y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first=1" in out
        assert "middle=[2, 3, 4, 5, 6]" in out
        assert "last=7" in out
        assert "rest=[3, 4, 5, 6, 7]" in out
        assert "x=6" in out
        assert "y=7" in out

    def test_nested_unpacking(self, nb_runner):
        """Nested tuple/list unpacking."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                records = [
                    ('Alice', (95, 87, 92)),
                    ('Bob', (78, 82, 90)),
                    ('Charlie', (88, 91, 85)),
                ]
                summaries = []
                for name, (s1, s2, s3) in records:
                    avg = round((s1 + s2 + s3) / 3, 1)
                    summaries.append((name, avg))
            """),
            "print(f'summaries={summaries}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "91.3" in out

    def test_swap_and_multi_assign(self, nb_runner):
        """Swap and multiple assignment in one line."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                a, b = 10, 20
                a, b = b, a
                x = y = z = 42
                p, q = divmod(100, 7)
            """),
            "print(f'a={a} b={b} x={x} y={y} z={z} p={p} q={q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a=20" in out
        assert "b=10" in out
        assert "x=42" in out
        assert "p=14" in out
        assert "q=2" in out

    def test_dict_unpacking(self, nb_runner):
        """Dict unpacking with ** operator."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                defaults = {'color': 'red', 'size': 10}
                overrides = {'size': 20, 'weight': 5}
                merged = {**defaults, **overrides, 'label': 'item'}
            """),
            "print(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'color': 'red'" in out
        assert "'size': 20" in out
        assert "'weight': 5" in out
        assert "'label': 'item'" in out

    def test_unpacking_propagation(self, nb_runner):
        """Unpacking with upstream data change."""
        nb_runner.create_notebook([
            "data = (10, 20, 30)",
            textwrap.dedent("""\
                a, b, c = data
                total = a + b + c
            """),
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = (100, 200, 300)")
        nb_runner.run_cells([1, 2, 3])
        assert "total=600" in nb_runner.get_output(3)
