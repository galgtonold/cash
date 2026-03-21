"""Batch 87 – complex multi-cell data pipelines with many dependencies."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMultiCellPipeline:
    """Complex multi-cell workflows testing dependency chains."""

    def test_five_cell_transform_chain(self, nb_runner):
        """5-cell pipeline: generate → filter → transform → aggregate → report."""
        nb_runner.create_notebook([
            # Cell 1: Generate data
            textwrap.dedent("""\
                import random
                random.seed(42)
                raw_data = [{'name': f'item_{i}', 'value': random.randint(1, 100), 'category': random.choice(['A', 'B', 'C'])} for i in range(20)]
            """),
            # Cell 2: Filter
            textwrap.dedent("""\
                filtered = [d for d in raw_data if d['value'] > 30]
                filter_count = len(filtered)
            """),
            # Cell 3: Transform
            textwrap.dedent("""\
                transformed = []
                for d in filtered:
                    transformed.append({
                        'name': d['name'].upper(),
                        'value': d['value'] * 1.1,
                        'category': d['category'],
                        'tier': 'premium' if d['value'] > 70 else 'standard',
                    })
            """),
            # Cell 4: Aggregate
            textwrap.dedent("""\
                from collections import defaultdict
                by_cat = defaultdict(list)
                for t in transformed:
                    by_cat[t['category']].append(t['value'])
                agg = {cat: round(sum(vals)/len(vals), 2) for cat, vals in by_cat.items()}
            """),
            # Cell 5: Report
            textwrap.dedent("""\
                premium_count = sum(1 for t in transformed if t['tier'] == 'premium')
                report = f"filtered={filter_count} premium={premium_count} categories={len(agg)}"
            """),
            "print(report)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(6)
        assert "filtered=" in out
        assert "premium=" in out
        assert "categories=" in out

    def test_branching_pipeline(self, nb_runner):
        """Two branches from same source, merged at the end."""
        nb_runner.create_notebook([
            # Cell 1: Source
            "data = list(range(1, 21))",
            # Cell 2: Branch A – evens
            "evens = [x for x in data if x % 2 == 0]",
            # Cell 3: Branch B – odds
            "odds = [x for x in data if x % 2 != 0]",
            # Cell 4: Merge
            textwrap.dedent("""\
                even_sum = sum(evens)
                odd_sum = sum(odds)
                ratio = round(even_sum / odd_sum, 3)
            """),
            "print(f'even_sum={even_sum} odd_sum={odd_sum} ratio={ratio}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "even_sum=110" in out
        assert "odd_sum=100" in out

    def test_pipeline_mid_change(self, nb_runner):
        """Change a middle cell in a 4-cell pipeline, verify downstream updates."""
        nb_runner.create_notebook([
            "numbers = [1, 2, 3, 4, 5]",
            "doubled = [x * 2 for x in numbers]",
            "total = sum(doubled)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=30" in nb_runner.get_output(4)

        # Change the middle transformation
        nb_runner.set_cell_source(2, "doubled = [x ** 2 for x in numbers]")
        nb_runner.run_cells([2, 3, 4])
        assert "total=55" in nb_runner.get_output(4)  # 1+4+9+16+25

    def test_diamond_dependency(self, nb_runner):
        """Diamond data dependency: A → B,C → D."""
        nb_runner.create_notebook([
            # A: source
            "base = 100",
            # B: depends on A
            "tax = base * 0.08",
            # C: depends on A
            "discount = base * 0.10",
            # D: depends on B and C
            "final = base + tax - discount",
            "print(f'final={final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 100 + 8 - 10 = 98
        assert "final=98.0" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "base = 200")
        nb_runner.run_cells([1, 2, 3, 4, 5])
        # 200 + 16 - 20 = 196
        assert "final=196.0" in nb_runner.get_output(5)

    def test_accumulating_cells(self, nb_runner):
        """Each cell adds to a shared list, testing cumulative state."""
        nb_runner.create_notebook([
            "log = ['start']",
            "log = log + ['step1']",
            "log = log + ['step2']",
            "log = log + ['step3']",
            "print(f'log={log}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "start" in out
        assert "step1" in out
        assert "step2" in out
        assert "step3" in out
