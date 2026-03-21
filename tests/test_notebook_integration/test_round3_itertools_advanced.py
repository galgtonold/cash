"""Batch 79: Itertools advanced — cash caching with itertools combinatorial patterns."""
import textwrap
import pytest


@pytest.mark.stress
class TestItertoolsCombinatorial:
    """Test itertools combinatorial patterns across cells."""

    def test_product(self, nb_runner):
        """itertools.product across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import product

                colors = ['red', 'blue']
                sizes = ['S', 'M', 'L']
                combos = list(product(colors, sizes))
                print(f"count={len(combos)}")
            """),
            textwrap.dedent("""\
                labels = [f"{c}-{s}" for c, s in combos]
                print(f"labels={labels}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "red-S" in out2
        assert "blue-L" in out2

    def test_combinations_permutations(self, nb_runner):
        """Combinations and permutations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import combinations, permutations

                items = [1, 2, 3, 4]
                combos = list(combinations(items, 2))
                perms = list(permutations(items, 2))
                print(f"C(4,2)={len(combos)} P(4,2)={len(perms)}")
            """),
            textwrap.dedent("""\
                combo_sums = [a + b for a, b in combos]
                print(f"combo_sums={combo_sums}")
                max_sum = max(combo_sums)
                print(f"max_combo_sum={max_sum}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "C(4,2)=6 P(4,2)=12" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "max_combo_sum=7" in out2

    def test_groupby(self, nb_runner):
        """itertools.groupby across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import groupby

                data = sorted([
                    ('A', 1), ('B', 2), ('A', 3), ('B', 4), ('C', 5)
                ], key=lambda x: x[0])

                grouped = {k: [v for _, v in g] for k, g in groupby(data, key=lambda x: x[0])}
                print(f"groups={grouped}")
            """),
            textwrap.dedent("""\
                sums = {k: sum(v) for k, v in grouped.items()}
                print(f"sums={sums}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "A" in out1
        assert "B" in out1
        out2 = nb_runner.get_output(2)
        assert "sums=" in out2


@pytest.mark.stress
class TestItertoolsInfinite:
    """Test infinite iterator patterns."""

    def test_islice_count(self, nb_runner):
        """islice with count across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import islice, count

                # First 10 squares from counter
                squares = list(islice((x**2 for x in count(1)), 10))
                print(f"squares={squares}")
            """),
            textwrap.dedent("""\
                total = sum(squares)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squares=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]" in nb_runner.get_output(1)
        assert "total=385" in nb_runner.get_output(2)

    def test_chain_accumulate(self, nb_runner):
        """chain and accumulate across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import chain, accumulate

                a = [1, 2, 3]
                b = [4, 5, 6]
                c = [7, 8, 9]
                combined = list(chain(a, b, c))
                running_sum = list(accumulate(combined))
                print(f"combined={combined}")
            """),
            textwrap.dedent("""\
                print(f"running_sum={running_sum}")
                print(f"final={running_sum[-1]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "final=45" in out2

    def test_itertools_propagation(self, nb_runner):
        """Itertools results propagate on change."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from itertools import combinations
                items = ['A', 'B', 'C']
                pairs = list(combinations(items, 2))
            """),
            textwrap.dedent("""\
                labels = [f"{a}-{b}" for a, b in pairs]
                print(f"labels={labels}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "labels=['A-B', 'A-C', 'B-C']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from itertools import combinations
            items = ['X', 'Y', 'Z', 'W']
            pairs = list(combinations(items, 2))
        """))
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "X-Y" in out
        assert "Z-W" in out
