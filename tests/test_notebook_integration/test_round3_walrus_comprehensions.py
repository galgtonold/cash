"""Batch 64: Walrus operator (:=) & complex comprehensions — cash caching."""
import textwrap
import pytest


@pytest.mark.stress
class TestWalrusOperator:
    """Test walrus operator patterns across cells."""

    def test_walrus_in_while(self, nb_runner):
        """Walrus operator in while loop."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = [1, 2, 3, 4, 5, 0, 6, 7]
                results = []
                idx = 0
                while (val := data[idx]) != 0:
                    results.append(val ** 2)
                    idx += 1
                print(f"results={results}")
            """),
            textwrap.dedent("""\
                total = sum(results)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 4, 9, 16, 25]" in nb_runner.get_output(1)
        assert "total=55" in nb_runner.get_output(2)

    def test_walrus_in_comprehension(self, nb_runner):
        """Walrus operator in list comprehension with filter."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                raw_data = ['  hello  ', '', '  world  ', '   ', 'python']
                cleaned = [stripped for s in raw_data if (stripped := s.strip())]
                print(f"cleaned={cleaned}")
            """),
            textwrap.dedent("""\
                lengths = [len(w) for w in cleaned]
                print(f"lengths={lengths}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=['hello', 'world', 'python']" in nb_runner.get_output(1)
        assert "lengths=[5, 5, 6]" in nb_runner.get_output(2)

    def test_walrus_propagation(self, nb_runner):
        """Walrus operator result propagation on change."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                numbers = [2, 8, 3, 15, 7, 20, 1]
                big = [y for x in numbers if (y := x * 2) > 10]
                print(f"big={big}")
            """),
            textwrap.dedent("""\
                count = len(big)
                print(f"count={count}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big=[16, 30, 14, 40]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)

        # Change threshold by changing input
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            numbers = [2, 8, 3, 15, 7, 20, 1]
            big = [y for x in numbers if (y := x * 3) > 10]
            print(f"big={big}")
        """))
        nb_runner.run_cells([1, 2])
        assert "big=[24, 45, 21, 60]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)


@pytest.mark.stress
class TestComplexComprehensions:
    """Test complex comprehension patterns."""

    def test_nested_dict_comprehension(self, nb_runner):
        """Nested dictionary comprehension."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                categories = {'fruit': ['apple', 'banana'], 'veg': ['carrot', 'pea']}
                indexed = {cat: {i: item for i, item in enumerate(items)}
                           for cat, items in categories.items()}
                print(f"fruit={indexed['fruit']}")
                print(f"veg={indexed['veg']}")
            """),
            textwrap.dedent("""\
                all_items = [item for cat_items in indexed.values() for item in cat_items.values()]
                print(f"all={sorted(all_items)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fruit={0: 'apple', 1: 'banana'}" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "apple" in out2
        assert "carrot" in out2

    def test_set_comprehension_with_condition(self, nb_runner):
        """Set comprehension with multiple conditions."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
                evens = {val for row in matrix for val in row if val % 2 == 0 if val > 4}
                print(f"evens={sorted(evens)}")
            """),
            textwrap.dedent("""\
                total = sum(evens)
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[6, 8, 10, 12]" in nb_runner.get_output(1)
        assert "total=36" in nb_runner.get_output(2)

    def test_generator_expression_chain(self, nb_runner):
        """Chained generator expressions."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                words = ['Hello World', 'Python Programming', 'Cash Library']
                letters = list(c.lower() for phrase in words for c in phrase if c.isalpha())
                freq = {}
                for ch in letters:
                    freq[ch] = freq.get(ch, 0) + 1
                top3 = sorted(freq.items(), key=lambda x: -x[1])[:3]
                print(f"top3={top3}")
            """),
            textwrap.dedent("""\
                total_letters = len(letters)
                print(f"total={total_letters}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "top3=" in out1
        out2 = nb_runner.get_output(2)
        assert "total=" in out2
