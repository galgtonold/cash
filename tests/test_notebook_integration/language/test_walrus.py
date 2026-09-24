"""The walrus operator across cells."""

import textwrap

import pytest


class TestWalrusOperator:
    """Walrus operator patterns with edit propagation."""

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_walrus_in_while_condition(self, nb_runner):
        """Walrus op in while, edit threshold."""
        nb_runner.create_notebook(
            [
                "data = [5, 3, 8, 1, 9, 2]",
                "results = []\ni = 0\nwhile i < len(data) and (val := data[i]) > 0:\n    results.append(val * 2)\n    i += 1\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [10, 6, 16, 2, 18, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [5, 3, 8, 1, 9, 2, 7, 4]")
        nb_runner.run_all()
        assert "results = [10, 6, 16, 2, 18, 4, 14, 8]" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_walrus_in_list_comp(self, nb_runner):
        """Walrus op in list comprehension, edit data."""
        nb_runner.create_notebook(
            [
                "values = [1, 4, 9, 16, 25, 36]",
                "import math\nfiltered = [(y := math.sqrt(x), x) for x in values if (y := math.sqrt(x)) > 3]\nprint(f'filtered = {filtered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(4.0, 16)" in out
        assert "(5.0, 25)" in out

        nb_runner.set_cell_source(1, "values = [49, 64, 81, 100]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "(7.0, 49)" in out2
        assert "(10.0, 100)" in out2

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_walrus_in_if_chain(self, nb_runner):
        """Walrus in conditional, edit input data."""
        nb_runner.create_notebook(
            [
                "text = 'Hello World Python'",
                "words = text.split()\nif (count := len(words)) > 2:\n    label = f'{count} words'\nelse:\n    label = 'short'\nprint(f'label = {label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = 3 words" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "text = 'Hi'")
        nb_runner.run_all()
        assert "label = short" in nb_runner.get_output(2)

    # Walrus operator (:=) & complex comprehensions — cash caching.
    @pytest.mark.stress
    def test_walrus_propagation(self, nb_runner):
        """Walrus operator result propagation on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                numbers = [2, 8, 3, 15, 7, 20, 1]
                big = [y for x in numbers if (y := x * 2) > 10]
                print(f"big={big}")
            """),
                textwrap.dedent("""\
                count = len(big)
                print(f"count={count}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big=[16, 30, 14, 40]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)

        # Change threshold by changing input
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            numbers = [2, 8, 3, 15, 7, 20, 1]
            big = [y for x in numbers if (y := x * 3) > 10]
            print(f"big={big}")
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "big=[24, 45, 21, 60]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestWalrusOperatorEdits:
    """Editing walrus operator patterns."""

    def test_edit_walrus_in_while(self, nb_runner):
        """Edit a walrus operator used in accumulation."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "total = 0\nresults = []\nfor x in data:\n    total += x\n    results.append(total)\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [1, 3, 6, 10, 15]" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "results = [10, 30, 60]" in nb_runner.get_output(2)

    def test_edit_walrus_in_comprehension(self, nb_runner):
        """Edit list used in filtered comprehension."""
        nb_runner.create_notebook(
            [
                "nums = [1, 5, 3, 8, 2, 9, 4]",
                "big = [x for x in nums if x > 4]\nprint(f'big = {big}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big = [5, 8, 9]" in nb_runner.get_output(2)

        # Edit threshold by changing the filter
        nb_runner.set_cell_source(2, "big = [x for x in nums if x > 3]\nprint(f'big = {big}')")
        nb_runner.run_all()
        assert "big = [5, 8, 9, 4]" in nb_runner.get_output(2)

    def test_edit_walrus_assignment(self, nb_runner):
        """Edit source data that feeds conditional logic."""
        nb_runner.create_notebook(
            [
                "values = [2, 4, 6, 8, 10]",
                "even_count = sum(1 for v in values if v % 2 == 0)\nprint(f'count = {even_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(2)

        # Add odd numbers
        nb_runner.set_cell_source(1, "values = [1, 2, 3, 4, 5, 6]")
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWalrusConditionalExpr:
    """walrus operator and conditional expressions."""

    def test_walrus_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [1, 5, 3, 8, 2, 7]",
                "big = [y for x in data if (y := x * 2) > 8]\nprint(f'big={big}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big=[10, 16, 14]" in nb_runner.get_output(2)

    def test_conditional_expr(self, nb_runner):
        nb_runner.create_notebook(
            [
                "x = 15",
                "label = 'even' if x % 2 == 0 else 'odd'\nsign = 'positive' if x > 0 else 'non-positive'\nprint(f'label={label} sign={sign}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=odd" in nb_runner.get_output(2)
        assert "sign=positive" in nb_runner.get_output(2)

    def test_walrus_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'hello world'",
                "if (n := len(text)) > 5:\n    result = f'long ({n})'\nelse:\n    result = f'short ({n})'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=long (11)" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "text = 'hi'")
        nb_runner.run_all()
        assert "result=short (2)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestWalrusWhileComprehension:
    """walrus operator in while and comprehension."""

    def test_walrus_in_while(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [1, 5, 3, 8, 2, 9]",
                "results = []\ni = 0\nwhile (val := data[i] if i < len(data) else None) is not None:\n    if val > 4:\n        results.append(val)\n    i += 1\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[5, 8, 9]" in nb_runner.get_output(2)

    def test_walrus_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30]",
                "total = 0\nresult = [(total := total + x) for x in data]\nprint(f'result={result} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[10, 30, 60]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = [5, 15, 25]")
        nb_runner.run_all()
        assert "result=[5, 20, 45]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNestedCompWalrus:
    """Test nested comprehension with walrus operator across cells."""

    def test_nested_comp_walrus(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: nested list comp with walrus
                "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nresults = [(s := sum(row), len(row), round(s / len(row), 1)) for row in data]\nprint(f'results={results}')",
                # Cell 2: dict comp with walrus filter
                "words = ['hello', 'world', 'hi', 'python', 'ok', 'testing']\nlong_upper = {w: u for w in words if len(u := w.upper()) > 3}\nprint(f'filtered={long_upper}')",
                # Cell 3: combine
                "total_sum = sum(r[0] for r in results)\nlong_count = len(long_upper)\nprint(f'total={total_sum}')\nprint(f'long={long_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "results=[(6, 3, 2.0), (9, 2, 4.5), (30, 4, 7.5)]" in out1
        out2 = nb_runner.get_output(2)
        assert "'HELLO'" in out2
        assert "'WORLD'" in out2
        assert "'PYTHON'" in out2
        assert "'TESTING'" in out2
        out3 = nb_runner.get_output(3)
        assert "total=45" in out3
        assert "long=4" in out3

    def test_nested_comp_walrus_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nresults = [(s := sum(row), len(row)) for row in data]\nprint(f'count={len(results)}')",
                "avgs = [round(s / n, 1) for s, n in results]\nprint(f'avgs={avgs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avgs=[2.0, 4.5, 7.5]" in nb_runner.get_output(2)

        # Add more data
        nb_runner.set_cell_source(
            1,
            "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 10]]\nresults = [(s := sum(row), len(row)) for row in data]\nprint(f'count={len(results)}')",
        )
        nb_runner.run_cells([1, 2])
        assert "count=4" in nb_runner.get_output(1)
        assert "avgs=[2.0, 4.5, 7.5, 10.0]" in nb_runner.get_output(2)

    def test_nested_comp_walrus_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [10, 20, 30, 40, 50]\nfiltered = [sq for n in nums if (sq := n * n) > 500]\nprint(f'filtered={filtered}')",
                "total = sum(filtered)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered=[900, 1600, 2500]" in nb_runner.get_output(1)
        assert "total=5000" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=5000" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictMergeWalrus:
    """Test dict merge and walrus operator caching."""

    def test_dict_merge_operator(self, nb_runner):
        """Dict merge with | operator, verify caching."""
        nb_runner.create_notebook(
            [
                "d1 = {'a': 1, 'b': 2}",
                "d2 = {'b': 3, 'c': 4}",
                "merged = d1 | d2\nprint(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'a': 1" in out
        assert "'b': 3" in out
        assert "'c': 4" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'b': 3" in out2

    def test_dict_merge_edit(self, nb_runner):
        """Edit one dict, verify merge result updates."""
        nb_runner.create_notebook(
            [
                "base = {'x': 10, 'y': 20}",
                "override = {'y': 99}",
                "final = base | override\nprint(f'y={final[\"y\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "y=99" in out

        nb_runner.set_cell_source(2, "override = {'y': 42}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y=42" in out2

    def test_walrus_in_condition(self, nb_runner):
        """Walrus operator in condition with caching."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "evens = [x for x in data if (y := x % 2) == 0]\ncount = len(evens)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=5" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "count=5" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDictPipeMergeWalrusLoop:
    """dict pipe merge and walrus loop filtering."""

    def test_dict_pipe_merge(self, nb_runner):
        nb_runner.create_notebook(
            [
                "defaults = {'color': 'blue', 'size': 10, 'font': 'Arial'}",
                "overrides = {'size': 20, 'weight': 'bold'}\nmerged = defaults | overrides\nprint(f'color={merged[\"color\"]} size={merged[\"size\"]} weight={merged[\"weight\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "color=blue" in out
        assert "size=20" in out
        assert "weight=bold" in out

    def test_walrus_filter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [5, 3, 8, 1, 9, 2]",
                "it = iter(data)\nresults = []\nwhile (val := next(it, None)) is not None:\n    if val > 4:\n        results.append(val)\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[5, 8, 9]" in nb_runner.get_output(2)

    def test_pipe_merge_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = {'x': 1}",
                "b = {'y': 2}\nc = a | b\nprint(f'keys={sorted(c.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = {'x': 1, 'z': 3}")
        nb_runner.run_all()
        assert "keys=['x', 'y', 'z']" in nb_runner.get_output(2)
