"""
Interaction test: re.findall with groups and re.split.
Tests re.findall with capturing groups, re.split with pattern,
and cross-cell regex parsing pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestReFindallSplit:
    """Test re.findall and re.split across cells."""

    def test_findall_split(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: findall with groups
            "import re\ntext = 'John:25, Jane:30, Bob:22'\nmatches = re.findall(r'(\\w+):(\\d+)', text)\nprint(f'matches={matches}')",
            # Cell 2: re.split
            "data = '1-2-3--4---5'\nparts = re.split(r'-+', data)\nprint(f'parts={parts}')",
            # Cell 3: use findall results
            "names = [m[0] for m in matches]\nages = [int(m[1]) for m in matches]\navg_age = sum(ages) / len(ages)\nprint(f'names={names}')\nprint(f'avg_age={avg_age:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('John', '25')" in out1
        assert "('Jane', '30')" in out1
        out2 = nb_runner.get_output(2)
        assert "parts=['1', '2', '3', '4', '5']" in out2
        out3 = nb_runner.get_output(3)
        assert "names=['John', 'Jane', 'Bob']" in out3
        assert "avg_age=25.7" in out3

    def test_findall_split_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ntext = 'a1b2c3'\ndigits = re.findall(r'\\d', text)\nprint(f'digits={digits}')",
            "total = sum(int(d) for d in digits)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "digits=['1', '2', '3']" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Edit text
        nb_runner.set_cell_source(1, "import re\ntext = 'x9y8z7w6'\ndigits = re.findall(r'\\d', text)\nprint(f'digits={digits}')")
        nb_runner.run_cells([1, 2])
        assert "total=30" in nb_runner.get_output(2)

    def test_findall_split_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import re\nwords = re.findall(r'\\w+', 'Hello, World! How are you?')\nprint(f'words={words}')",
            "count = len(words)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['Hello', 'World', 'How', 'are', 'you']" in nb_runner.get_output(1)
        assert "count=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=5" in nb_runner.get_output(2)
