"""
Interaction test: regex compile with named groups and finditer.
Tests compiled regex patterns with named groups, finditer iteration,
and group extraction across cells.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRegexCompileGroups:
    """Test compiled regex with named groups across cells."""

    def test_regex_compile_finditer(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: compile pattern with named groups
            "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'pattern_type={type(pattern).__name__}')",
            # Cell 2: finditer over text
            "text = 'Born 1990-05-14, graduated 2012-06-20, married 2018-09-03'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')\nfor m in matches:\n    print(f\"{m['year']}/{m['month']}/{m['day']}\")",
            # Cell 3: derive summary
            "years = [int(m['year']) for m in matches]\nspan = max(years) - min(years)\nprint(f'year_span={span}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "pattern_type=Pattern" in out1
        out2 = nb_runner.get_output(2)
        assert "count=3" in out2
        assert "1990/05/14" in out2
        out3 = nb_runner.get_output(3)
        assert "year_span=28" in out3

    def test_regex_pattern_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'pattern_type={type(pattern).__name__}')",
            "text = 'Born 1990-05-14, graduated 2012-06-20, married 2018-09-03'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')",
            "years = [int(m['year']) for m in matches]\nspan = max(years) - min(years)\nprint(f'year_span={span}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "year_span=28" in nb_runner.get_output(3)

        # Edit text to have more dates
        nb_runner.set_cell_source(2, "text = 'Born 1990-05-14, grad 2012-06-20, married 2018-09-03, child 2020-11-15'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')")
        nb_runner.run_cells([2, 3])
        assert "count=4" in nb_runner.get_output(2)
        assert "year_span=30" in nb_runner.get_output(3)

    def test_regex_cache_correctness(self, nb_runner):
        nb_runner.create_notebook([
            "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'compiled=True')",
            "text = 'Event on 2023-01-15 and 2023-12-31'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')",
            "months = sorted(set(int(m['month']) for m in matches))\nprint(f'months={months}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "months=[1, 12]" in nb_runner.get_output(3)

        # Re-run without changes - should use cache
        nb_runner.run_all()
        assert "months=[1, 12]" in nb_runner.get_output(3)
