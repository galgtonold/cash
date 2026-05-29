"""Batch 345: re (regex) pattern matching with cell edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRegexPatternEdit:
    def test_regex_findall(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ntext = 'Call 555-1234 or 555-5678 for info'",
            "pattern = r'\\d{3}-\\d{4}'\nnumbers = re.findall(pattern, text)\nprint(f'numbers={numbers}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "numbers=['555-1234', '555-5678']" in nb_runner.get_output(2)

    def test_regex_edit_text(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ntext = 'apple 3 banana 7 cherry 12'",
            "nums = [int(x) for x in re.findall(r'\\d+', text)]\ntotal = sum(nums)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=22" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import re\ntext = 'x 100 y 200 z 300'")
        nb_runner.run_all()
        assert "total=600" in nb_runner.get_output(2)

