"""Batch 423: re module findall and sub patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestReFindallSub:
    def test_findall(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ntext = 'Price: $10.50, Discount: $2.00, Total: $8.50'",
            "prices = re.findall(r'\\$([\\d.]+)', text)\nprint(f'prices={prices}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "prices=['10.50', '2.00', '8.50']" in nb_runner.get_output(2)

    def test_sub(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ntext = 'foo  bar   baz    qux'",
            "cleaned = re.sub(r'\\s+', ' ', text)\nprint(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=foo bar baz qux" in nb_runner.get_output(2)

    def test_regex_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import re\ndata = 'user123_test456'",
            "nums = re.findall(r'\\d+', data)\nprint(f'nums={nums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums=['123', '456']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import re\ndata = 'abc99_def88_ghi77'")
        nb_runner.run_all()
        assert "nums=['99', '88', '77']" in nb_runner.get_output(2)
