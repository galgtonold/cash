"""
Interaction test: string maketrans and translate.
Tests str.maketrans for character mapping, translate application,
ROT13-like transformations, and cross-cell encoding pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringMaketransTranslate:
    """Test string maketrans and translate across cells."""

    def test_maketrans_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic translation
            "table = str.maketrans('aeiou', '12345')\ntext = 'hello world'\ntranslated = text.translate(table)\nprint(f'translated={translated}')",
            # Cell 2: delete characters
            "delete_table = str.maketrans('', '', 'lo')\ncleaned = text.translate(delete_table)\nprint(f'cleaned={cleaned}')",
            # Cell 3: combine
            "combined = translated.translate(delete_table)\nprint(f'combined={combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "translated=h2ll4 w4rld" in out1
        out2 = nb_runner.get_output(2)
        assert "cleaned=he wrd" in out2
        out3 = nb_runner.get_output(3)
        assert "combined=h2" in out3

    def test_maketrans_edit(self, nb_runner):
        nb_runner.create_notebook([
            "table = str.maketrans('abc', 'xyz')\ntext = 'abcdef'\nresult = text.translate(table)\nprint(f'result={result}')",
            "has_a = 'a' in result\nprint(f'has_a={has_a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=xyzdef" in nb_runner.get_output(1)
        assert "has_a=False" in nb_runner.get_output(2)

        # Edit mapping
        nb_runner.set_cell_source(1, "table = str.maketrans('def', 'DEF')\ntext = 'abcdef'\nresult = text.translate(table)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "result=abcDEF" in nb_runner.get_output(1)
        assert "has_a=True" in nb_runner.get_output(2)

    def test_maketrans_cache(self, nb_runner):
        nb_runner.create_notebook([
            "table = str.maketrans('0123456789', 'abcdefghij')\ndigits = '12345'\nencoded = digits.translate(table)\nprint(f'encoded={encoded}')",
            "length = len(encoded)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=bcdef" in nb_runner.get_output(1)
        assert "length=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=5" in nb_runner.get_output(2)
