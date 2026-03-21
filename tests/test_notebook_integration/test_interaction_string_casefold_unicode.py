"""Batch 428: string casefold and unicode normalization."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringCasefoldUnicode:
    def test_casefold(self, nb_runner):
        nb_runner.create_notebook([
            "a = 'Straße'\nb = 'STRASSE'",
            "match = a.casefold() == b.casefold()\nfolded = a.casefold()\nprint(f'match={match} folded={folded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match=True" in nb_runner.get_output(2)
        assert "folded=strasse" in nb_runner.get_output(2)

    def test_unicode_normalize(self, nb_runner):
        nb_runner.create_notebook([
            "import unicodedata\ns1 = 'caf\\u00e9'\ns2 = 'cafe\\u0301'",
            "eq_raw = s1 == s2\nn1 = unicodedata.normalize('NFC', s1)\nn2 = unicodedata.normalize('NFC', s2)\neq_norm = n1 == n2\nprint(f'eq_raw={eq_raw} eq_norm={eq_norm}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq_raw=False" in nb_runner.get_output(2)
        assert "eq_norm=True" in nb_runner.get_output(2)

    def test_casefold_edit(self, nb_runner):
        nb_runner.create_notebook([
            "word = 'Hello'",
            "lower = word.lower()\ncfold = word.casefold()\nprint(f'lower={lower} cfold={cfold}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lower=hello" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "word = 'Straße'")
        nb_runner.run_all()
        assert "cfold=strasse" in nb_runner.get_output(2)
