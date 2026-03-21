"""
Interaction test: string casefold and unicode normalization.
Tests casefold for case-insensitive comparisons, unicode normalization
concepts, and cross-cell string equality checks.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringCasefoldNorm:
    """Test string casefold and normalization across cells."""

    def test_casefold_comparisons(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: casefold comparisons
            "s1 = 'Straße'\ns2 = 'STRASSE'\ns3 = 'straße'\nprint(f'lower_eq={s1.lower() == s2.lower()}')\nprint(f'casefold_eq={s1.casefold() == s2.casefold()}')\nprint(f's1_cf={s1.casefold()}')",
            # Cell 2: build case-insensitive lookup
            "words = ['Hello', 'WORLD', 'Python', 'hello', 'python']\nunique_cf = set(w.casefold() for w in words)\nprint(f'original={len(words)}')\nprint(f'unique={len(unique_cf)}')",
            # Cell 3: case-insensitive search
            "search = 'HELLO'\nfound = [w for w in words if w.casefold() == search.casefold()]\nprint(f'matches={found}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "lower_eq=False" in out1
        assert "casefold_eq=True" in out1
        assert "s1_cf=strasse" in out1
        out2 = nb_runner.get_output(2)
        assert "original=5" in out2
        assert "unique=3" in out2
        out3 = nb_runner.get_output(3)
        assert "matches=['Hello', 'hello']" in out3

    def test_casefold_edit(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['Cat', 'DOG', 'cat', 'Bird', 'dog']\nunique_cf = sorted(set(w.casefold() for w in words))\nprint(f'unique={unique_cf}')",
            "lookup = {w.casefold(): w for w in words}\nprint(f'keys={sorted(lookup.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=['bird', 'cat', 'dog']" in nb_runner.get_output(1)

        # Add more words
        nb_runner.set_cell_source(1, "words = ['Cat', 'DOG', 'cat', 'Bird', 'dog', 'FISH']\nunique_cf = sorted(set(w.casefold() for w in words))\nprint(f'unique={unique_cf}')")
        nb_runner.run_cells([1, 2])
        assert "unique=['bird', 'cat', 'dog', 'fish']" in nb_runner.get_output(1)

    def test_casefold_cache(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'The Quick Brown Fox'\ncf = text.casefold()\nprint(f'cf={cf}')",
            "word_count = len(cf.split())\nprint(f'words={word_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cf=the quick brown fox" in nb_runner.get_output(1)
        assert "words=4" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "words=4" in nb_runner.get_output(2)
