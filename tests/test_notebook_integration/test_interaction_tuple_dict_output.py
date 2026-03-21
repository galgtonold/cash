"""
Batch 310: Tuple/dict function output patterns interaction tests.
Tests multiple return value patterns with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestTupleDictOutputInteraction:
    """Test tuple/dict output patterns with cache invalidation."""

    def test_tuple_stats_edit(self, nb_runner):
        """Editing data for tuple-returning stats function."""
        nb_runner.create_notebook([
            "def calc_stats(d):\n    return min(d), max(d), sum(d) / len(d)",
            "vals = [10, 20, 30, 40, 50]",
            "lo, hi, avg = calc_stats(vals)",
            "print(f'lo={lo},hi={hi},avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "lo=10,hi=50,avg=30.0" in out

        nb_runner.set_cell_source(2, "vals = [100, 200, 300]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "lo=100,hi=300,avg=200.0" in out

    def test_dict_analyze_edit(self, nb_runner):
        """Editing data for dict-returning analysis function."""
        nb_runner.create_notebook([
            "def word_info(text):\n    w = text.split()\n    return {'wc': len(w), 'uw': len(set(w))}",
            "text = 'hello world hello python'",
            "info = word_info(text)",
            "print(f\"wc={info['wc']},uw={info['uw']}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "wc=4,uw=3" in out

        nb_runner.set_cell_source(2, "text = 'one two three four five'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "wc=5,uw=5" in out
