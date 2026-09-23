"""match/case (structural pattern matching, Python 3.10+)."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMatchCase:
    """Tests for match/case statement caching."""

    def test_basic_match(self, nb_runner):
        """match/case with literal patterns."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def classify(code):
                    match code:
                        case 200:
                            return 'ok'
                        case 404:
                            return 'not found'
                        case 500:
                            return 'server error'
                        case _:
                            return 'unknown'
                results = [classify(c) for c in [200, 404, 500, 301]]
            """),
                "print(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ok" in out
        assert "not found" in out
        assert "server error" in out
        assert "unknown" in out

    def test_match_guard_propagation(self, nb_runner):
        """match/case with guard conditions, upstream change propagation."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                textwrap.dedent("""\
                def categorize(val, thresh):
                    match val:
                        case x if x > thresh:
                            return 'high'
                        case x if x > thresh // 2:
                            return 'medium'
                        case _:
                            return 'low'
                cats = [categorize(v, threshold) for v in [10, 30, 70]]
            """),
                "print(f'cats={cats}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "high" in nb_runner.get_output(3)
        assert "low" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "threshold = 20")
        nb_runner.run_cells([1, 2, 3])
        out = nb_runner.get_output(3)
        # With threshold=20: 10 is low, 30 is high, 70 is high
        assert "high" in out
