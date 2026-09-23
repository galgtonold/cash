"""Tests for the %cash_stats magic."""


class TestCashStats:
    """Test %cash_stats magic."""

    def test_stats_initial_state(self, cash_magics):
        assert cash_magics._session.stats["cells_executed"] == 0
        assert cash_magics._session.stats["statements_computed"] == 0
        assert cash_magics._session.stats["statements_restored"] == 0

    def test_stats_after_compute(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        cash_magics.cash_stats("")
        captured = capsys.readouterr()
        assert "Session Statistics" in captured.out

    def test_stats_json_format(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        # Clear any previous output
        capsys.readouterr()
        cash_magics.cash_stats("json")
        captured = capsys.readouterr()
        import json

        # The output should be valid JSON
        data = json.loads(captured.out.strip())
        assert "hit_rate_percent" in data
        # cache_entries was removed from the JSON output in the 2026-05-18
        # overhead pass (listing entries is O(N) on disk backends).
        # Just verify the basic stats fields are present.
        assert "statements_computed" in data

    def test_stats_reset(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        cash_magics.cash_stats("reset")
        assert cash_magics._session.stats["cells_executed"] == 0
        assert cash_magics._session.stats["statements_computed"] == 0

    def test_stats_display_all_fields(self, cash_magics, capsys):
        cash_magics.cash_stats("")
        captured = capsys.readouterr()
        assert "Cells executed:" in captured.out
        assert "Cache hit rate:" in captured.out
        # Gross / overhead / net are shown separately so the headline saving
        # cannot overstate what cash actually bought.
        assert "Gross time saved:" in captured.out
        assert "Cash overhead:" in captured.out
        assert "Net time saved:" in captured.out
        # "Cache entries:" was removed in the 2026-05-18 overhead pass;
        # the command now refers users to %cash_admin for backend info.
        assert "Tracked variables:" in captured.out


def test_stats_say_they_cover_this_kernel_only(cash_magics, capsys):
    """The totals reset on a kernel restart and a user read them as
    the project's. Say the scope, and where the on-disk numbers are."""
    cash_magics.cash("", "x = 42")
    cash_magics.cash_stats("")
    out = capsys.readouterr().out
    assert "since this kernel started" in out, out
    assert "cash info" in out and "entries" in out, out
