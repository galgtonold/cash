"""Integration test to reproduce missing timing for skipped upstream items after restart.

The user reports that skipped upstream items (e.g., df = df.sort_values(...))
show '-' for timing in the badge after kernel restart. This test reproduces
that scenario by running ONLY a downstream cell after reset, forcing upstream
simulation + backwards restoration.

Root cause: TieredBackend (the default backend) was missing get_metadata() and
set_metadata_only() methods. Metadata for cheap statements (rejected by the
TieredBackend promotion policy) was never persisted to disk, so after kernel
restart the metadata lookup fell back to backend.get() which also returned None.
"""
import re
import pytest


def _get_badge_html(cell) -> str:
    """Extract badge HTML from cell outputs."""
    for output in cell.get('outputs', []):
        if output.output_type in ('execute_result', 'display_data'):
            data = output.get('data', {})
            html = data.get('text/html', '')
            if html and ('Skipped' in html or 'Restored' in html):
                return html
    return ''


def _full_restart_code(vars_to_clear):
    """Generate code that simulates a full kernel restart."""
    var_list = ', '.join(f"'{v}'" for v in vars_to_clear)
    return f"""
try:
    _cash_magics = get_ipython().magics_manager.registry.get('CashMagics')
    if _cash_magics:
        _cash_magics._tracking_state.variable_sources.clear()
        _cash_magics._tracking_state.variable_hashes.clear()
        if hasattr(_cash_magics, '_statement_processor'):
            _cash_magics._statement_processor.variable_lineage.clear()
            _cash_magics._statement_processor.executed_cell_codes.clear()
            _cash_magics._statement_processor.executed_cell_hashes.clear()
except Exception:
    pass

for _v in [{var_list}]:
    try:
        del globals()[_v]
    except KeyError:
        pass
"""


@pytest.mark.core
@pytest.mark.timeout(60)
class TestSkippedTimingAfterRestart:
    """Reproduce the issue where skipped items show '-' for timing after restart."""

    def test_skipped_items_badge_shows_timing_not_dash(self, nb_runner, tmp_path):
        """
        Core test: after kernel restart, run ONLY the last cell.
        The badge's "Skipped" section should show actual timing, not '-'.
        
        Checks the badge HTML directly for the absence of '-' in timing.
        """
        import pandas as pd
        import numpy as np
        
        csv_path = tmp_path / "test_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        
        np.random.seed(42)
        n = 50000
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=n, freq='h').astype(str),
            'Ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n),
            'Close': np.random.randn(n).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, n),
        })
        df.to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            f"df = pd.read_csv('{csv_path_str}')",
            "df = df.sort_values(by=['Ticker', 'Date'])",
            "df['Date'] = pd.to_datetime(df['Date'])",
            "import time; time.sleep(1.5)\ndf['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())",
            "print(len(df))",
        ])
        nb_runner.start_kernel()
        
        # First run - execute all cells normally
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        assert str(n) in output1, f"Expected {n} in output, got: {output1}"

        # Simulate FULL kernel restart
        nb_runner.reset_cash_state()
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            nb_runner.client.kc._async_execute_interactive(
                _full_restart_code(['pd', 'np', 'df', 'time']),
                store_history=False
            )
        )
        
        # Run ONLY last cell — triggers upstream simulation
        nb_runner.run_cell(6)
        
        # Get the badge HTML
        badge_html = _get_badge_html(nb_runner.nb.cells[5])  # 0-indexed
        
        # Result should still be correct
        raw6 = nb_runner.get_raw_output(6)
        assert str(n) in raw6, f"Expected {n} in output, got: {raw6[:500]}"
        
        # Check that the badge has a Skipped section
        assert 'Skipped' in badge_html, \
            "Expected 'Skipped' section in badge"
        
        # Find all skipped item timing values in the badge HTML.
        # Skipped rows have class skip_*_d and a timing td at the end.
        # The timing column for skipped items uses s_time_str which is either
        # "{s_saved:.2f}s" or "-".
        # We want to ensure that non-trivial items (sort_values, to_datetime)
        # show actual timing, not "-".
        
        # Extract skipped item rows - they have the ⏩ emoji and display:none
        skipped_rows = re.findall(
            r'<tr class="skip_.*?</tr>',
            badge_html,
            re.DOTALL
        )
        
        print(f"\nSkipped rows found: {len(skipped_rows)}")
        
        items_with_dash = []
        items_with_timing = []
        for row in skipped_rows:
            # Get code snippet from the row
            code_match = re.search(r'font-size: 10px;">(.*?)</span>', row)
            code = code_match.group(1) if code_match else 'unknown'
            
            # Get the last td content (timing)
            tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            timing_td = tds[-1].strip() if tds else ''
            
            # Print each item (ASCII safe)
            code_ascii = code.encode('ascii', 'replace').decode()
            timing_ascii = timing_td.encode('ascii', 'replace').decode()
            print(f"  Row: code='{code_ascii}' timing='{timing_ascii}'")
            
            if timing_td == '-':
                items_with_dash.append(code)
            else:
                items_with_timing.append((code, timing_td))
        
        # sort_values and to_datetime should have timing, not '-'
        for code in items_with_dash:
            assert 'sort_values' not in code and 'to_datetime' not in code, \
                f"Item '{code}' shows '-' but should show timing"

    def test_skipped_items_metadata_hits(self, nb_runner, tmp_path):
        """
        Verify all skipped items find their metadata from disk cache.
        
        This test specifically exercises the TieredBackend path (default backend)
        where cheap statements (below promotion threshold) would previously
        have no metadata on disk. Now set_metadata_only ensures metadata
        is always persisted regardless of the promotion policy.
        """
        import pandas as pd
        import numpy as np
        
        csv_path = tmp_path / "test_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
        })
        df.to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            f"df = pd.read_csv('{csv_path_str}')",
            "df = df.sort_values('A')",
            "import time; time.sleep(1.5)\nresult = df['A'].sum()",
            "print(result)",
        ])
        nb_runner.start_kernel()
        
        # First run
        nb_runner.run_all()
        output1 = nb_runner.get_output(5)
        result_val = output1.strip()
        assert result_val, f"Expected output, got: {output1}"

        # Enable debug
        nb_runner.enable_debug()

        # Simulate FULL kernel restart
        nb_runner.reset_cash_state()
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            nb_runner.client.kc._async_execute_interactive(
                _full_restart_code(['pd', 'np', 'df', 'result', 'time']),
                store_history=False
            )
        )
        
        # Run ONLY cell 5 (downstream)
        nb_runner.run_cell(5)
        
        raw5 = str(nb_runner.get_raw_output(5))
        
        # Result should still be correct
        assert result_val in raw5, \
            f"Expected '{result_val}' in output, got: {raw5[:500]}"
        
        # Check for metadata hits (no misses for data-bearing statements)
        # Note: import statements and simple assignments may legitimately miss
        # because they execute so fast (<1ms) that no timing data is stored.
        # The key check is that sort_values and read_csv DON'T miss.
        if "miss cache" in raw5:
            missed_lines = [l for l in raw5.split('\n') if 'miss cache' in l]
            # Filter out import/trivial misses
            data_misses = [l for l in missed_lines 
                          if 'import ' not in l and "data_path" not in l
                          and "= '" not in l]
            if data_misses:
                pytest.fail(f"Data-bearing skipped statements missed cache metadata: {data_misses}")
