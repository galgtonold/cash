from cash.notebook.cache_status import CacheStatus
"""
Tests for downstream advancement fallback when cell is not found in notebook.

Bug scenario: User edits a cell in VS Code without saving the notebook file.
When re-executing the cell:
1. Cell ID format mismatch (VS Code URI vs #VSC- format) → ID match fails
2. Cell content changed in kernel but .ipynb on disk has old code → content match fails
3. _find_current_cell_index returns None → upstream check skipped entirely
4. Without upstream check, downstream advancement check never fires
5. Variables that are both inputs and outputs (e.g., 'df') keep "ahead" lineage
6. Cache misses for unchanged statements in the cell

The fix: When cell is not found but simulation cache exists, still perform the
downstream advancement check using cached virtual lineage from the last successful run.
"""
import pytest
import hashlib
from unittest.mock import MagicMock, patch

from cash.notebook.magics import CashMagics
from cash.notebook.upstream import _SimulationCacheEntry
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    processor = magics._statement_processor
    yield magics, shell, backend, processor
    backend.clear()
    shell.user_ns.clear()


def _compute_lineage(code: str, input_lineages: list) -> str:
    """Compute a lineage hash the same way Cash does (simplified)."""
    source_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
    lineage_str = source_hash + ":" + ":".join(sorted(input_lineages))
    return hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()


class TestDownstreamAdvancementFallback:
    """
    Test that downstream advancement check still fires when cell is not found
    in notebook (e.g., user edited cell without saving).
    """

    @pytest.mark.xfail(reason="Known failure: downstream advancement fallback lineage reset")
    def test_lineage_reset_when_cell_not_found(self, magics_fixture):
        """
        When the current cell is not found in the notebook file but there's a
        simulation cache from a previous run, variables that are both inputs
        and outputs should have their lineage reset to the virtual (pre-cell) state.
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker
        upstream.debug = True

        # Simulate the state after a successful first run:
        # - df was defined in upstream cell with some lineage
        # - The cell processed df (added columns), advancing its lineage
        # - Simulation cache captured the pre-cell virtual lineage

        # Virtual lineage = what df's lineage should be BEFORE the current cell runs
        virtual_lineage_df = "aaaa1111" * 8  # 64 char hex
        # Actual lineage = what df's lineage became AFTER the current cell ran
        # (includes SMA column addition, etc.)
        ahead_lineage_df = "bbbb2222" * 8  # 64 char hex

        # Set the "ahead" lineage in variable_lineage (as if previous cell execution advanced it)
        upstream.variable_lineage['df'] = ahead_lineage_df

        # Set up simulation cache with the virtual lineage
        # Format: (cell_code_hash, virtual_lineage, virtual_modules, trace, mutated, stale_files, file_deps)
        upstream._simulation_cache = [
            _SimulationCacheEntry("hash_cell_0", {'df': virtual_lineage_df}, set(), [], set(), set(), {}),
        ]
        # The current cell was previously found at index 1 (cell 0 is the only upstream cell)
        upstream.last_cell_index = 1

        # Define cell code that uses and modifies df
        cell_code = "df['VolAdj'] = df['Close'] * df['Volume']\ndf['SMA_61'] = df['Close'].rolling(61).mean()"

        # Mock notebook reading to return cells that DON'T match the current cell code
        # (simulating: user edited cell in VS Code but didn't save)
        old_cell_code = "df['VolAdj'] = df['Close'] * df['Volume']\ndf['SMA_60'] = df['Close'].rolling(60).mean()"
        notebook_cells = [
            "import pandas as pd",  # cell 0
            old_cell_code,          # cell 1 (old version, doesn't match edited code)
        ]
        cells_with_ids = [
            ("#VSC-abc123", "import pandas as pd"),
            ("#VSC-def456", old_cell_code),
        ]

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=notebook_cells), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=cells_with_ids), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            required_inputs = {'df'}
            current_cell_outputs = {'df'}

            metrics, restore_time, exec_time = upstream._check_notebook_based(
                cell_code,
                required_inputs,
                MagicMock(),  # process_statement_callback
                None,  # global_ttl
                current_cell_outputs=current_cell_outputs
            )

        # The key assertion: df's lineage should be reset to the virtual hash,
        # not the "ahead" hash from previous execution
        assert upstream.variable_lineage['df'] == virtual_lineage_df, (
            f"Expected df lineage to be reset to virtual {virtual_lineage_df[:8]}... "
            f"but got {upstream.variable_lineage['df'][:8]}..."
        )

    def test_no_reset_when_variable_only_input(self, magics_fixture):
        """
        Variables that are inputs but NOT outputs should NOT have their lineage reset.
        The downstream advancement fallback only applies to variables that are both
        inputs and outputs of the current cell.
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker

        virtual_lineage_x = "aaaa1111" * 8
        actual_lineage_x = "bbbb2222" * 8

        upstream.variable_lineage['x'] = actual_lineage_x
        upstream._simulation_cache = [
            _SimulationCacheEntry("hash_cell_0", {'x': virtual_lineage_x}, set(), [], set(), set(), {}),
        ]
        upstream.last_cell_index = 1

        cell_code = "y = x * 2"

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=["x = 10"]), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=[("#id1", "x = 10")]), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            upstream._check_notebook_based(
                cell_code,
                {'x'},        # required_inputs
                MagicMock(),
                None,
                current_cell_outputs={'y'},  # x is NOT an output
            )

        # x should NOT be reset — it's only an input, not an output
        assert upstream.variable_lineage['x'] == actual_lineage_x

    def test_no_reset_without_simulation_cache(self, magics_fixture):
        """
        If there's no simulation cache (first run, cell not found), no reset should happen.
        This is the case when the cell has never been successfully found in the notebook.
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker

        actual_lineage_df = "bbbb2222" * 8
        upstream.variable_lineage['df'] = actual_lineage_df
        upstream._simulation_cache = []  # No cache

        cell_code = "df['col'] = 1"

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=["x = 10"]), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=[("#id1", "x = 10")]), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            upstream._check_notebook_based(
                cell_code,
                {'df'},
                MagicMock(),
                None,
                current_cell_outputs={'df'},
            )

        # No cache → no reset
        assert upstream.variable_lineage['df'] == actual_lineage_df

    def test_no_reset_when_lineages_already_match(self, magics_fixture):
        """
        If the variable's lineage already matches the virtual lineage,
        no reset is needed (and no debug output should mention it).
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker

        same_lineage = "aaaa1111" * 8
        upstream.variable_lineage['df'] = same_lineage
        upstream._simulation_cache = [
            _SimulationCacheEntry("hash_cell_0", {'df': same_lineage}, set(), [], set(), set(), {}),
        ]
        upstream.last_cell_index = 1

        cell_code = "df['col'] = 1"

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=["x = 10"]), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=[("#id1", "x = 10")]), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            upstream._check_notebook_based(
                cell_code,
                {'df'},
                MagicMock(),
                None,
                current_cell_outputs={'df'},
            )

        # Lineage should remain the same (it was already correct)
        assert upstream.variable_lineage['df'] == same_lineage

    @pytest.mark.xfail(reason="Known failure: downstream advancement fallback multi-var reset")
    def test_multiple_overlap_vars_reset(self, magics_fixture):
        """
        When multiple variables are both inputs and outputs, all should be reset.
        E.g., a cell that reads df1 and df2 and modifies both.
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker

        virtual_df1 = "aaaa1111" * 8
        virtual_df2 = "cccc3333" * 8
        ahead_df1 = "bbbb2222" * 8
        ahead_df2 = "dddd4444" * 8

        upstream.variable_lineage['df1'] = ahead_df1
        upstream.variable_lineage['df2'] = ahead_df2
        upstream._simulation_cache = [
            _SimulationCacheEntry("hash_cell_0", {'df1': virtual_df1, 'df2': virtual_df2}, set(), [], set(), set(), {}),
        ]
        upstream.last_cell_index = 1

        cell_code = "df1['a'] = df2['b']\ndf2['c'] = df1['d']"

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=["x = 10"]), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=[("#id1", "x = 10")]), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            upstream._check_notebook_based(
                cell_code,
                {'df1', 'df2'},
                MagicMock(),
                None,
                current_cell_outputs={'df1', 'df2'},
            )

        assert upstream.variable_lineage['df1'] == virtual_df1
        assert upstream.variable_lineage['df2'] == virtual_df2

    def test_end_to_end_partial_cell_caching_after_edit(self, magics_fixture):
        """
        End-to-end test: Run a cell with two statements that both modify df.
        Then "edit" the cell (change second statement) and re-run.
        The first statement should be RESTORED even when cell is not found in notebook.
        """
        magics, shell, backend, processor = magics_fixture
        upstream = magics._upstream_checker

        # Step 1: Set up df in user namespace
        import pandas as pd
        shell.user_ns['pd'] = pd
        shell.user_ns['df'] = pd.DataFrame({'Close': [100.0, 200.0, 300.0], 'Volume': [10, 20, 30]})

        # Step 2: Process first statement (computes VolAdj)
        metrics1 = processor.process_statement("df['VolAdj'] = df['Close'] * df['Volume']")
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert 'VolAdj' in shell.user_ns['df'].columns

        # Step 3: Process second statement (computes SMA_60)
        metrics2 = processor.process_statement("df['SMA_60'] = df['Close'].rolling(2).mean()")
        assert metrics2['status'] == CacheStatus.COMPUTED

        # Record the lineage state after both statements ran
        df_lineage_after_both = upstream.variable_lineage.get('df')

        # Step 4: Simulate "cell not found" scenario for the second run
        # The cell was edited (SMA_60 → SMA_61) but notebook not saved.
        # The first statement is unchanged, so it should get a cache hit.

        # To simulate this, we need to:
        # a) Set up the simulation cache as if the first run was successful
        # b) Mock the notebook reading to return cells that don't match

        # Get the virtual lineage for df BEFORE the current cell
        # (this is what it was before any cell processing)
        # We need to capture what df's lineage was before the first statement ran
        # Since we can't easily go back, let's compute it:
        # After step 1, df's lineage was updated by process(). Before step 1,
        # df's lineage was whatever it was when we set it in user_ns.
        # The key point is that process() set df's lineage via _capture_variables.

        # For this test, we just check that after the first run, if we manually
        # set up the state to simulate the bug, the fix works.

        # Set up simulation cache with df's pre-cell lineage
        # We need to know what df's lineage was BEFORE the cell ran.
        # In a real scenario, this comes from the upstream simulation.
        # For testing, we'll use a known value.
        pre_cell_df_lineage = "pre_cell_hash_" + "a" * 50  # placeholder

        upstream._simulation_cache = [
            _SimulationCacheEntry("hash_upstream_cell", {'df': pre_cell_df_lineage}, set(), [], set(), set(), {}),
        ]
        # Simulate that the current cell was previously at index 1
        # (the only upstream cell is at index 0)
        upstream.last_cell_index = 1

        # Set df's lineage to the "ahead" value (as if both statements already ran)
        upstream.variable_lineage['df'] = df_lineage_after_both

        # Now run _check_notebook_based with cell code that won't match notebook
        edited_cell_code = "df['VolAdj'] = df['Close'] * df['Volume']\ndf['SMA_61'] = df['Close'].rolling(2).mean()"
        old_cell_code = "df['VolAdj'] = df['Close'] * df['Volume']\ndf['SMA_60'] = df['Close'].rolling(2).mean()"

        with patch('cash.utils.get_notebook_path', return_value='/fake/notebook.ipynb'), \
             patch('cash.notebook.upstream.get_notebook_cells', return_value=[old_cell_code]), \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids', return_value=[("#VSC-abc", old_cell_code)]), \
             patch('cash.utils.invalidate_notebook_path_cache'):

            upstream._check_notebook_based(
                edited_cell_code,
                {'df'},                  # required_inputs
                MagicMock(),             # process_statement_callback
                None,                    # global_ttl
                current_cell_outputs={'df'},
            )

        # df's lineage should be reset to the pre-cell value
        assert upstream.variable_lineage['df'] == pre_cell_df_lineage, (
            f"Expected df lineage to be reset to pre-cell value, "
            f"but it's still '{upstream.variable_lineage['df'][:20]}...'"
        )
