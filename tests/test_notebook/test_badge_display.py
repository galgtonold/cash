"""Tests for badge display improvements: loop grouping, skipped expansion, iteration context stripping."""
import pytest
from unittest.mock import MagicMock, patch
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.magics import CashMagics
from cash.notebook import badge_renderer as _badge
from cash.notebook.badge_renderer import format_loop_var
from cash.backends.backend import InMemoryBackend
from cash.notebook.cache_status import CacheStatus


def _extract_html(mock_display):
    """Extract HTML string from a mocked display() call."""
    if not mock_display.called:
        return None
    html_obj = mock_display.call_args[0][0]
    if hasattr(html_obj, 'data'):
        return html_obj.data
    return str(html_obj)


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
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


# ============================================================================
# Tests for _group_loop_iterations
# ============================================================================

class TestGroupLoopIterations:
    """Tests for the _group_loop_iterations static method."""

    def test_empty_list_returns_empty(self):
        """Empty metrics list returns empty result."""
        result = _badge.group_loop_iterations([])
        assert result == []

    def test_single_non_loop_metric(self):
        """A single non-loop metric passes through as single."""
        metrics = [{'code': 'x = 1', 'status': CacheStatus.COMPUTED}]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 1
        assert result[0]['type'] == 'single'
        assert result[0]['metric'] is metrics[0]

    def test_multiple_non_loop_metrics(self):
        """Multiple non-loop metrics all pass through as singles."""
        metrics = [
            {'code': 'x = 1', 'status': CacheStatus.COMPUTED},
            {'code': 'y = x + 2', 'status': CacheStatus.RESTORED},
            {'code': 'print(y)', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 3
        for item in result:
            assert item['type'] == 'single'

    def test_loop_iterations_grouped(self):
        """Consecutive iterations of the same code are grouped."""
        metrics = [
            {'code': '# __iteration_context__: abc123\nresult = process(item)', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: def456\nresult = process(item)', 'status': CacheStatus.RESTORED},
            {'code': '# __iteration_context__: ghi789\nresult = process(item)', 'status': CacheStatus.RESTORED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 1
        assert result[0]['type'] == 'for_loop_group'
        # for_loop_group contains stmt_groups, each a loop_group
        stmt_groups = result[0]['stmt_groups']
        assert len(stmt_groups) == 1
        assert stmt_groups[0]['base_code'] == 'result = process(item)'
        assert len(stmt_groups[0]['metrics']) == 3

    def test_different_loop_codes_separate_groups(self):
        """Different code in loop iterations creates separate stmt_groups under one for_loop_group."""
        metrics = [
            {'code': '# __iteration_context__: abc\nstep_a()', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: abc\nstep_b()', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 1
        assert result[0]['type'] == 'for_loop_group'
        stmt_groups = result[0]['stmt_groups']
        assert len(stmt_groups) == 2
        assert stmt_groups[0]['base_code'] == 'step_a()'
        assert stmt_groups[1]['base_code'] == 'step_b()'

    def test_mixed_loop_and_non_loop(self):
        """Mix of loop and non-loop metrics produces correct types."""
        metrics = [
            {'code': 'setup()', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nprocess()', 'status': CacheStatus.RESTORED},
            {'code': '# __iteration_context__: b\nprocess()', 'status': CacheStatus.RESTORED},
            {'code': 'cleanup()', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 3
        assert result[0]['type'] == 'single'
        assert result[1]['type'] == 'for_loop_group'
        assert len(result[1]['stmt_groups'][0]['metrics']) == 2
        assert result[2]['type'] == 'single'

    def test_loop_vars_from_metric(self):
        """Loop variable values are collected from loop_vars in metrics."""
        metrics = [
            {'code': '# __iteration_context__: a\nprocess(ticker)', 'status': CacheStatus.COMPUTED,
             'loop_vars': {'ticker': 'AAPL'}},
            {'code': '# __iteration_context__: b\nprocess(ticker)', 'status': CacheStatus.COMPUTED,
             'loop_vars': {'ticker': 'MSFT'}},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert len(result) == 1
        group = result[0]
        assert group['type'] == 'for_loop_group'
        assert group['loop_var_name'] == 'ticker'
        assert 'AAPL' in group['loop_var_values']
        assert 'MSFT' in group['loop_var_values']

    def test_loop_var_assignment_detection(self):
        """Loop variable assignment (e.g., ticker = 'AAPL') stays separate from loop groups."""
        metrics = [
            {'code': "ticker = 'AAPL'", 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nprocess(ticker)', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nanalyze(ticker)', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        # The ticker assignment is a 'single', loop items are a 'for_loop_group'
        assert result[0]['type'] == 'single'
        loop_groups = [r for r in result if r['type'] == 'for_loop_group']
        assert len(loop_groups) == 1

    def test_interleaved_var_assigns_merged(self):
        """Loop var assignments interleaved between iteration groups create separate for_loop_groups."""
        metrics = [
            {'code': "ticker = 'AAPL'", 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a1\nprocess(ticker)', 'status': CacheStatus.COMPUTED},
            {'code': "ticker = 'MSFT'", 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a2\nprocess(ticker)', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        # Each loop section is separated by the ticker assignment, so we get:
        # single, for_loop_group, single, for_loop_group
        loop_groups = [r for r in result if r['type'] == 'for_loop_group']
        assert len(loop_groups) == 2

    def test_non_identifier_assignment_not_detected(self):
        """Assignments to non-simple identifiers are not treated as loop var assignments."""
        metrics = [
            {'code': 'result["key"] = compute()', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nprocess()', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        # The dict assignment should stay as a single, not be absorbed
        assert result[0]['type'] == 'single'

    def test_underscore_prefix_assignment_not_detected(self):
        """Assignments to underscore-prefixed vars are not treated as loop var assignments."""
        metrics = [
            {'code': '_internal = 42', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nprocess()', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert result[0]['type'] == 'single'
        loop_groups = [r for r in result if r['type'] == 'for_loop_group']
        # Should not have absorbed the underscore var
        for g in loop_groups:
            assert g.get('loop_var_name') != '_internal'

    def test_multiline_assignment_not_detected(self):
        """Multi-line assignments are not treated as loop var assignments."""
        metrics = [
            {'code': 'result = some_func(\n    arg1, arg2\n)', 'status': CacheStatus.COMPUTED},
            {'code': '# __iteration_context__: a\nprocess()', 'status': CacheStatus.COMPUTED},
        ]
        result = _badge.group_loop_iterations(metrics)
        assert result[0]['type'] == 'single'

    def test_nested_control_inside_loop_grouped(self):
        """Nested if/try inside a for loop are grouped within the loop group.

        When a for loop body contains an if-statement processed per-statement,
        the resulting metrics carry both __iteration_context__ and
        control_context.  They must stay inside the for_loop_group, not
        break out as separate control_group rows.
        """
        metrics = [
            # Regular loop statement - iteration 1
            {'code': '# __iteration_context__: iter1\nx = compute(i)', 'status': CacheStatus.COMPUTED,
             'loop_vars': {'i': 0}},
            # Nested if body statement inside loop - iteration 1
            {'code': '# __iteration_context__: iter1\n# control_context: if_abc\nlog(x)',
             'status': CacheStatus.COMPUTED, 'loop_vars': {'i': 0}, 'control_context': 'if_abc'},
            # Regular loop statement - iteration 2
            {'code': '# __iteration_context__: iter2\nx = compute(i)', 'status': CacheStatus.RESTORED,
             'loop_vars': {'i': 1}},
            # Nested if body statement inside loop - iteration 2
            {'code': '# __iteration_context__: iter2\n# control_context: if_abc\nlog(x)',
             'status': CacheStatus.RESTORED, 'loop_vars': {'i': 1}, 'control_context': 'if_abc'},
        ]
        result = _badge.group_loop_iterations(metrics)
        # Everything should be in ONE for_loop_group (not a mix of loop + control groups)
        assert len(result) == 1, f"Expected 1 group, got {len(result)}: {[r['type'] for r in result]}"
        assert result[0]['type'] == 'for_loop_group'
        stmt_groups = result[0]['stmt_groups']
        # Two distinct statements: "x = compute(i)" and "log(x)"
        assert len(stmt_groups) == 2
        # base_code should NOT contain control_context comment
        for sg in stmt_groups:
            assert '# control_context' not in sg['base_code']
        assert stmt_groups[0]['base_code'] == 'x = compute(i)'
        assert stmt_groups[1]['base_code'] == 'log(x)'
        # Each statement group has 2 iteration metrics
        assert len(stmt_groups[0]['metrics']) == 2
        assert len(stmt_groups[1]['metrics']) == 2


# ============================================================================
# Tests for _render_for_loop_group and _render_loop_stmt_row
# ============================================================================

class TestRenderForLoopGroup:
    """Tests for _render_for_loop_group and _render_loop_stmt_row."""

    @staticmethod
    def _make_for_loop_group(stmt_groups, loop_var_name=None, loop_var_values=None):
        """Helper to wrap loop_group dicts into a for_loop_group."""
        all_loop_var_names = [loop_var_name] if loop_var_name else []
        all_loop_var_values = {loop_var_name: loop_var_values or []} if loop_var_name else {}
        return {
            'type': 'for_loop_group',
            'stmt_groups': stmt_groups,
            'loop_var_name': loop_var_name,
            'loop_var_values': loop_var_values or [],
            'all_loop_var_names': all_loop_var_names,
            'all_loop_var_values': all_loop_var_values,
        }

    @staticmethod
    def _make_stmt_group(base_code, metrics, loop_var_name=None, loop_var_values=None):
        """Helper to create a loop_group (statement group) dict."""
        return {
            'type': 'loop_group',
            'base_code': base_code,
            'metrics': metrics,
            'loop_var_name': loop_var_name,
            'loop_var_values': loop_var_values or [],
        }

    def test_all_cached_group(self):
        """Group with all RESTORED metrics shows all-cached indicator."""
        sg = self._make_stmt_group('process(item)', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 1.0},
            {'status': CacheStatus.RESTORED, 'total_time': 0.02, 'saved_time': 0.8},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group)
        assert '⚡' in html  # cached icon
        assert 'All 2 cached' in html
        assert 'process(item)' in html

    def test_all_computed_group(self):
        """Group with all COMPUTED metrics shows all-computed indicator."""
        sg = self._make_stmt_group('process(item)', [
            {'status': CacheStatus.COMPUTED, 'total_time': 1.0, 'saved_time': 0.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.5, 'saved_time': 0.0},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group)
        assert '⚙️' in html  # computed icon
        assert 'All 2 computed' in html

    def test_mixed_group(self):
        """Group with mixed statuses shows counts."""
        sg = self._make_stmt_group('process(item)', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 1.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.5, 'saved_time': 0.0},
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 0.9},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group)
        assert '🔄' in html  # mixed icon
        assert '2 cached' in html
        assert '1 computed' in html

    def test_loop_var_display_short(self):
        """Loop variable values shown inline when ≤5 values."""
        sg = self._make_stmt_group('process(ticker)', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 1.0, 'loop_vars': {'ticker': 'AAPL'}},
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 1.0, 'loop_vars': {'ticker': 'MSFT'}},
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 1.0, 'loop_vars': {'ticker': 'GOOGL'}},
        ], loop_var_name='ticker', loop_var_values=['AAPL', 'MSFT', 'GOOGL'])
        group = self._make_for_loop_group([sg], loop_var_name='ticker',
                                          loop_var_values=['AAPL', 'MSFT', 'GOOGL'])
        html = _badge.render_for_loop_group(group)
        assert 'ticker' in html
        assert 'AAPL' in html
        assert 'MSFT' in html
        assert 'GOOGL' in html
        assert '∈' in html

    def test_loop_var_display_long_cutoff(self):
        """Loop variable values truncated when >5 values."""
        values = [f'val_{i}' for i in range(12)]
        sg = self._make_stmt_group('process(x)', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0}
            for _ in values
        ], loop_var_name='x', loop_var_values=values)
        group = self._make_for_loop_group([sg], loop_var_name='x', loop_var_values=values)
        html = _badge.render_for_loop_group(group)
        assert 'val_0' in html
        assert 'val_1' in html
        assert 'val_2' in html
        assert 'val_11' in html  # last value
        assert '…' in html
        assert '12' in html  # iteration count displayed in summary

    def test_no_loop_var_fallback(self):
        """Without loop var info, shows 'N iterations' fallback."""
        sg = self._make_stmt_group('process()', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group)
        assert '3 iterations' in html

    def test_upstream_mode(self):
        """Upstream groups show upstream icon and status-based label."""
        sg = self._make_stmt_group('compute()', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 0.5},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group, is_upstream=True)
        assert '⬆️' in html
        assert 'Restored' in html

    def test_current_mode(self):
        """Non-upstream groups show Loop label."""
        sg = self._make_stmt_group('compute()', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.5, 'saved_time': 0.0},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group, is_upstream=False)
        assert 'Loop' in html

    def test_many_iterations_collapsed(self):
        """More than MAX_VISIBLE*2 iterations shows collapsed middle in stmt row."""
        sg = self._make_stmt_group('process()', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0}
            for _ in range(15)
        ])
        html = _badge.render_loop_stmt_row(sg)
        assert 'more' in html

    def test_saved_time_shown(self):
        """Saved time indicator shown for restored iterations."""
        sg = self._make_stmt_group('slow_computation()', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 5.0, 'loop_vars': {'i': 0}},
        ], loop_var_name='i', loop_var_values=[0])
        group = self._make_for_loop_group([sg], loop_var_name='i', loop_var_values=[0])
        html = _badge.render_for_loop_group(group)
        assert 'Saved' in html or '↑' in html

    def test_details_element_present(self):
        """For loop group uses JS-based expand/collapse for flat rows."""
        sg = self._make_stmt_group('process()', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
        ])
        group = self._make_for_loop_group([sg])
        html = _badge.render_for_loop_group(group)
        # Uses JS onclick toggle instead of <details>/<summary>
        assert 'onclick' in html
        assert '▶' in html  # expand arrow indicator

    def test_iteration_labels_from_loop_vars(self):
        """Iteration rows show loop variable values when available."""
        sg = self._make_stmt_group('process(ticker)', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0, 'loop_vars': {'ticker': 'AAPL'}},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.2, 'saved_time': 0.0, 'loop_vars': {'ticker': 'MSFT'}},
        ], loop_var_name='ticker', loop_var_values=['AAPL', 'MSFT'])
        html = _badge.render_loop_stmt_row(sg)
        assert 'ticker=AAPL' in html
        assert 'ticker=MSFT' in html

    def test_iteration_labels_fallback_to_index(self):
        """Iteration rows fall back to 'iter N' when no loop vars available."""
        sg = self._make_stmt_group('process()', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
        ])
        html = _badge.render_loop_stmt_row(sg)
        assert 'iter 1' in html
        assert 'iter 2' in html

    def test_multiple_stmt_groups(self):
        """For loop with multiple statements shows all of them."""
        sg1 = self._make_stmt_group('step_a(item)', [
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
            {'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'saved_time': 0.0},
        ])
        sg2 = self._make_stmt_group('step_b(item)', [
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 0.5},
            {'status': CacheStatus.RESTORED, 'total_time': 0.01, 'saved_time': 0.5},
        ])
        group = self._make_for_loop_group([sg1, sg2])
        html = _badge.render_for_loop_group(group)
        assert 'step_a(item)' in html
        assert 'step_b(item)' in html
        assert '2 stmts' in html


# ============================================================================
# Tests for iteration context stripping in _render_row
# ============================================================================

class TestIterationContextStripping:
    """Tests for stripping __iteration_context__ from code display."""

    def test_iteration_context_stripped_from_html(self, magics_fixture):
        """HTML badge strips __iteration_context__ from code display."""
        magics, shell, backend = magics_fixture
        metrics = [
            {
                'code': '# __iteration_context__: abc123hash\nresult = process(item)',
                'status': CacheStatus.COMPUTED,
                'total_time': 0.5,
                'outputs': ['result'],
            }
        ]
        # Use _render_interactive_badge and capture the HTML
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_id')
            html_str = _extract_html(mock_display)
            if html_str:
                # Should NOT show the __iteration_context__ hash
                assert '__iteration_context__' not in html_str or 'abc123hash' not in html_str

    def test_text_badge_strips_iteration_context(self, magics_fixture, capsys):
        """Text badge strips __iteration_context__ from code display."""
        magics, shell, backend = magics_fixture
        metrics = [
            {
                'code': '# __iteration_context__: abc123hash\nresult = process(item)',
                'status': CacheStatus.COMPUTED,
                'total_time': 0.5,
                'outputs': ['result'],
            }
        ]
        magics._print_text_badge(metrics)
        captured = capsys.readouterr()
        # Should show the actual code, not the context comment
        assert 'result = process(item)' in captured.out
        assert 'abc123hash' not in captured.out


# ============================================================================
# Tests for expandable skipped steps
# ============================================================================

class TestExpandableSkippedSteps:
    """Tests for skipped steps expandable section in the badge."""

    def test_skipped_steps_expandable(self, magics_fixture):
        """Skipped upstream steps rendered as expandable <details>."""
        magics, shell, backend = magics_fixture
        metrics = [
            {
                'code': 'upstream_step_1()',
                'status': CacheStatus.SKIPPED,
                'is_upstream': True,
                'saved_time': 0.3,
                'total_time': 0.0,
            },
            {
                'code': 'upstream_step_2()',
                'status': CacheStatus.SKIPPED,
                'is_upstream': True,
                'saved_time': 0.2,
                'total_time': 0.0,
            },
            {
                'code': 'current_step()',
                'status': CacheStatus.COMPUTED,
                'total_time': 0.5,
                'outputs': ['result'],
            }
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_skip')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # Should have expandable details for skipped section
            assert '<details' in html_str
            assert 'intermediate dependency step' in html_str
            # Should show individual skipped steps inside
            assert 'upstream_step_1' in html_str
            assert 'upstream_step_2' in html_str
            # Should show saved time summary
            assert '0.50' in html_str  # total skipped saved time (0.3+0.2)

    def test_skipped_section_shows_count(self, magics_fixture):
        """Skipped section summary shows step count."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': f'step_{i}()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.1, 'total_time': 0.0}
            for i in range(5)
        ] + [
            {'code': 'current()', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['r']}
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_count')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert '5 intermediate dependency steps' in html_str

    def test_single_skipped_step_grammar(self, magics_fixture):
        """Single skipped step uses singular grammar."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'step()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.5, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['r']}
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_singular')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert '1 intermediate dependency step' in html_str
            # Should NOT have 'steps' (plural)
            assert '1 intermediate dependency steps' not in html_str

    def test_skipped_loop_iterations_grouped(self, magics_fixture):
        """Skipped loop iterations within the skipped section are grouped."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': '# __iteration_context__: a\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': '# __iteration_context__: b\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': '# __iteration_context__: c\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['r']}
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_skip_loop')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # Should show loop iteration grouping in the skipped section
            # for_loop_group has "N iterations" in its label
            assert 'iterations' in html_str
            assert 'for loop' in html_str


# ============================================================================
# Tests for upstream loop grouping
# ============================================================================

class TestUpstreamLoopGrouping:
    """Tests for loop grouping in the upstream section of the badge."""

    def test_upstream_loop_iterations_grouped(self, magics_fixture):
        """Upstream loop iterations are grouped like current cell iterations."""
        magics, shell, backend = magics_fixture
        metrics = [
            # Upstream loop iterations (RESTORED)
            {'code': '# __iteration_context__: a\ncompute(item)', 'status': CacheStatus.RESTORED,
             'is_upstream': True, 'total_time': 0.01, 'saved_time': 1.0},
            {'code': '# __iteration_context__: b\ncompute(item)', 'status': CacheStatus.RESTORED,
             'is_upstream': True, 'total_time': 0.01, 'saved_time': 1.0},
            # Current cell
            {'code': 'result = analyze(data)', 'status': CacheStatus.COMPUTED,
             'total_time': 0.5, 'outputs': ['result']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_upstream_loop')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # Should show upstream loop group (all cached → 'Restored')
            assert 'Restored' in html_str
            assert '2' in html_str  # 2 iterations
            assert 'compute(item)' in html_str

    def test_upstream_section_shown_with_only_skipped(self, magics_fixture):
        """Upstream history section shown even when there are only skipped items."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'setup()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.2, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_upstream_only_skip')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert 'UPSTREAM HISTORY' in html_str


# ============================================================================
# Tests for format_loop_var
# ============================================================================

class TestFormatLoopVar:
    """Tests for the format_loop_var utility function."""

    def test_format_short_string(self):
        """Short strings pass through."""
        assert format_loop_var('AAPL') == 'AAPL'

    def test_format_long_string_truncated(self):
        """Long strings are truncated to 30 chars."""
        long_str = 'a' * 50
        result = format_loop_var(long_str)
        assert len(result) <= 33  # 30 + '...'
        assert result.endswith('...')

    def test_format_number(self):
        """Numbers use repr."""
        assert format_loop_var(42) == '42'
        assert format_loop_var(3.14) == '3.14'

    def test_format_none(self):
        """None uses repr."""
        assert format_loop_var(None) == 'None'

    def test_format_list(self):
        """Lists use repr and truncate if long."""
        result = format_loop_var([1, 2, 3])
        assert result == '[1, 2, 3]'


# ============================================================================
# Tests for text badge with iteration context
# ============================================================================

class TestTextBadge:
    """Tests for _print_text_badge with new features."""

    def test_text_badge_basic_output(self, magics_fixture, capsys):
        """Text badge produces expected output for basic metrics."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'x = 1', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['x']},
            {'code': 'y = x + 2', 'status': CacheStatus.RESTORED, 'total_time': 0.01,
             'saved_time': 0.5, 'outputs': ['y']},
        ]
        magics._print_text_badge(metrics)
        captured = capsys.readouterr()
        assert '[Cash]' in captured.out
        assert 'COMPUTED' in captured.out
        assert 'RESTORED' in captured.out

    def test_text_badge_upstream_and_current_separated(self, magics_fixture, capsys):
        """Text badge separates upstream and current sections."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'setup()', 'status': CacheStatus.RESTORED, 'is_upstream': True,
             'total_time': 0.01, 'saved_time': 0.5},
            {'code': 'compute()', 'status': CacheStatus.COMPUTED, 'total_time': 0.3, 'outputs': ['r']},
        ]
        magics._print_text_badge(metrics)
        captured = capsys.readouterr()
        assert 'Upstream' in captured.out
        assert '⬆️' in captured.out

    def test_text_badge_strips_iteration_context(self, magics_fixture, capsys):
        """Text badge strips iteration context from code snippets."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': '# __iteration_context__: deadbeef\nprocess(item)', 'status': CacheStatus.COMPUTED,
             'total_time': 0.1, 'outputs': ['r']},
        ]
        magics._print_text_badge(metrics)
        captured = capsys.readouterr()
        assert 'process(item)' in captured.out
        assert 'deadbeef' not in captured.out
        assert '__iteration_context__' not in captured.out


# ============================================================================
# Tests for _render_interactive_badge overall
# ============================================================================

class TestRenderInteractiveBadge:
    """Tests for the full _render_interactive_badge output."""

    def test_badge_with_no_metrics(self, magics_fixture):
        """Badge with empty metrics list doesn't crash."""
        magics, shell, backend = magics_fixture
        magics._badge_mode = 'html'
        # Should not raise
        magics._render_interactive_badge([], display_id='test_empty')

    def test_badge_with_none_metrics(self, magics_fixture):
        """Badge with None metrics list doesn't crash."""
        magics, shell, backend = magics_fixture
        magics._badge_mode = 'html'
        magics._render_interactive_badge(None, display_id='test_none')

    def test_badge_mode_off(self, magics_fixture):
        """Badge in 'off' mode produces no output."""
        magics, shell, backend = magics_fixture
        magics._badge_mode = 'off'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(
                [{'code': 'x=1', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['x']}],
                display_id='test_off'
            )
            mock_display.assert_not_called()

    def test_badge_current_cell_section(self, magics_fixture):
        """Badge shows CURRENT CELL section when there are upstream items too."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'up()', 'status': CacheStatus.RESTORED, 'is_upstream': True,
             'total_time': 0.01, 'saved_time': 0.5},
            {'code': 'compute()', 'status': CacheStatus.COMPUTED, 'total_time': 0.3, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_sections')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert 'UPSTREAM HISTORY' in html_str
            assert 'CURRENT CELL' in html_str

    def test_badge_no_current_cell_header_without_upstream(self, magics_fixture):
        """Badge omits CURRENT CELL header when there are no upstream items."""
        magics, shell, backend = magics_fixture
        metrics = [
            {'code': 'x = 1', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['x']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_no_upstream')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert 'UPSTREAM HISTORY' not in html_str
