"""End-to-end tests for the public badge rendering API.

These exercise ``_render_interactive_badge`` and ``_print_text_badge``
through the ``CashMagics`` wrapper. They are deliberately concrete about
*what users see* (CACHED label, upstream-step disclosure, etc.) and
indifferent to the rendering implementation.

Implementation-level tests for the BadgeView IR and its renderers live in
``test_badge_view_builder.py``, ``test_badge_html_renderer.py`` and
``test_badge_text_renderer.py``.
"""
import pytest
from unittest.mock import MagicMock, patch
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
from cash.backends import InMemoryBackend
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


# ----------------------------------------------------------------------------
# Iteration context stripping
# ----------------------------------------------------------------------------

class TestIterationContextStripping:
    """The internal ``# __iteration_context__:`` marker is never user-visible."""

    def test_iteration_context_stripped_from_html(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [{
            'code': '# __iteration_context__: abc123hash\nresult = process(item)',
            'status': CacheStatus.COMPUTED, 'total_time': 0.5, 'outputs': ['result'],
        }]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_id')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert 'abc123hash' not in html_str
            assert '__iteration_context__' not in html_str

    def test_text_badge_strips_iteration_context(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        metrics = [{
            'code': '# __iteration_context__: abc123hash\nresult = process(item)',
            'status': CacheStatus.COMPUTED, 'total_time': 0.5, 'outputs': ['result'],
        }]
        magics._print_text_badge(metrics)
        out = capsys.readouterr().out
        assert 'result = process(item)' in out
        assert 'abc123hash' not in out


# ----------------------------------------------------------------------------
# Skipped intermediate steps
# ----------------------------------------------------------------------------

class TestExpandableSkippedSteps:
    """Skipped upstream dependencies render as a collapsible bucket."""

    def test_skipped_steps_expandable(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'upstream_step_1()', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.3, 'total_time': 0.0},
            {'code': 'upstream_step_2()', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.2, 'total_time': 0.0},
            {'code': 'current_step()', 'status': CacheStatus.COMPUTED,
             'total_time': 0.5, 'outputs': ['result']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_skip')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert '<details' in html_str
            assert 'upstream step' in html_str and 'not re-run' in html_str
            assert 'upstream_step_1' in html_str
            assert 'upstream_step_2' in html_str
            assert '0.50' in html_str

    def test_skipped_section_shows_count(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': f'step_{i}()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.1, 'total_time': 0.0}
            for i in range(5)
        ] + [{'code': 'current()', 'status': CacheStatus.COMPUTED,
              'total_time': 0.1, 'outputs': ['r']}]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_count')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert '5 upstream steps not re-run' in html_str

    def test_single_skipped_step_grammar(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'step()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.5, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED,
             'total_time': 0.1, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_singular')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert '1 upstream step not re-run' in html_str
            assert '1 upstream steps not re-run' not in html_str

    def test_skipped_loop_iterations_grouped(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': '# __iteration_context__: a\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': '# __iteration_context__: b\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': '# __iteration_context__: c\nprocess(x)', 'status': CacheStatus.SKIPPED,
             'is_upstream': True, 'saved_time': 0.1, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED,
             'total_time': 0.1, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_skip_loop')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # v3: loop renders as <details> with histogram bars; for-line is
            # syntax-highlighted (`for` wrapped in c3-kw).
            assert 'c3-loop-head' in html_str
            assert '>for</span>' in html_str


# ----------------------------------------------------------------------------
# Upstream loop grouping
# ----------------------------------------------------------------------------

class TestUpstreamLoopGrouping:
    """Upstream loop iterations group the same way as current-cell ones."""

    def test_upstream_loop_iterations_grouped(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': '# __iteration_context__: a\ncompute(item)', 'status': CacheStatus.RESTORED,
             'is_upstream': True, 'total_time': 0.01, 'saved_time': 1.0},
            {'code': '# __iteration_context__: b\ncompute(item)', 'status': CacheStatus.RESTORED,
             'is_upstream': True, 'total_time': 0.01, 'saved_time': 1.0},
            {'code': 'result = analyze(data)', 'status': CacheStatus.COMPUTED,
             'total_time': 0.5, 'outputs': ['result']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_upstream_loop')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # v3: upstream sits under a "upstream context" <details>, loop
            # collapses with the all-cached badge color.
            assert 'upstream context' in html_str
            assert 'c3-loop-head' in html_str
            assert 'compute(item)' in html_str

    def test_upstream_section_shown_with_only_skipped(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'setup()', 'status': CacheStatus.SKIPPED, 'is_upstream': True,
             'saved_time': 0.2, 'total_time': 0.0},
            {'code': 'current()', 'status': CacheStatus.COMPUTED,
             'total_time': 0.1, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_upstream_only_skip')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # v3 renames "UPSTREAM HISTORY" header to a "upstream context" disclosure.
            assert 'upstream context' in html_str or 'c3-skipped' in html_str


# ----------------------------------------------------------------------------
# Text badge
# ----------------------------------------------------------------------------

class TestTextBadge:
    """The text badge mirrors the HTML structure in flat plain text."""

    def test_text_badge_basic_output(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'x = 1', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['x']},
            {'code': 'y = x + 2', 'status': CacheStatus.RESTORED, 'total_time': 0.01,
             'saved_time': 0.5, 'outputs': ['y']},
        ]
        magics._print_text_badge(metrics)
        out = capsys.readouterr().out
        assert '[Cash]' in out
        assert 'EXECUTED' in out
        assert 'CACHED' in out

    def test_text_badge_upstream_and_current_separated(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'setup()', 'status': CacheStatus.RESTORED, 'is_upstream': True,
             'total_time': 0.01, 'saved_time': 0.5},
            {'code': 'compute()', 'status': CacheStatus.COMPUTED, 'total_time': 0.3, 'outputs': ['r']},
        ]
        magics._print_text_badge(metrics)
        out = capsys.readouterr().out
        assert 'Upstream' in out
        assert '^CACHED' in out  # ASCII upstream marker
        # Must survive a legacy-codepage reader: this mode exists to be parsed
        # by a headless agent / nbconvert, not by the kernel that wrote it.
        out.encode('cp1252')


# ----------------------------------------------------------------------------
# Top-level summary behaviour
# ----------------------------------------------------------------------------

class TestRenderInteractiveBadge:
    """Smoke tests for the public interactive badge function."""

    def test_badge_with_no_metrics(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        magics._badge_mode = 'html'
        magics._render_interactive_badge([], display_id='test_empty')

    def test_badge_with_none_metrics(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        magics._badge_mode = 'html'
        magics._render_interactive_badge(None, display_id='test_none')

    def test_badge_mode_off(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        magics._badge_mode = 'off'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(
                [{'code': 'x=1', 'status': CacheStatus.COMPUTED, 'total_time': 0.1, 'outputs': ['x']}],
                display_id='test_off',
            )
            mock_display.assert_not_called()

    def test_badge_current_cell_section(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [
            {'code': 'up()', 'status': CacheStatus.RESTORED, 'is_upstream': True,
             'total_time': 0.01, 'saved_time': 0.5},
            {'code': 'compute()', 'status': CacheStatus.COMPUTED, 'total_time': 0.3, 'outputs': ['r']},
        ]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_sections')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            # v3: upstream rows live in a "upstream context" disclosure;
            # current rows live in the main panel area without a CURRENT CELL banner.
            assert 'upstream context' in html_str
            assert 'compute()' in html_str

    def test_badge_no_upstream_disclosure_without_upstream(self, magics_fixture):
        magics, _shell, _backend = magics_fixture
        metrics = [{'code': 'x = 1', 'status': CacheStatus.COMPUTED,
                    'total_time': 0.1, 'outputs': ['x']}]
        magics._badge_mode = 'html'
        with patch('cash.notebook.ipython.magics.display') as mock_display:
            magics._render_interactive_badge(metrics, display_id='test_no_upstream')
            html_str = _extract_html(mock_display)
            assert html_str is not None
            assert 'upstream context' not in html_str
