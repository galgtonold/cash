"""nbconvert preprocessors for stripping cache badges and executing with cash."""

from __future__ import annotations

from .notebook.cache_status import CacheStatus

try:
    from nbconvert.preprocessors import Preprocessor
    HAS_NBCONVERT = True
except ImportError:
    # Provide a dummy base class when nbconvert isn't installed
    HAS_NBCONVERT = False
    class Preprocessor:
        """Dummy base class when nbconvert is not installed."""
        def preprocess(self, nb, resources):
            return nb, resources
        def preprocess_cell(self, cell, resources, index):
            return cell, resources


class CashStripPreprocessor(Preprocessor):
    """
    Preprocessor that strips cash-specific outputs from notebook cells.

    This removes:
    - HTML badge outputs (cache status badges)
    - Debug output lines ([UPSTREAM_DEBUG], [LINEAGE_DEBUG], etc.)
    - Cash magic commands from cell source (%cash_on, %cash_debug, etc.)

    Usage:
        jupyter nbconvert --to html \\
            --Exporter.preprocessors='["cash.nbconvert.CashStripPreprocessor"]' \\
            notebook.ipynb
    """

    # Debug output markers
    DEBUG_MARKERS = [
        '[TIMING', '[UPSTREAM', '[LINEAGE', '[ALREADY',
        '[CACHE', '[CONTROL', 'DEBUG', 'Cash:'
    ]

    # Cash magic commands to strip
    CASH_MAGICS = [
        '%cash_on', '%cash_off', '%cash_debug',
        '%cash_verify', '%cash_repair',
        '%cash_stats', '%cash_export', '%cash_import',
        '%load_ext cash',
    ]

    strip_badges = True
    strip_debug = True
    strip_magics = False  # Off by default - user may want to show magics in docs

    def preprocess_cell(self, cell, resources, index) -> tuple:
        """Process a single cell."""
        if cell.cell_type != 'code':
            return cell, resources

        # Strip badge HTML outputs
        if self.strip_badges:
            cell.outputs = [
                output for output in cell.outputs
                if not self._is_badge_output(output)
            ]

        # Strip debug lines from stream outputs
        if self.strip_debug:
            for output in cell.outputs:
                if output.get('output_type') == 'stream':
                    output['text'] = self._filter_debug_lines(output.get('text', ''))

        # Strip cash magic commands from source
        if self.strip_magics:
            cell.source = self._strip_magic_commands(cell.source)

        return cell, resources

    def _is_badge_output(self, output) -> bool:
        """Check if an output is a cash badge (HTML display)."""
        if output.get('output_type') in ('display_data', 'execute_result'):
            data = output.get('data', {})
            html = data.get('text/html', '')
            if 'cash-badge' in html or CacheStatus.COMPUTED.value in html or CacheStatus.RESTORED.value in html or CacheStatus.SKIPPED.value in html:
                return True
            text = data.get('text/plain', '')
            if '<IPython.core.display.HTML object>' in text:
                return True
        return False

    def _filter_debug_lines(self, text: str) -> str:
        """Remove debug output lines."""
        lines = text.split('\n')
        filtered = [
            line for line in lines
            if not any(marker in line for marker in self.DEBUG_MARKERS)
        ]
        return '\n'.join(filtered)

    def _strip_magic_commands(self, source: str) -> str:
        """Remove cash magic commands from cell source."""
        lines = source.split('\n')
        filtered = [
            line for line in lines
            if not any(line.strip().startswith(magic) for magic in self.CASH_MAGICS)
        ]
        # Remove leading blank lines
        while filtered and not filtered[0].strip():
            filtered.pop(0)
        return '\n'.join(filtered)
