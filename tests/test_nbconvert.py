"""Tests for cash.nbconvert module."""
import pytest
from cash.nbconvert import CashStripPreprocessor


@pytest.fixture
def preprocessor():
    return CashStripPreprocessor()


def _make_cell(source, outputs=None, cell_type='code'):
    """Create a mock cell dict."""
    from types import SimpleNamespace
    cell = SimpleNamespace()
    cell.cell_type = cell_type
    cell.source = source
    cell.outputs = outputs or []
    return cell


def _make_output(output_type, data=None, text=None):
    """Create a mock output dict."""
    output = {'output_type': output_type}
    if data:
        output['data'] = data
    if text:
        output['text'] = text
    return output


class TestCashStripPreprocessor:
    """Test the nbconvert preprocessor."""

    def test_skip_markdown_cells(self, preprocessor):
        cell = _make_cell("# Title", cell_type='markdown')
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.source == "# Title"

    def test_strip_badge_outputs(self, preprocessor):
        outputs = [
            _make_output('display_data', data={
                'text/html': '<div class="cash-badge">COMPUTED</div>',
                'text/plain': '<IPython.core.display.HTML object>'
            }),
            _make_output('stream', text='Hello world\n'),
        ]
        cell = _make_cell("x = 1", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert len(result.outputs) == 1
        assert result.outputs[0]['text'] == 'Hello world\n'

    def test_strip_debug_lines(self, preprocessor):
        outputs = [
            _make_output('stream', text='Result: 42\n[UPSTREAM_DEBUG] checking...\nDone\n'),
        ]
        cell = _make_cell("x = 42", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert '[UPSTREAM_DEBUG]' not in result.outputs[0]['text']
        assert 'Result: 42' in result.outputs[0]['text']
        assert 'Done' in result.outputs[0]['text']

    def test_strip_magic_commands(self, preprocessor):
        preprocessor.strip_magics = True
        cell = _make_cell("%cash_on\n%cash_debug on\nx = 42\nprint(x)")
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert '%cash_on' not in result.source
        assert '%cash_debug' not in result.source
        assert 'x = 42' in result.source

    def test_preserve_non_badge_html(self, preprocessor):
        outputs = [
            _make_output('display_data', data={
                'text/html': '<table><tr><td>Data</td></tr></table>',
                'text/plain': 'some data'
            }),
        ]
        cell = _make_cell("df.head()", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert len(result.outputs) == 1

    def test_empty_cell(self, preprocessor):
        cell = _make_cell("", [])
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.source == ""

    def test_no_strip_magics_by_default(self, preprocessor):
        cell = _make_cell("%cash_on\nx = 42")
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert '%cash_on' in result.source  # Not stripped by default
