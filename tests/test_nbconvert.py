"""Tests for cash.nbconvert module."""

import pytest

from cash.nbconvert import CashStripPreprocessor


@pytest.fixture
def preprocessor():
    return CashStripPreprocessor()


def _make_cell(source, outputs=None, cell_type="code"):
    """Create a mock cell dict."""
    from types import SimpleNamespace

    cell = SimpleNamespace()
    cell.cell_type = cell_type
    cell.source = source
    cell.outputs = outputs or []
    return cell


def _make_output(output_type, data=None, text=None):
    """Create a mock output dict."""
    output = {"output_type": output_type}
    if data:
        output["data"] = data
    if text:
        output["text"] = text
    return output


class TestCashStripPreprocessor:
    """Test the nbconvert preprocessor."""

    def test_skip_markdown_cells(self, preprocessor):
        cell = _make_cell("# Title", cell_type="markdown")
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.source == "# Title"

    @pytest.mark.parametrize("status", ["DONE", "RUNNING"])
    def test_strip_badge_outputs(self, preprocessor, status):
        """The badge cash actually renders, done or in progress, is removed."""
        from cash.notebook.badge_renderer._badge import render_interactive_badge

        badge = render_interactive_badge(
            [{"status": "COMPUTED", "code": "x = 1", "execution_time": 0.2}],
            "html",
            status=status,
            current_step=1,
            total_steps=2,
        )
        assert badge, "the renderer produced no badge"
        outputs = [
            _make_output(
                "display_data",
                data={"text/html": badge, "text/plain": "<IPython.core.display.HTML object>"},
            ),
            _make_output("stream", text="Hello world\n"),
        ]
        cell = _make_cell("x = 1", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert len(result.outputs) == 1
        assert result.outputs[0]["text"] == "Hello world\n"

    def test_strip_debug_lines(self, preprocessor):
        outputs = [
            _make_output("stream", text="Result: 42\n[UPSTREAM_DEBUG] checking...\nDone\n"),
        ]
        cell = _make_cell("x = 42", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert "[UPSTREAM_DEBUG]" not in result.outputs[0]["text"]
        assert "Result: 42" in result.outputs[0]["text"]
        assert "Done" in result.outputs[0]["text"]

    def test_strip_magic_commands(self, preprocessor):
        preprocessor.strip_magics = True
        cell = _make_cell("%cash_on\n%cash_debug on\nx = 42\nprint(x)")
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert "%cash_on" not in result.source
        assert "%cash_debug" not in result.source
        assert "x = 42" in result.source

    def test_a_users_own_html_display_is_kept(self, preprocessor):
        """``display(HTML(...))`` from any library has this text/plain; only
        the badge markup marks a badge."""
        outputs = [
            _make_output(
                "display_data",
                data={
                    "text/html": "<b>Model trained: COMPUTED 3 folds</b>",
                    "text/plain": "<IPython.core.display.HTML object>",
                },
            ),
        ]
        cell = _make_cell("display(HTML(summary))", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert len(result.outputs) == 1

    def test_a_users_printed_lines_are_kept(self, preprocessor):
        text = "DEBUG mode is on\nCash: 1,200 EUR\nlevel=DEBUG\ncash: caching disabled\n"
        outputs = [_make_output("stream", text=text)]
        cell = _make_cell("report()", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.outputs[0]["text"] == text

    def test_cash_log_records_are_stripped(self, preprocessor):
        text = "before\ncash.core: [CACHE] miss for f\n[cash.notebook.magics] debug on\n[TIMING_PROXY] Badge init: 1.0ms\nafter\n"
        outputs = [_make_output("stream", text=text)]
        cell = _make_cell("f()", outputs)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.outputs[0]["text"] == "before\nafter\n"

    def test_every_registered_magic_is_stripped(self, preprocessor):
        from cash.notebook.ipython.magics import CashMagics

        preprocessor.strip_magics = True
        names = sorted(CashMagics.magics["line"])
        assert "cash_badge" in names  # positive control: a magic the old hand-kept list lacked
        source = (
            "\n".join(f"%{name} on" for name in names) + "\n%load_ext cash\nx = 1\n%cash_onward = 2\n%% not_a_magic"
        )
        cell = _make_cell(source)
        result, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result.source == "x = 1\n%cash_onward = 2\n%% not_a_magic"

    def test_preserve_non_badge_html(self, preprocessor):
        outputs = [
            _make_output(
                "display_data", data={"text/html": "<table><tr><td>Data</td></tr></table>", "text/plain": "some data"}
            ),
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
        assert "%cash_on" in result.source  # Not stripped by default


def test_the_documented_options_are_constructor_arguments():
    """The docs construct it as ``CashStripPreprocessor(strip_magics=True)``;
    the options were plain class attributes, which nbconvert's base class
    does not take as arguments."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preprocessor = CashStripPreprocessor(strip_badges=False, strip_magics=True)
    assert preprocessor.strip_badges is False
    assert preprocessor.strip_magics is True
    assert CashStripPreprocessor().strip_magics is False
