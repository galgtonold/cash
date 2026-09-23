"""nbconvert preprocessors for stripping cache badges and executing with cash."""

from __future__ import annotations

import re
from functools import cache

from .notebook.ipython.magics import CashMagics

try:
    from nbconvert.preprocessors import Preprocessor
    from traitlets import Bool

    HAS_NBCONVERT = True
except ImportError:
    # Provide a dummy base class when nbconvert isn't installed
    HAS_NBCONVERT = False

    class Preprocessor:
        """Dummy base class when nbconvert is not installed."""

        def __init__(self, **options):
            for name, value in options.items():
                if not hasattr(type(self), name):
                    raise TypeError(f"{type(self).__name__} has no option {name!r}")
                setattr(self, name, value)

        def preprocess(self, nb, resources):
            for index, cell in enumerate(nb.cells):
                nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
            return nb, resources

        def preprocess_cell(self, cell, resources, index):
            return cell, resources

    def Bool(default, **_kwargs):  # noqa: N802 - stands in for traitlets.Bool
        return _Plain(default)

    class _Plain:
        """A default value that ``.tag(config=True)`` leaves as it is."""

        def __init__(self, value):
            self.value = value

        def tag(self, **_kwargs):
            return self.value


class CashStripPreprocessor(Preprocessor):
    """
    Preprocessor that strips cash-specific outputs from notebook cells.

    This removes:
    - the cell badges cash displays (HTML outputs carrying the badge markup)
    - cash's debug lines from stream outputs: its log records
      (``cash.<module>: ...`` / ``[cash.<module>] ...``) and its tagged debug
      prints (``[UPSTREAM] ...``, ``[TIMING_PROXY] ...``, ...)
    - optionally, cash magic commands from cell source (``%cash_on``, ...)

    Every other output is left exactly as it was: a user's own HTML, and
    printed lines that merely mention "DEBUG" or "Cash:", are kept.

    Usage:
        jupyter nbconvert --to html \\
            --Exporter.preprocessors='["cash.nbconvert.CashStripPreprocessor"]' \\
            notebook.ipynb
    """

    #: What every badge's HTML contains right after its ``<style>`` block
    #: (``cash.notebook.badge_renderer.renderers.html.render_html``).
    BADGE_MARKUP = '<div class="c3-wrap"><details class="c3-card"'

    #: A line cash's own logging or debug printing produced: a log record
    #: formatted as ``cash.x: msg`` or ``[cash.x] msg`` (a bare ``cash: ...``
    #: line is a summary meant for the reader, not debug output), or a tagged debug
    #: print such as ``[UPSTREAM] ...`` or ``[TIMING_PROXY] ...``.
    DEBUG_LINE = re.compile(
        r"^(?:cash(?:\.\w+)+: |\[cash(?:\.\w+)*\] |\[(?:TIMING|UPSTREAM|LINEAGE|ALREADY|CACHE|CONTROL)(?:_[A-Z_]+)?\] )"
    )

    strip_badges = Bool(True, help="Remove the cell badges cash displays.").tag(config=True)
    strip_debug = Bool(True, help="Remove cash's debug lines from stream outputs.").tag(config=True)
    # Off by default: a user may want to show the magics in docs.
    strip_magics = Bool(False, help="Remove cash magic commands from cell source.").tag(config=True)

    def preprocess_cell(self, cell, resources, index) -> tuple:
        """Process a single cell."""
        if cell.cell_type != "code":
            return cell, resources

        if self.strip_badges:
            cell.outputs = [output for output in cell.outputs if not self._is_badge_output(output)]

        if self.strip_debug:
            for output in cell.outputs:
                if output.get("output_type") == "stream":
                    output["text"] = self._filter_debug_lines(output.get("text", ""))

        if self.strip_magics:
            cell.source = self._strip_magic_commands(cell.source)

        return cell, resources

    def _is_badge_output(self, output) -> bool:
        """Is *output* a cash cell badge?"""
        if output.get("output_type") not in ("display_data", "execute_result"):
            return False
        html = output.get("data", {}).get("text/html", "")
        if isinstance(html, list):  # nbformat may store multi-line strings split
            html = "".join(html)
        return html.startswith("<style>") and self.BADGE_MARKUP in html

    def _filter_debug_lines(self, text: str) -> str:
        """Remove the lines cash's debug output produced."""
        if isinstance(text, list):
            text = "".join(text)
        lines = text.split("\n")
        return "\n".join(line for line in lines if not self.DEBUG_LINE.match(line))

    def _strip_magic_commands(self, source: str) -> str:
        """Remove cash magic commands from cell source."""
        pattern = _magic_line_pattern()
        filtered = [line for line in source.split("\n") if not pattern.match(line)]
        # Remove leading blank lines
        while filtered and not filtered[0].strip():
            filtered.pop(0)
        return "\n".join(filtered)


@cache
def _magic_line_pattern() -> re.Pattern[str]:
    """A source line that invokes a cash magic, built from the registered set."""

    def names(kind: str) -> str:
        return "|".join(sorted(map(re.escape, CashMagics.magics[kind]), key=len, reverse=True))

    # An empty alternation would match a bare ``%``/``%%``, so a kind with no
    # registered magic (cash has no cell magic) contributes nothing.
    forms = [
        f"{prefix}(?:{names(kind)})" for prefix, kind in (("%", "line"), ("%%", "cell")) if CashMagics.magics[kind]
    ]
    forms.append(r"%load_ext\s+cash")
    return re.compile(rf"^\s*(?:{'|'.join(forms)})(?:\s|$)")
