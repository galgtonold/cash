"""A snippet on a reference page brings its own imports.

A reader lands on one section from a search (``ttl=``, ``cache_if=``) and
presses the copy button. The docs run executes a page's blocks in order, so
a block that leaned on the ``import cash`` of the page's first block passed
there and gave the reader ``NameError: name 'cash' is not defined``.

Checked without running anything: each Python block must compile, and every
name it reads must be bound in the block itself, or in an earlier block under
the same heading (a section may build one example in steps). The stand-ins
below are the exception: names the text uses for the reader's own objects.
"""

from __future__ import annotations

import builtins
import re
import symtable
import textwrap
from pathlib import Path

import pytest

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

# Page -> the names that stand for something of the reader's own.
PAGES = {
    "decorator.md": {"db", "cache_backend", "fit_model"},
    "tutorials/feature-guides/custom-hashers.md": {"MyPydanticModel", "DatasetConfig"},
    "tutorials/feature-guides/async-caching.md": {"price_api", "api"},
}

_HEADING = re.compile(r"#{1,6} ")
_OPEN = re.compile(r"\s*```(?P<info>\S.*)?$")
_CLOSE = re.compile(r"\s*```\s*$")


def python_blocks(text: str) -> list[tuple[int, int, str]]:
    """``(section number, line of the fence, source)`` per Python block."""
    blocks, section, opened, info, body = [], 0, 0, None, []
    for number, line in enumerate(text.splitlines(), 1):
        if info is None:
            if _HEADING.match(line):
                section += 1
            found = _OPEN.match(line)
            if found:
                info, opened, body = found.group("info") or "", number, []
        elif _CLOSE.match(line):
            if info.startswith("python"):
                blocks.append((section, opened, textwrap.dedent("\n".join(body))))
            info = None
        else:
            body.append(line)
    return blocks


def bound_and_read(source: str) -> tuple[set[str], set[str]]:
    """The names a block binds at its top level, and the names it reads
    from there (in its own lines or from inside a function or class)."""
    top = symtable.symtable(source, "<block>", "exec")
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    read = {s.get_name() for s in top.get_symbols() if s.is_referenced()}

    def inner(table: symtable.SymbolTable) -> None:
        for child in table.get_children():
            read.update(s.get_name() for s in child.get_symbols() if s.is_global())
            inner(child)

    inner(top)
    return bound, read


def missing_names(text: str, stand_ins: set[str]) -> list[str]:
    problems, section_now, known = [], None, set()
    for section, line, source in python_blocks(text):
        if section != section_now:
            section_now, known = section, set()
        try:
            bound, read = bound_and_read(source)
        except SyntaxError as error:
            problems.append(f"line {line}: does not compile ({error.msg})")
            continue
        known |= bound
        missing = sorted(n for n in read - known - stand_ins if not hasattr(builtins, n))
        if missing:
            problems.append(f"line {line}: nothing in the block binds {', '.join(missing)}")
    return problems


@pytest.mark.parametrize("page", sorted(PAGES))
def test_each_block_binds_the_names_it_reads(page):
    text = (DOCS_ROOT / page).read_text(encoding="utf-8")
    assert python_blocks(text), f"{page} has no Python block: the check would pass on nothing"
    assert missing_names(text, PAGES[page]) == []


def test_a_block_leaning_on_an_earlier_sections_import_is_named():
    page = textwrap.dedent(
        """\
        ## First

        ```python
        import cash

        @cash.cache
        def double(x):
            return x * 2
        ```

        ```python
        double(2)
        ```

        ### Second

        ```python
        @cash.cache(ttl=300)
        def rates():
            return requests.get(URL).json()
        ```
        """
    )
    assert missing_names(page, {"URL"}) == ["line 17: nothing in the block binds cash, requests"]
