"""`docs/warnings.md` is the lookup target for every warning code.

A code in a message with no section here is a dead link in someone's terminal,
so the page's structure is pinned rather than trusted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from cash.diagnostics import DIAGNOSTIC_CODES
from cash.exceptions import CashWarning

PAGE = Path(__file__).resolve().parents[2] / "docs" / "warnings.md"
#: A code's own section: an H3 under its family's H2.
SECTION = re.compile(r"^### ([A-Z][A-Z-]+) \{#([a-z][a-z-]+)\}$", re.M)
#: A family: an H2 whose sections all share one prefix.
FAMILY = re.compile(r"^## .+ \{#([a-z]+)-codes\}$", re.M)
#: A row of a family's table of codes: ``| [CODE](#code) | path | meaning |``.
INDEX_ROW = re.compile(r"^\| \[([A-Z][A-Z-]+)\]\(#([a-z-]+)\) \| (decorator|notebook|both) \| .+ \|$", re.M)
REQUIRED = (
    "**What happened.**",
    "**Why it matters.**",
    "**What to do.**",
    "**When it is safe to ignore.**",
)


def documented_codes() -> dict[str, str]:
    """Code -> anchor, for every section on the page."""
    return {m.group(1): m.group(2) for m in SECTION.finditer(PAGE.read_text("utf-8"))}


def test_the_page_has_sections():
    assert len(documented_codes()) >= 12


@pytest.mark.parametrize("code", sorted(documented_codes()))
def test_every_anchor_is_the_lowercased_code(code):
    """`doc_url` builds the anchor by lowercasing, so a hand-written anchor
    that differs produces a link to nowhere."""
    assert documented_codes()[code] == code.lower()


def test_every_section_answers_all_four_questions():
    text = PAGE.read_text("utf-8")
    bodies = [b for b in re.split(r"\n#{2,3} ", text)[1:] if SECTION.match("### " + b)]
    assert len(bodies) == len(documented_codes())
    missing = [(body.split(" ")[0], heading) for body in bodies for heading in REQUIRED if heading not in body]
    assert not missing, f"sections missing required headings: {missing}"


def _families() -> dict[str, str]:
    """Family prefix -> the text of that family's H2 section."""
    text = PAGE.read_text("utf-8")
    heads = list(FAMILY.finditer(text))
    return {
        m.group(1).upper(): text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        for i, m in enumerate(heads)
    }


def test_every_section_has_an_index_row_in_its_family():
    """A code with a section but no row cannot be found by scanning the
    tables, and a row in the wrong family sends the reader to the wrong part
    of the page."""
    families = _families()
    assert families, "the page has no family sections"
    missing = []
    for code, anchor in documented_codes().items():
        family = families.get(code.split("-")[0], "")
        rows = {m.group(1): m.group(2) for m in INDEX_ROW.finditer(family)}
        if rows.get(code) != anchor:
            missing.append(code)
    assert not missing, f"sections with no row in their family's table: {missing}"


def test_every_index_row_has_a_section():
    text = PAGE.read_text("utf-8")
    rows = {m.group(1) for m in INDEX_ROW.finditer(text)}
    orphaned = sorted(rows - set(documented_codes()))
    assert not orphaned, f"index rows with no section: {orphaned}"


def test_every_section_sits_in_its_own_family():
    """A code's H3 lives under the H2 of its prefix, so the table of contents
    groups it where a reader scanning by prefix looks for it."""
    placed = [(prefix, code) for prefix, body in _families().items() for code, _ in SECTION.findall(body)]
    assert len(placed) == len(documented_codes()), "some sections sit outside every family"
    misplaced = [code for prefix, code in placed if code.split("-")[0] != prefix]
    assert not misplaced, f"sections under the wrong family: {misplaced}"


def test_every_registered_code_has_a_section():
    """The load-bearing half: a code with no section is a dead link."""
    undocumented = sorted(DIAGNOSTIC_CODES - set(documented_codes()))
    assert not undocumented, f"registered but not documented: {undocumented}"


def test_every_section_is_a_registered_code():
    """The other half: a section for a code nothing emits is dead weight that
    reads as a feature. Retiring a code means leaving a stub that points at its
    replacement -- and removing it from the registry, which this catches."""
    orphaned = sorted(set(documented_codes()) - DIAGNOSTIC_CODES)
    assert not orphaned, f"documented but not registered: {orphaned}"


SRC = Path(__file__).resolve().parents[2] / "src" / "cash"

#: The three names the derivation below MUST find, spanning all three ways a
#: Cash warning class can reach it: the root, a class defined in
#: ``exceptions.py`` beside it, and one defined in a notebook submodule far
#: from it. Asserted separately so a derivation that quietly returns nothing
#: fails loudly instead of making ``_raw_cash_warns`` vacuously empty.
_DERIVATION_MUST_FIND = frozenset(
    {
        "CashWarning",
        "CashCacheIneffectiveWarning",
        "CashNotebookDiscoveryWarning",
    }
)


def _class_defs() -> list[tuple[str, set[str]]]:
    """``(class name, base names)`` for every class defined under ``src/cash``."""
    defs: list[tuple[str, set[str]]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            bases |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
            defs.append((node.name, bases))
    return defs


def cash_warning_categories() -> set[str]:
    """Every ``CashWarning`` subclass name, DERIVED rather than hand-listed.

    The previous version of this was a literal seven-name set with nothing
    asserting it stayed complete: add a ``CashXWarning`` next month, emit it
    with a raw ``warnings.warn``, and the totality test below passes without
    ever looking at it. A test that is silently not total is worse than no
    test, because it gets cited as proof.

    Two passes, because neither is sufficient alone:

    * the **runtime** closure over ``CashWarning.__subclasses__()`` catches
      anything ``import cash`` pulls in, including a class defined dynamically;
    * the **static** closure over every ``class`` statement under ``src/cash``
      catches one defined in a module that ``import cash`` does not reach --
      a backend whose client library is not installed, say -- which the
      runtime walk cannot see.
      Iterated to a fixed point, so a subclass whose file happens to be read
      before its base's is still found.
    """
    names = set()
    stack = [CashWarning]
    while stack:
        cls = stack.pop()
        if cls.__name__ in names:
            continue
        names.add(cls.__name__)
        stack.extend(cls.__subclasses__())

    defs = _class_defs()
    grew = True
    while grew:
        grew = False
        for name, bases in defs:
            if name not in names and bases & names:
                names.add(name)
                grew = True
    return names


def test_the_category_set_is_derived_and_not_empty():
    """The guard on the guard. ``_raw_cash_warns`` finds nothing when the
    category set is empty, so an unnoticed break in the derivation would read
    as 'every site is migrated' rather than as a broken test."""
    categories = cash_warning_categories()
    missing = sorted(_DERIVATION_MUST_FIND - categories)
    assert not missing, f"the CashWarning derivation missed: {missing}"
    assert len(categories) >= len(_DERIVATION_MUST_FIND) + 3


def _raw_cash_warns() -> list[str]:
    """`warnings.warn*(...)` calls naming a Cash category, outside diagnostics.py.

    ``warn_explicit`` is matched as well as ``warn``. The notebook warnings that
    blame a cell line (in ``tracking/randomness/`` and
    ``notebook/upstream/checker.py``) once called it directly, and a matcher
    checking only ``attr == "warn"`` passed while every one of them was
    unmigrated.

    Both call shapes are matched: ``warnings.warn(...)`` (an ``ast.Attribute``)
    and a bare ``warn(...)`` from ``from warnings import warn`` (an
    ``ast.Name``). Nothing in ``src/`` imports it that way today -- which is
    exactly why the hole was invisible, and exactly how cheap it is to close.
    """
    categories = cash_warning_categories()
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "diagnostics.py":
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                called = func.attr
            elif isinstance(func, ast.Name):
                called = func.id
            else:
                continue
            if called not in {"warn", "warn_explicit"}:
                continue
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & categories:
                found.append(f"{path.name}:{node.lineno}")
    return found


def test_no_cash_warning_is_emitted_without_a_code():
    """Without this, the next warning added silently has no code and the
    bijection test still passes -- it only checks codes that exist."""
    assert not _raw_cash_warns(), (
        f"these emit a Cash warning directly; route them through warn_diagnostic instead: {_raw_cash_warns()}"
    )
