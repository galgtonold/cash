"""Ground doc *claims* against source of truth (the P3 grounding layer).

The fence-execution harness (``test_tutorials.py`` / ``_harness.py``) only runs
```python fences and asserts opt-in hit/miss counts — it is blind to prose,
tables, and links. These checks fill part of that gap by verifying the
structured, mechanically-checkable claims that tend to drift:

* every ``CASH_*`` env var named in the docs is a real config binding;
* every internal Markdown link resolves to a file, and its ``#anchor`` exists.

They read the docs as text and compare against the code, so a copy-pasteable
error — a wrong env-var name, a dead ``#anchor`` — fails CI instead of shipping.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import fields
from pathlib import Path

from cash.config import CashConfig, TierConfig
from tests.docs._claims import published_pages, strip_code_fences


def slugify(value: str, separator: str = "-") -> str:
    """Vendored copy of ``markdown.extensions.toc.slugify`` (default, non-unicode).

    Reproduced here so this module has no dependency on ``markdown`` — the CI
    ``docs-parity`` job doesn't install mkdocs. mkdocs.yml configures ``toc``
    with no custom ``slugify``, so it uses exactly this default with ``-``.
    """
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(rf"[{re.escape(separator)}\s]+", separator, value)

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
# The set of pages mkdocs actually builds -- ``superpowers/`` internal
# planning docs and mkdocs.yml's ``exclude_docs`` entries are out of scope.
# ``_claims.py::published_pages`` is the single definition; this used to be a
# byte-equivalent second copy of the same comprehension, which is exactly the
# kind of duplication an anti-drift mechanism shouldn't itself have.
ALL_MD = published_pages()


def _is_autodoc(path: Path) -> bool:
    """True if the page renders member anchors via mkdocstrings (``::: cash.X``).

    Those anchors (``#cash.Cash``, ``#cash.CacheExplanation``, …) are generated
    at build time and are invisible in the source, so we can't verify a link's
    ``#anchor`` into such a page — only that the file exists.
    """
    return any(
        line.startswith(":::") for line in path.read_text(encoding="utf-8").splitlines()
    )


# --------------------------------------------------------------------------- #
# CASH_* environment variables                                                #
# --------------------------------------------------------------------------- #

# ``_load_env_config`` binds ``CASH_<FIELD>`` for each CashConfig field and
# ``CASH_TIER_<N>_<FIELD>`` for each TierConfig field; every other CASH_* token
# is silently ignored at runtime, so one in the docs is a doc bug.
_ENV_TOKEN_RE = re.compile(r"\bCASH_[A-Z0-9_]+\b")
_TIER_TOKEN_RE = re.compile(r"^CASH_TIER_\d+_(?P<field>[A-Z0-9_]+)$")

# Real env vars that are NOT plain CashConfig fields (documented, read elsewhere,
# or placeholders that appear in illustrative env-var tables). Keep this small
# and justified — it is the escape hatch, not a silencer.
_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CASH_TIER_N_FIELD",  # the literal pattern placeholder in docs
        "CASH_TIER_",  # captured prefix of the ``CASH_TIER_<N>_FIELD`` pattern
    }
)


def _valid_top_level_env_vars() -> set[str]:
    return {
        f"CASH_{f.name.upper()}"
        for f in fields(CashConfig)
        if not f.name.startswith("_")
    }


def test_env_vars_in_docs_are_real() -> None:
    valid = _valid_top_level_env_vars()
    tier_fields = {f.name.upper() for f in fields(TierConfig)}
    problems: list[str] = []

    for md in ALL_MD:
        text = md.read_text(encoding="utf-8")
        for tok in sorted(set(_ENV_TOKEN_RE.findall(text))):
            if tok in valid or tok in _ENV_ALLOWLIST:
                continue
            m = _TIER_TOKEN_RE.match(tok)
            if m and m.group("field") in tier_fields:
                continue
            problems.append(f"  {md.relative_to(DOCS_ROOT).as_posix()}: {tok}")

    assert not problems, (
        "Docs reference CASH_* env vars with no matching CashConfig/TierConfig "
        "binding (they would be silently ignored at runtime):\n" + "\n".join(problems)
    )


# --------------------------------------------------------------------------- #
# Internal links and #anchors                                                 #
# --------------------------------------------------------------------------- #

# Markdown links: ``[label](target "title")`` — capture target, skip images.
_LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(\s*(?P<target>[^)\s]+)")
# Explicit heading ids via attr_list: ``### Foo { #custom-id }``.
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*$", re.MULTILINE)
_ATTR_ID_RE = re.compile(r"\{[^}]*#(?P<id>[\w-]+)[^}]*\}\s*$")
# Raw-HTML anchors (iframes, divs, <a id=...>).
_HTML_ID_RE = re.compile(r"""\sid=["'](?P<id>[^"']+)["']""")
# Inline markdown link inside heading text -> keep the label only.
_INLINE_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Fenced code block, so ``# comment`` lines inside aren't read as headings.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<ticks>`{3,}|~{3,})", re.MULTILINE)


def _anchor_ids(text: str) -> set[str]:
    """The set of #anchors a rendered page would expose (as mkdocs/toc slugs)."""
    ids: set[str] = set()
    body = strip_code_fences(text)

    for m in _HEADING_RE.finditer(body):
        heading = m.group("text")
        attr = _ATTR_ID_RE.search(heading)
        if attr:
            ids.add(attr.group("id"))
            heading = _ATTR_ID_RE.sub("", heading).strip()
        # Links in headings contribute only their label to the slug.
        heading = _INLINE_LINK_RE.sub(r"\1", heading)
        ids.add(slugify(heading, "-"))

    # Raw-HTML ids come from the full text (may live inside md_in_html blocks).
    for m in _HTML_ID_RE.finditer(text):
        ids.add(m.group("id"))
    return ids


def _iter_internal_links() -> list[tuple[Path, str, str | None]]:
    """Yield (source_md, target_path_or_empty, anchor_or_None) for each
    internal Markdown link across the docs."""
    out: list[tuple[Path, str, str | None]] = []
    for md in ALL_MD:
        text = md.read_text(encoding="utf-8")
        body = strip_code_fences(text)
        for m in _LINK_RE.finditer(body):
            target = m.group("target")
            if target.startswith(("http://", "https://", "mailto:", "//", "www.")):
                continue
            path_part, _, anchor = target.partition("#")
            anchor = anchor or None
            # Same-page anchor link.
            if not path_part:
                out.append((md, "", anchor))
                continue
            # Only resolve links that point at Markdown pages.
            if not path_part.endswith(".md"):
                continue
            out.append((md, path_part, anchor))
    return out


def test_internal_anchors_resolve() -> None:
    id_cache: dict[Path, set[str]] = {}

    def ids_for(p: Path) -> set[str]:
        if p not in id_cache:
            id_cache[p] = _anchor_ids(p.read_text(encoding="utf-8"))
        return id_cache[p]

    problems: list[str] = []
    for src, path_part, anchor in _iter_internal_links():
        src_rel = src.relative_to(DOCS_ROOT).as_posix()
        if path_part:
            target = (src.parent / path_part).resolve()
            if not target.exists():
                problems.append(f"  {src_rel}: link to missing file '{path_part}'")
                continue
            # Anchors into a mkdocstrings page are generated at build time and
            # unknowable from source — file existence is all we can verify.
            if anchor and _is_autodoc(target):
                continue
        else:
            target = src
        if anchor and anchor not in ids_for(target):
            tgt_rel = target.relative_to(DOCS_ROOT).as_posix() if target != src else src_rel
            problems.append(f"  {src_rel}: '#{anchor}' not found in {tgt_rel}")

    assert not problems, "Broken internal links / anchors in docs:\n" + "\n".join(problems)


# --------------------------------------------------------------------------- #
# Config default values quoted in docs                                        #
# --------------------------------------------------------------------------- #

def _config_defaults() -> dict[str, object]:
    import dataclasses

    out: dict[str, object] = {}
    for f in fields(CashConfig):
        if f.name.startswith("_"):
            continue
        if f.default is dataclasses.MISSING:  # factory fields (tiers) — skip
            continue
        out[f.name] = f.default
    return out


_CODE_SPAN_RE = re.compile(r"`([^`]+)`")


def _parse_doc_default(cell: str):
    """Best-effort parse of a doc 'Default' cell to a Python value.

    Returns (value, ok). ``ok`` is False when the cell isn't a simple literal
    we can compare (so the row is skipped rather than falsely failing).
    """
    cell = cell.strip()
    cell = re.sub(r"\([^)]*\)", "", cell)      # drop "(auto)" / "(unlimited)"
    cell = cell.replace("**", "").strip()
    m = _CODE_SPAN_RE.search(cell)
    token = (m.group(1) if m else cell).strip()
    # Strip trailing unit words the docs add for readability.
    token = re.sub(r"\s*(seconds|sec|s|bytes|b)$", "", token, flags=re.IGNORECASE).strip()
    low = token.lower()
    if low in {"null", "none"}:
        return None, True
    if low == "true":
        return True, True
    if low == "false":
        return False, True
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1], True
    try:
        return int(token), True
    except ValueError:
        pass
    try:
        return float(token), True
    except ValueError:
        pass
    return None, False


def _tables(text: str):
    """Yield (header_cells, rows) for each pipe table in the markdown text."""
    lines = strip_code_fences(text).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            yield header, rows
            i = j
        else:
            i += 1


def test_config_defaults_in_docs_match_source() -> None:
    defaults = _config_defaults()
    problems: list[str] = []

    for md in ALL_MD:
        text = md.read_text(encoding="utf-8")
        rel = md.relative_to(DOCS_ROOT).as_posix()
        for header, rows in _tables(text):
            lower = [h.lower() for h in header]
            if "field" not in lower or "default" not in lower:
                continue
            fi, di = lower.index("field"), lower.index("default")
            for row in rows:
                if len(row) <= max(fi, di):
                    continue
                fm = _CODE_SPAN_RE.search(row[fi])
                if not fm:
                    continue
                field = fm.group(1).strip()
                if field not in defaults:
                    continue
                want = defaults[field]
                got, ok = _parse_doc_default(row[di])
                if not ok:
                    continue
                if isinstance(want, float) or isinstance(got, float):
                    match = want is not None and got is not None and float(want) == float(got)
                else:
                    match = want == got
                if not match:
                    problems.append(
                        f"  {rel}: `{field}` default documented as {row[di].strip()!r} "
                        f"but CashConfig default is {want!r}"
                    )

    assert not problems, "Documented config defaults disagree with CashConfig:\n" + "\n".join(
        problems
    )


# --------------------------------------------------------------------------- #
# Magics documented vs magics registered                                      #
# --------------------------------------------------------------------------- #
#
# ``docs/magics.md`` calls itself "the canonical reference for all N magics" and
# carries both an at-a-glance table and a detail section per magic. Nothing else
# checks that against the code: a new ``@line_magic`` lands, the page is not
# touched, and the page is now silently a reference to 20 of 21 things while
# still claiming to be canonical. A fingerprint anchor is the wrong instrument
# here -- it would fire on every unrelated edit to magics.py -- so the
# completeness property gets its own structural check.

_MAGICS_PAGE = DOCS_ROOT / "magics.md"
_MAGIC_SOURCES = (
    "notebook/ipython/magics.py",
    "notebook/ipython/admin.py",
)


def _registered_magics() -> set[str]:
    """Every magic name IPython would register, read from the decorators."""
    import ast

    src_root = DOCS_ROOT.parent / "src" / "cash"
    found: set[str] = set()
    for rel in _MAGIC_SOURCES:
        tree = ast.parse((src_root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                if "line_magic" not in text and "cell_magic" not in text:
                    continue
                # ``@line_magic("cash_foo")`` renames; bare ``@line_magic`` does not.
                named = (
                    dec.args[0].value
                    if isinstance(dec, ast.Call)
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    else node.name
                )
                found.add(("%%" if "cell_magic" in text else "%") + named)
    return found


def test_every_registered_magic_is_documented() -> None:
    doc = _MAGICS_PAGE.read_text(encoding="utf-8")
    registered = _registered_magics()
    assert registered, "found no magics in source -- the decorator scan broke"

    table = set(re.findall(r"^\| \[`(%%?[a-z_]+)`\]", doc, re.M))
    headings = set(re.findall(r"^### `(%%?[a-z_]+)", doc, re.M))

    problems = []
    for label, documented in (("at-a-glance table", table), ("detail section", headings)):
        for name in sorted(registered - documented):
            problems.append(f"  {name} is registered in source but has no {label}")
        for name in sorted(documented - registered):
            problems.append(f"  {name} has a {label} but is not registered in source")
    assert not problems, "docs/magics.md is out of sync with the registered magics:\n" + "\n".join(
        problems
    )


def test_magics_page_states_the_right_count() -> None:
    """The page says 'all **N** magics'. N is a claim, so check it."""
    doc = _MAGICS_PAGE.read_text(encoding="utf-8")
    m = re.search(r"canonical\s+reference\s+for\s+all\s+\*\*(\d+)\*\*\s+magics", doc)
    assert m, "could not find the 'all **N** magics' claim -- did the wording change?"
    claimed, actual = int(m.group(1)), len(_registered_magics())
    assert claimed == actual, (
        f"docs/magics.md claims {claimed} magics; source registers {actual}"
    )


# --------------------------------------------------------------------------- #
# Line-pinned source references                                               #
# --------------------------------------------------------------------------- #
#
# A reference like ``annotations.py:85-86`` rots on ANY edit above line 85, and
# nothing notices. docs/annotations.md carried sixteen of them and every single
# one was wrong -- pointing at a docstring, at `result = CacheAnnotation()`, at
# a matplotlib comment block, and in one case at a test that verified something
# else entirely. They did not drift one at a time; they drifted together, and
# silently, because a stale line number still looks authoritative.
#
# Renumbering only resets the clock. The fix is to name the SYMBOL, which moves
# with the code (and, for a load-bearing claim, to add a claim anchor so the
# fingerprint check re-verifies it). This is a burn-down ratchet, not a gate:
# the counts below may fall, never rise.
#
# CAS-126 filed this idea; annotations.md is the proof it was worth doing.

# Both separators, because the docs used both. thread-safety.md carried
# ``core.py,526`` and ``core.py,526,1503-1504`` -- comma-delimited, so a
# colon-only pattern scored that page 1 when it really had 3, and the two it
# missed were BOTH rotted (526 landed on a comment about backends; 1503-1504
# inside _explain_call). A ratchet that under-counts reports progress it did
# not make, so match ``.py`` followed by any digit/comma/hyphen run.
#
# Third form, found later in choosing-a-backend.md: the path and the line
# number in SEPARATE code spans -- ``(`…/memory_backend.py`, `:210-221`)``.
# A pattern anchored on ``.py`` cannot see it, so the second alternative
# matches a bare ``:NNN`` / ``:NNN-MMM`` span, which has no other use in prose.
_LINE_PIN_RE = re.compile(r"`[\w./-]+\.py[:,]\d[\d,-]*`|`:\d+(?:-\d+)?`")

# page -> number of line-pinned refs still present. Burn these down; do not add.
#
# EMPTY as of the 2026-07-27 audit: every line-pinned reference in the published
# docs has been replaced with a symbol reference. It started at 22 across five
# pages and all 22 were checked -- 20 of them had rotted, most by the same
# ~68-line shift when something was inserted above them. Keeping the dict (rather
# than deleting the mechanism) means a single new pin fails the build.
_KNOWN_LINE_PINS: dict[str, int] = {}


def _line_pin_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for md in ALL_MD:
        n = len(_LINE_PIN_RE.findall(md.read_text(encoding="utf-8")))
        if n:
            out[md.relative_to(DOCS_ROOT).as_posix()] = n
    return out


def test_no_new_line_pinned_source_references() -> None:
    actual = _line_pin_counts()
    problems: list[str] = []
    for page, count in sorted(actual.items()):
        allowed = _KNOWN_LINE_PINS.get(page, 0)
        if count > allowed:
            problems.append(
                f"  {page}: {count} line-pinned refs (allowed {allowed}). "
                f"Reference the symbol instead of the line -- line numbers rot "
                f"on any edit above them and nothing catches it."
            )
    assert not problems, (
        "New line-pinned source references in the docs:\n" + "\n".join(problems)
    )


def test_line_pin_ratchet_has_no_stale_entries() -> None:
    """A page that was fixed must leave the list, or the ratchet stops biting."""
    actual = _line_pin_counts()
    stale: list[str] = []
    for page, allowed in sorted(_KNOWN_LINE_PINS.items()):
        found = actual.get(page, 0)
        if found < allowed:
            stale.append(
                f"  {page}: listed as {allowed} but only {found} remain -- "
                f"lower the number (or delete the entry at 0)"
            )
    assert not stale, "Line-pin ratchet is out of date:\n" + "\n".join(stale)


# --------------------------------------------------------------------------- #
# Cited test names                                                            #
# --------------------------------------------------------------------------- #
#
# "Test reference: ``test_foo``" is the strongest citation a doc page can make:
# it says a claim is not just true but *guarded*. It is also the citation that
# rots most quietly -- renaming or deleting the test leaves the sentence intact
# and still authoritative-sounding.
#
# iterator-caching.md cited eight tests by name AND line; the names were all
# real but six of the eight line numbers had drifted by ~68 lines. Dropping the
# lines is only half the fix -- without this check, a later rename would put the
# page right back where it started, with no line number left to look wrong.

_TEST_NAME_RE = re.compile(r"`(test_[a-z0-9_]+)`")
_TESTS_ROOT = Path(__file__).resolve().parents[1]


def _defined_test_names() -> set[str]:
    names: set[str] = set()
    for py in _TESTS_ROOT.rglob("test_*.py"):
        # ``async def`` too -- omitting it reported all six of async-caching.md's
        # citations as dead when every one of them exists.
        names.update(
            re.findall(
                r"^\s*(?:async\s+)?def (test_[a-z0-9_]+)",
                py.read_text(encoding="utf-8"),
                re.M,
            )
        )
    return names


def test_every_cited_test_name_exists() -> None:
    defined = _defined_test_names()
    missing: list[str] = []
    for md in ALL_MD:
        text = strip_code_fences(md.read_text(encoding="utf-8"))
        for name in sorted(set(_TEST_NAME_RE.findall(text))):
            if name not in defined:
                missing.append(f"  {md.relative_to(DOCS_ROOT).as_posix()}: `{name}`")
    assert not missing, (
        "Docs cite test names that no longer exist:\n" + "\n".join(missing)
        + "\n\nA 'Test reference:' claiming a guard that isn't there is worse "
        "than no citation. Rename the reference or drop it."
    )


# --------------------------------------------------------------------------- #
# Annotation directives                                                       #
# --------------------------------------------------------------------------- #
#
# The same completeness argument as the magics gate: a page that says "that's
# the whole language" is making a claim about a SET, and a set claim goes stale
# the moment the set grows. controlling-cache-behavior.md -- titled "this guide
# covers every knob" -- never mentioned `# @cash:cache-calls` after it shipped.
#
# Canonical names only (the `nocache` / `cachefit` run-together spellings are
# aliases the parser accepts, not directives a page has to advertise).

_DIRECTIVE_RE = re.compile(r"directive == '([a-z][a-z-]+)'")
_ANNOTATIONS_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "cash" / "notebook" / "annotations.py"
)
# Pages that claim to cover the directive set, and so must cover all of it.
_DIRECTIVE_PAGES = (
    "annotations.md",
    "tutorials/feature-guides/controlling-cache-behavior.md",
)


def _parsed_directives() -> set[str]:
    src = _ANNOTATIONS_SRC.read_text(encoding="utf-8")
    found = set(_DIRECTIVE_RE.findall(src))
    # Aliases are the run-together spelling of a canonical name already found.
    return {d for d in found if "-" in d or not any(d == c.replace("-", "") for c in found)}


def test_every_annotation_directive_is_documented() -> None:
    directives = _parsed_directives()
    assert directives, "parsed no directives -- the scan pattern has drifted"
    problems: list[str] = []
    for page in _DIRECTIVE_PAGES:
        text = (DOCS_ROOT / page).read_text(encoding="utf-8")
        for d in sorted(directives):
            if f"@cash:{d}" not in text:
                problems.append(f"  {page}: never mentions `# @cash:{d}`")
    assert not problems, (
        "Annotation directives the parser accepts but the docs don't cover:\n"
        + "\n".join(problems)
        + "\n\nThese pages present themselves as complete; a missing directive "
        "is a feature users cannot discover."
    )
