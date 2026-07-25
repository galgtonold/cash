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
from dataclasses import fields
from pathlib import Path

from markdown.extensions.toc import slugify

from cash.config import CashConfig, TierConfig

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
# ``superpowers/`` holds internal planning/spec docs that are not part of the
# published site (and are riddled with intentionally-stale links), so they are
# out of scope — same blacklist the fence harness uses.
_BLACKLIST_DIRS = {"superpowers"}
ALL_MD = sorted(
    p
    for p in DOCS_ROOT.rglob("*.md")
    if not any(part in _BLACKLIST_DIRS for part in p.relative_to(DOCS_ROOT).parts)
)


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


def _strip_code_fences(text: str) -> str:
    """Blank out fenced code blocks so their contents don't parse as headings."""
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            out.append("")
            continue
        if in_fence:
            out.append("")
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        out.append(line)
    return "\n".join(out)


def _anchor_ids(text: str) -> set[str]:
    """The set of #anchors a rendered page would expose (as mkdocs/toc slugs)."""
    ids: set[str] = set()
    body = _strip_code_fences(text)

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
        body = _strip_code_fences(text)
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
    lines = _strip_code_fences(text).splitlines()
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
