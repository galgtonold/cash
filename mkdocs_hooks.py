"""mkdocs build hooks for the Cash docs.

Wired up via ``hooks:`` in ``mkdocs.yml``.  mkdocs imports this file directly
(by path) and calls any ``on_*`` event functions it defines.

Every change here is a build-time rewrite: the Markdown source keeps what the
docs tests (``tests/docs``) read, and the published page shows what a reader
needs.

Badge iframe paths
------------------
The "Cash badge" example iframes are authored with a *root-absolute* ``src``::

    <iframe class="cash-badge" src="/_badges/anatomy_hero.html" ...></iframe>

A root-absolute path only resolves when the site is mounted at the domain root
(local ``mkdocs serve``, GitHub Pages at the apex).  Read the Docs serves each
version under a prefix (``/en/latest/``), so ``/_badges/...`` points at the
domain root and 404s -- the browser then renders RTD's unstyled 404 page inside
every badge frame (the "huge icons / page without CSS" symptom).

mkdocs rewrites Markdown links/images to be page-relative, but it does *not*
touch ``src`` attributes inside raw HTML, so the absolute paths survive into the
built site.  ``on_post_page`` rewrites them at build time to a path relative to
the page being rendered, so they resolve no matter what base path the site is
mounted at -- and with no client-side flash.

Doc-number markers
------------------
``scripts/doc_numbers.py`` keeps ``<!-- docnum:NAME -->`` markers around derived
numbers. In prose they are invisible HTML comments, but inside inline code or a
code block they render literally, so the site would show
``cash-lib<!-- docnum:version_pin -->~=...``. The value between the markers
stays; only the markers go.

Test-harness lines
------------------
A ``# test:inject: <code>`` line in a code block is a step the docs harness runs
(an import, a file edit, a clock jump) that the example does not show. It is
test plumbing, so it is dropped from the page and from what the copy button
copies. The harness reads the source, where the line stays.

The "Applies to" box
--------------------
Pages open with ``!!! info "Applies to: <path>"`` and one line saying who the
page is for. The source keeps the box (the page conventions and their checks
rely on it); the site shows it as a small path chip under the H1, with the
audience line beside it. On the home page it is dropped, and on the Project
pages only the audience line stays: "both paths" tells those readers nothing.

Printed output
--------------
A code block titled ``Output`` (```` ```text title="Output" ````) is what a
program prints, not code to type. It gets the ``cash-output`` class for its own
look in ``cash-design.css`` and ``no-copy``, which drops Material's copy button.
"""

from __future__ import annotations

import re

# Match the root-absolute badge src and split off the leading slash so we can
# splice a page-relative prefix in front of the "_badges/..." remainder.
_ABS_BADGE_SRC = re.compile(r'(<iframe\b[^>]*\bsrc=")/(_badges/[^"]+)(")')

# The opening and closing markers of scripts/doc_numbers.py's MARKER.
_DOCNUM_MARKER = re.compile(r"<!--\s*(?:docnum:[a-z0-9_]+|/docnum)\s*-->")

# The same line tests/docs/_harness.py turns into code, with its line break.
_TEST_INJECT_LINE = re.compile(r"^[ \t]*# test:inject:.*(?:\n|$)", re.MULTILINE)

# The page-opening box: the title line, then its 4-space-indented body up to
# the first line that is neither blank nor indented.
_APPLIES_BOX = re.compile(
    r'^!!! info "Applies to: (?P<path>[^"]+)"[ \t]*\n(?P<body>(?:(?:[ ]{4}.*)?\n)*)',
    re.MULTILINE,
)
_PATH_LABELS = {"decorator": "Decorator", "notebook": "Notebook", "both paths": "Both paths"}

# A titled code block whose title starts with "Output".
_OUTPUT_BLOCK = re.compile(r'<div class="(?P<cls>[^"]*\bhighlight\b[^"]*)"><span class="filename">Output\b')


def strip_docnum_markers(markdown: str) -> str:
    """Remove ``docnum`` markers, keeping the value between them."""
    return _DOCNUM_MARKER.sub("", markdown)


def strip_test_injects(markdown: str) -> str:
    """Drop every ``# test:inject:`` line; the docs harness reads the source."""
    return _TEST_INJECT_LINE.sub("", markdown)


def applies_to_chip(markdown: str, *, chip: bool = True, keep: bool = True) -> str:
    """Turn the first "Applies to" box before any H2 into a chip line.

    ``chip=False`` keeps only the audience line; ``keep=False`` drops both.
    """
    m = _APPLIES_BOX.search(markdown)
    if not m or re.search(r"^## ", markdown[: m.start()], re.MULTILINE):
        return markdown
    body = " ".join(line.strip() for line in m.group("body").splitlines() if line.strip())
    if not keep:
        return markdown[: m.start()] + markdown[m.end() :]
    path = m.group("path").strip()
    parts = []
    if chip:
        slug = re.sub(r"[^a-z]+", "-", path.lower()).strip("-")
        label = _PATH_LABELS.get(path.lower(), path[:1].upper() + path[1:])
        parts.append(f'<span class="cash-path cash-path--{slug}" title="Applies to: {path}">{label}</span>')
    if body:
        parts.append(body)
    return (
        markdown[: m.start()]
        + '<p class="cash-applies" markdown>\n'
        + "\n".join(parts)
        + "\n</p>\n\n"
        + markdown[m.end() :]
    )


def mark_output_blocks(html: str) -> str:
    """Give code blocks titled "Output" the ``cash-output`` and ``no-copy`` classes."""
    return _OUTPUT_BLOCK.sub(
        lambda m: f'<div class="{m.group("cls")} cash-output no-copy"><span class="filename">Output', html
    )


def rewrite_badge_paths(html: str, page_url: str) -> str:
    """Rewrite root-absolute ``/_badges/...`` iframe srcs to page-relative ones.

    ``page_url`` is mkdocs' ``Page.url`` -- the page's URL relative to the site
    root, always with a trailing slash under directory-URL mode (the default):
    ``""`` for the home page, ``"badges/"`` one level down,
    ``"a/b/"`` two levels down.  The number of ``/`` therefore equals how many
    levels we must climb to reach the site root.
    """
    prefix = "../" * page_url.count("/")
    return _ABS_BADGE_SRC.sub(lambda m: m.group(1) + prefix + m.group(2) + m.group(3), html)


def _top_section(page) -> str | None:
    """Title of the nav tab the page sits in (``"Project"``), if any."""
    ancestors = getattr(page, "ancestors", None) or []
    return ancestors[-1].title if ancestors else None


def on_page_markdown(markdown: str, *, page, config, files, **kwargs) -> str:
    """mkdocs hook: rewrite the Markdown source before it is rendered."""
    markdown = strip_docnum_markers(markdown)
    markdown = strip_test_injects(markdown)
    return applies_to_chip(
        markdown,
        chip=_top_section(page) != "Project",
        # Not page.is_homepage: that is False for a home page nested in a nav
        # section, as ours is under "Home".
        keep=page.file.src_uri != "index.md",
    )


def on_post_page(output: str, *, page, config, **kwargs) -> str:
    """mkdocs hook: fix badge iframe paths and mark output blocks."""
    return mark_output_blocks(rewrite_badge_paths(output, page.url))
