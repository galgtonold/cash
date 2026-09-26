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

The Warnings page marks each code with the path it comes from
(``<span class="md-tag cash-warning-path">notebook</span>``); the same
``cash-path--<path>`` class as the chip is added, so the two look alike.

Printed output
--------------
A code block titled ``Output`` (```` ```text title="Output" ````) is what a
program prints, not code to type. It gets the ``cash-output`` class for its own
look in ``cash-design.css`` and ``no-copy``, which drops Material's copy button.

Nav cross-links
---------------
mkdocs lists a page once. ``on_nav`` adds the entries in ``_NAV_CROSS_LINKS``:
a link in one nav section to a page that lives in another, such as "Why did it
miss?" in the Decorator tab opening the inspecting page of How it works.

Repository files
----------------
``<!-- include: CHANGELOG.md -->`` on a line of its own is replaced with that
file from the repository root, so the Changelog, Security policy and License
pages show the files GitHub shows, with no second copy to keep in sync. A
link in the file to ``docs/<page>.md`` becomes a link to that page; a link to
another repository file goes to it on GitHub.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_REPO_BLOB_URL = "https://github.com/galgtonold/cash/blob/main/"

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

# The path tag of a code on the Warnings page.
_WARNING_PATH_TAG = re.compile(r'<span class="md-tag cash-warning-path">(?P<path>decorator|notebook|both paths)</span>')

# A titled code block whose title starts with "Output".
_OUTPUT_BLOCK = re.compile(r'<div class="(?P<cls>[^"]*\bhighlight\b[^"]*)"><span class="filename">Output\b')

#: (tab, section, label, page): a nav link under tab > section to a page listed
#: elsewhere in the nav.
_NAV_CROSS_LINKS = [
    ("Decorator", "Start", "Why did it miss?", "how-it-works/inspecting.md"),
]

_INCLUDE = re.compile(r"^<!--\s*include:\s*(?P<path>[\w./-]+)\s*-->[ \t]*$", re.MULTILINE)
_MD_LINK_TARGET = re.compile(r"\]\((?P<target>(?![a-z][a-z0-9+.-]*:|#|/)[^)\s]+)\)")


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


def mark_warning_paths(html: str) -> str:
    """Give each Warnings-page path tag the chip class of its path."""
    return _WARNING_PATH_TAG.sub(
        lambda m: (
            f'<span class="md-tag cash-warning-path cash-path '
            f'cash-path--{m.group("path").replace(" ", "-")}">{m.group("path")}</span>'
        ),
        html,
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


def include_repo_files(markdown: str, page_src_uri: str, root: Path = _REPO_ROOT) -> str:
    """Replace ``<!-- include: PATH -->`` lines with the repository file PATH.

    ``page_src_uri`` is the page's path under ``docs/`` (``changelog.md``); a
    link in the included file is rewritten to work from there.
    """
    page_dir = posixpath.dirname(page_src_uri)

    def relink(m: re.Match[str]) -> str:
        target = m.group("target")
        path, _, anchor = target.partition("#")
        if path.startswith("docs/"):
            new = posixpath.relpath(path[len("docs/") :], page_dir or ".")
        else:
            new = _REPO_BLOB_URL + posixpath.normpath(path)
        return "](" + new + ("#" + anchor if anchor else "") + ")"

    def include(m: re.Match[str]) -> str:
        text = (root / m.group("path")).read_text(encoding="utf-8")
        return _MD_LINK_TARGET.sub(relink, text).rstrip("\n")

    return _INCLUDE.sub(include, markdown)


def _top_section(page) -> str | None:
    """Title of the nav tab the page sits in (``"Project"``), if any."""
    ancestors = getattr(page, "ancestors", None) or []
    return ancestors[-1].title if ancestors else None


def add_nav_cross_links(items, files, links=_NAV_CROSS_LINKS) -> None:
    """Append each cross-link to its nav section; fail if either end is gone."""
    from mkdocs.structure.nav import Link

    for tab, section, label, src in links:
        target = files.get_file_from_path(src)
        parent = None
        children = items
        for title in (tab, section):
            parent = next((i for i in children if getattr(i, "is_section", False) and i.title == title), None)
            if parent is None:
                break
            children = parent.children
        if parent is None or target is None:
            raise ValueError(f"nav cross-link {label!r}: no section {tab} > {section} or no page {src}")
        link = Link(label, target.url)
        link.parent = parent
        parent.children.append(link)


def on_nav(nav, *, config, files, **kwargs):
    """mkdocs hook: add the nav cross-links."""
    add_nav_cross_links(nav.items, files)
    return nav


def on_page_markdown(markdown: str, *, page, config, files, **kwargs) -> str:
    """mkdocs hook: rewrite the Markdown source before it is rendered."""
    markdown = include_repo_files(markdown, page.file.src_uri)
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
    """mkdocs hook: fix badge iframe paths, mark output blocks and path tags."""
    return mark_warning_paths(mark_output_blocks(rewrite_badge_paths(output, page.url)))
