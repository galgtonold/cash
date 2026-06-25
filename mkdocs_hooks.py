"""mkdocs build hooks for the Cash docs.

Wired up via ``hooks:`` in ``mkdocs.yml``.  mkdocs imports this file directly
(by path) and calls any ``on_*`` event functions it defines.

Why this exists
---------------
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
"""
from __future__ import annotations

import re

# Match the root-absolute badge src and split off the leading slash so we can
# splice a page-relative prefix in front of the "_badges/..." remainder.
_ABS_BADGE_SRC = re.compile(r'(<iframe\b[^>]*\bsrc=")/(_badges/[^"]+)(")')


def rewrite_badge_paths(html: str, page_url: str) -> str:
    """Rewrite root-absolute ``/_badges/...`` iframe srcs to page-relative ones.

    ``page_url`` is mkdocs' ``Page.url`` -- the page's URL relative to the site
    root, always with a trailing slash under directory-URL mode (the default):
    ``""`` for the home page, ``"badges/"`` one level down,
    ``"a/b/"`` two levels down.  The number of ``/`` therefore equals how many
    levels we must climb to reach the site root.
    """
    prefix = "../" * page_url.count("/")
    return _ABS_BADGE_SRC.sub(
        lambda m: m.group(1) + prefix + m.group(2) + m.group(3), html
    )


def on_post_page(output: str, *, page, config, **kwargs) -> str:  # noqa: ARG001
    """mkdocs hook: fix badge iframe paths in every rendered page."""
    return rewrite_badge_paths(output, page.url)
