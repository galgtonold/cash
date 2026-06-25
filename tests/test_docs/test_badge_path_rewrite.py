"""Unit tests for the mkdocs build hook that fixes badge iframe paths.

Badge example iframes are authored with a root-absolute ``src`` (e.g.
``src="/_badges/anatomy_hero.html"``).  That only resolves when the site is
served at the domain root (local ``mkdocs serve``).  On Read the Docs the site
is served under ``/en/<version>/``, so those absolute paths 404 and the browser
renders RTD's unstyled 404 page inside each badge frame.

``mkdocs_hooks.rewrite_badge_paths`` rewrites the absolute path to one relative
to the current page so it resolves regardless of the base path the site is
mounted at.  These tests pin that behaviour at every nav depth.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "mkdocs_hooks.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rewrite = _load_hook().rewrite_badge_paths

_IFRAME = '<iframe class="cash-badge" src="{src}" loading="lazy"></iframe>'


@pytest.mark.parametrize(
    "page_url,expected_prefix",
    [
        ("", ""),                                   # home: /en/latest/
        ("badges/", "../"),                         # /en/latest/badges/
        ("cost-model/", "../"),                     # /en/latest/cost-model/
        ("getting-started/quickstart/", "../../"),  # two levels deep
        ("tutorials/use-cases/data-science/", "../../../"),  # three levels deep
    ],
)
def test_absolute_badge_src_becomes_page_relative(page_url, expected_prefix):
    html = _IFRAME.format(src="/_badges/anatomy_hero.html")
    out = rewrite(html, page_url)
    assert f'src="{expected_prefix}_badges/anatomy_hero.html"' in out
    # The root-absolute form must be gone entirely.
    assert 'src="/_badges/' not in out


def test_multiple_iframes_all_rewritten():
    html = (
        _IFRAME.format(src="/_badges/status_computed.html")
        + "\n"
        + _IFRAME.format(src="/_badges/status_restored.html")
    )
    out = rewrite(html, "badges/")
    assert 'src="../_badges/status_computed.html"' in out
    assert 'src="../_badges/status_restored.html"' in out
    assert 'src="/_badges/' not in out


def test_non_badge_absolute_src_left_untouched():
    html = '<iframe src="/some/other/page.html"></iframe>'
    assert rewrite(html, "badges/") == html


def test_already_relative_src_is_idempotent():
    html = _IFRAME.format(src="../_badges/anatomy_hero.html")
    assert rewrite(html, "badges/") == html
