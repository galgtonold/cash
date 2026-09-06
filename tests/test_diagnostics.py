"""The diagnostic code registry is the canonical list of warning codes.

Codes are permanent once released — a rename breaks a link that exists in
someone's terminal scrollback forever — so their shape is pinned here.
"""
from __future__ import annotations

import re

import pytest

from cash.diagnostics import DIAGNOSTIC_CODES, doc_url

SLUG = re.compile(r"^[A-Z]+(?:-[A-Z]+)+$")
AREAS = {"CACHE", "KEY", "STORE", "IMPURE", "REMOTE", "RANDOM", "NOTEBOOK", "ANNOT"}


def test_the_registry_is_not_empty():
    assert len(DIAGNOSTIC_CODES) >= 25


@pytest.mark.parametrize("code", sorted(DIAGNOSTIC_CODES))
def test_every_code_follows_the_slug_rule(code):
    assert SLUG.match(code), f"{code!r} is not AREA-PROBLEM in uppercase"
    assert len(code) <= 24, f"{code!r} is {len(code)} chars, limit is 24"
    assert code.split("-")[0] in AREAS, f"{code!r} starts with an unknown area"


def test_doc_url_points_at_the_stable_docs():
    url = doc_url("CACHE-THRASH")
    assert url == (
        "https://cash-lib.readthedocs.io/en/stable/warnings/#cache-thrash"
    )


def test_doc_url_rejects_an_unregistered_code():
    """A typo must fail here rather than ship a dead link."""
    with pytest.raises(KeyError):
        doc_url("CACHE-TYPPO")


import warnings

from cash.diagnostics import format_diagnostic, warn_diagnostic, warn_diagnostic_explicit
from cash.exceptions import CashCacheIneffectiveWarning


def test_the_rendered_message_carries_code_fix_and_link():
    text = format_diagnostic(
        "CACHE-THRASH", "the cache is full at its 500 MB cap.", "raise max_cache_size."
    )
    assert text.startswith("[CACHE-THRASH] the cache is full at its 500 MB cap.")
    assert "\n  Fix: raise max_cache_size." in text
    assert text.endswith(
        "\n  https://cash-lib.readthedocs.io/en/stable/warnings/#cache-thrash"
    )


def test_the_code_reaches_the_handler_as_an_attribute():
    """Callers must be able to branch on `w.message.code` rather than
    substring-matching prose, which is not a stable interface."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_diagnostic(
            CashCacheIneffectiveWarning, "CACHE-THRASH", "the cache is full.", "raise it."
        )
    assert len(caught) == 1
    assert caught[0].message.code == "CACHE-THRASH"
    assert isinstance(caught[0].message, CashCacheIneffectiveWarning)


def test_an_unregistered_code_raises_rather_than_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KeyError):
            warn_diagnostic(
                CashCacheIneffectiveWarning, "CACHE-NOPE", "something.", "do a thing."
            )
    assert caught == [], "nothing should have been emitted for a bad code"


def test_the_explicit_variant_keeps_the_caller_s_location():
    """These sites blame the user's cell on purpose; losing that is a
    regression even though the text would still be correct."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_diagnostic_explicit(
            CashCacheIneffectiveWarning, "CACHE-THRASH", "the cache is full.",
            "raise it.", filename="<cash>", lineno=42, registry=None,
        )
    assert len(caught) == 1
    assert caught[0].filename == "<cash>"
    assert caught[0].lineno == 42
    assert str(caught[0].message).startswith("[CACHE-THRASH] ")
    # The asymmetry itself, which docs/warnings.md now warns readers about:
    # ``warn_explicit`` takes a message STRING, so there is no object for
    # ``.code`` to ride on. ``diagnostics.py`` says this is "pinned by test";
    # until this line it was not -- the assertions above pin the location and
    # the rendered prefix, both of which would survive an attribute appearing.
    assert getattr(caught[0].message, "code", None) is None
