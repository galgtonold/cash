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
