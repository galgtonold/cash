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


# ---------------------------------------------------------------------------
# Which frame a warning blames
#
# This replaced 34 hand-tuned `stacklevel` constants. They were unverifiable by
# reading, and four diagnostics shipped pointing at a line inside `core.py` --
# which tells a reader nothing they can act on, and is the one question a code
# and a doc link do not answer. The depth is now resolved at emit time, so this
# is a property of every site rather than a number per site.
# ---------------------------------------------------------------------------

import os
import tempfile

from cash import Cash  # noqa: E402
from cash.backends.file_backend import FileBackend  # noqa: E402
from cash.diagnostics import _is_cash_frame, _stacklevel_of_first_user_frame  # noqa: E402

CASH_ROOT = os.path.dirname(os.path.abspath(__import__("cash").__file__))


class _FakeFrame:
    def __init__(self, filename):
        self.f_code = type("C", (), {"co_filename": filename})()


@pytest.mark.parametrize(
    "filename, internal",
    [
        (os.path.join(CASH_ROOT, "core.py"), True),
        (os.path.join(CASH_ROOT, "notebook", "randomness.py"), True),
        # A notebook statement compiles under these. They are the USER's code,
        # and a substring test for "cash" would call them internal — silently
        # skipping past the frames we most want to blame.
        ("<cash>", False),
        ("<cash-f83fdfd0c487>", False),
        ("<stdin>", False),
        (os.path.join("some", "user", "script.py"), False),
    ],
)
def test_what_counts_as_a_cash_frame(filename, internal):
    assert _is_cash_frame(_FakeFrame(filename)) is internal


def test_the_resolver_points_past_this_module():
    """Called straight from a test, the nearest non-Cash frame is this one, so
    the answer is the minimum: blame the caller of the resolver's caller."""
    assert _stacklevel_of_first_user_frame() == 1


def _blamed_files(trigger) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger()
    assert caught, "trigger emitted no warning; the test proves nothing"
    return [w.filename for w in caught]


def _unhashable_arg():
    c = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)

    @c.cache
    def f(x):
        return 1

    f(lambda z: z)


def _unseeded_randomness():
    import random

    c = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)

    @c.cache
    def draws():
        return random.random()

    return draws


def _bound_method_state():
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("deliberately unpicklable")

        def method(self):
            return 1

    c = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    c.cache(Unpicklable().method)()


def _cache_if_bypassed():
    c = Cash(backend=FileBackend(tempfile.mkdtemp(), flush_interval=0),
             register_magic=False)

    @c.cache(chunk_max_items=3, cache_if=lambda r: True)
    def gen():
        yield from range(10)

    list(gen())


@pytest.mark.parametrize(
    "trigger",
    [_unhashable_arg, _unseeded_randomness, _bound_method_state, _cache_if_bypassed],
    ids=["unhashable-arg", "unseeded-randomness", "bound-method", "cache-if-bypassed"],
)
def test_no_warning_blames_a_frame_inside_cash(trigger):
    """The property the constants could not hold.

    These four span both emit helpers and a range of call depths — decoration
    time, argument hashing, key folding, and mid-iteration inside a generator.
    Every one of them blamed `core.py` at some point in this library's history.
    """
    for filename in _blamed_files(trigger):
        assert not filename.startswith(CASH_ROOT), (
            f"{trigger.__name__} blamed {filename}, which is inside Cash — "
            "the reader cannot act on that"
        )
