"""Re-spelling a numeric literal must not throw the cache away.

``0.5`` and ``0.50`` are the same IEEE double -- identical bits, identical
entry in ``co_consts`` -- so a function that swaps one for the other is the
same function once compiled. Hashing the literal's TEXT meant that swap
recomputed everything.

Found by watching a user do it: he re-typed ``0.5`` as ``0.50``, saw a long
computation re-run, and was told on the spot that it was floating-point
imprecision. It was not. ``0.5 == 0.50`` is exactly ``True``, and
``test_the_two_spellings_really_are_one_value`` below pins that so the wrong
explanation cannot come back.

The rule is *same value AND same type*. Collapsing ``1`` into ``1.0`` would be
a real bug -- different type, different behaviour downstream -- so the
distinctness half of this file matters at least as much as the collapsing
half.
"""
from __future__ import annotations

import sys

import pytest

from cash.source_norm import _canonical_number, normalize_source_for_hash


def _digest(literal: str) -> str:
    return normalize_source_for_hash(f"def f():\n    return {literal}\n")


# Each group is one value written several ways. Every member must digest
# identically to the first.
SAME_VALUE = {
    "float spellings": ["0.5", "0.50", ".5", "0.500", "5e-1", "5E-1"],
    "int bases and separators": ["1000", "1_000", "0x3e8", "0X3E8", "0o1750",
                                 "0b1111101000"],
    "float exponents": ["1000.0", "1e3", "1E3", "1_000.0"],
    "complex": ["1j", "1J"],
}


@pytest.mark.parametrize("group", sorted(SAME_VALUE))
def test_one_value_has_one_digest(group):
    spellings = SAME_VALUE[group]
    digests = {_digest(s) for s in spellings}
    assert len(digests) == 1, (
        f"{group}: {spellings} are the same value but produced "
        f"{len(digests)} different cache keys"
    )


def test_the_two_spellings_really_are_one_value():
    """The claim underneath the whole file, pinned against a wrong retelling.

    This was explained to a user as floating-point imprecision -- as though
    0.5 and 0.50 were merely *close*. They are the same object-level constant.
    """
    assert 0.5 == 0.50
    assert (0.5).hex() == (0.50).hex()
    assert (compile("x = 0.5", "<s>", "exec").co_consts
            == compile("x = 0.50", "<s>", "exec").co_consts)


# Pairs that must NOT collapse. Type is behaviour: `1` and `1.0` index, divide
# and serialize differently, so they are different functions.
DISTINCT = [
    ("1", "1.0"),
    ("1", "1j"),
    ("1.0", "1j"),
    ("0.5", "0.6"),
    ("1000", "10000"),
    ("1", "1_0"),          # 1_0 is ten, not one
    ("0.1", "0.2"),
]


@pytest.mark.parametrize(("left", "right"), DISTINCT)
def test_different_values_keep_different_digests(left, right):
    assert _digest(left) != _digest(right)


def test_a_literal_edit_still_invalidates():
    """The control. A rule that collapsed everything would pass the tests above."""
    assert _digest("0.5") != _digest("0.500001")


def test_an_unrenderable_integer_falls_back_to_its_text():
    """A hasher must never raise; a coarse digest beats an exception.

    Python refuses to ``repr`` an integer past ``sys.set_int_max_str_digits``
    (4300 by default), which is reachable in generated code.
    """
    huge = "1" + "0" * (sys.get_int_max_str_digits() + 100)
    assert _canonical_number(huge) == huge
    assert _digest(huge)  # and the surrounding digest still computes


def test_hex_containing_an_e_is_not_read_as_a_float():
    """``0x1e3`` is 483. Testing the float branch before the base branch would
    have made it 1e3, silently colliding two different constants."""
    assert _canonical_number("0x1e3") == repr(483)
    assert _digest("0x1e3") != _digest("1e3")


def test_numbers_inside_strings_are_untouched():
    """Only NUMBER tokens are canonicalised; a string is data, not a literal."""
    assert (normalize_source_for_hash('def f():\n    return "0.50"\n')
            != normalize_source_for_hash('def f():\n    return "0.5"\n'))


def test_the_normalizer_still_ignores_comments_and_layout():
    """Guard against the new branch disturbing what the digest already dropped."""
    plain = "def f():\n    return 0.5\n"
    dressed = "def f():\n\n    # a note\n    return 0.50\n"
    assert normalize_source_for_hash(plain) == normalize_source_for_hash(dressed)
