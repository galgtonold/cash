"""The installed version of a library a function body imports is part of
what keys it (`library_versions`), as one it reads as a global is, whether
the import has run yet or not; a relative import of the user's own package
and the standard library add none."""

from __future__ import annotations

import sys

from cash.decorator.library_versions import library_versions


def _uses_six():
    from six import moves

    return moves


def _uses_stdlib():
    from email.mime.text import MIMEText

    return MIMEText


def test_a_body_import_of_an_installed_library_adds_its_version():
    held = sys.modules.pop("six", None)
    try:
        names = [name for name, _version in library_versions([_uses_six], None)]
        assert "six" not in sys.modules  # read without importing it
    finally:
        if held is not None:
            sys.modules["six"] = held
    assert "six" in names


def test_a_body_import_of_the_standard_library_adds_none():
    assert library_versions([_uses_stdlib], None) == []
