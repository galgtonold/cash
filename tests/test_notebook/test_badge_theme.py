"""Smoke test for the badge_renderer.theme design-token module."""

from __future__ import annotations

from cash.notebook.badge_renderer import theme


def test_typography_tokens_are_non_empty_strings() -> None:
    assert isinstance(theme.FONT_SANS, str) and theme.FONT_SANS
    assert isinstance(theme.FONT_MONO, str) and theme.FONT_MONO


def test_display_limits_are_positive() -> None:
    assert theme.CODE_SNIPPET_MAX_LEN > 0
    assert theme.HEADER_MAX_LEN > 0
    assert theme.MIN_TIME_DISPLAY_S > 0
    assert theme.MIN_TIME_DISPLAY_MS > 0
    # MS threshold must be finer than the S threshold (overhead vs row).
    assert theme.MIN_TIME_DISPLAY_MS < theme.MIN_TIME_DISPLAY_S


def test_status_palette_uses_hex_colors() -> None:
    for token in (theme.BADGE_COLOR_RESTORED, theme.BADGE_COLOR_DEFAULT):
        assert token.startswith("#") and len(token) in (4, 7)


def test_legacy_underscore_aliases_no_longer_exist() -> None:
    # Slice 1 removed the underscore-prefixed shim from _types.py.
    # Imports must come from .theme directly going forward.
    from cash.notebook.badge_renderer import _types
    for name in (
        "_FONT_SANS", "_FONT_MONO",
        "_CODE_SNIPPET_MAX_LEN", "_HEADER_MAX_LEN",
        "_MIN_TIME_DISPLAY_S", "_MIN_TIME_DISPLAY_MS",
        "_BADGE_COLOR_RESTORED", "_BADGE_COLOR_DEFAULT",
    ):
        assert not hasattr(_types, name), (
            f"{name} should live in .theme now, not ._types"
        )
