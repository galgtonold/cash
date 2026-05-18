"""Smoke test for the badge_renderer.theme design-token module."""

from __future__ import annotations

import pytest

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
    # All family accents and shades are 6-digit hex strings.
    for token in (
        theme.RAIL_CACHED, theme.BAR_CACHED, theme.CHIP_BG_CACHED, theme.CHIP_FG_CACHED,
        theme.RAIL_EXEC, theme.BAR_EXEC, theme.CHIP_BG_EXEC, theme.CHIP_FG_EXEC,
        theme.RAIL_WARN, theme.BAR_WARN, theme.CHIP_BG_WARN, theme.CHIP_FG_WARN,
        theme.RAIL_MIXED,
    ):
        assert token.startswith("#") and len(token) in (4, 7)


def test_legacy_types_module_is_gone() -> None:
    """``_types.py`` was retired in slice 7 alongside the rest of the
    legacy renderer; the TypedDicts it held are replaced by BadgeView nodes."""
    import importlib
    with pytest.raises(ImportError):
        importlib.import_module("cash.notebook.badge_renderer._types")
