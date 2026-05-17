"""Tests for the text renderer over the BadgeView IR."""

from __future__ import annotations

from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def test_cached_summary_header() -> None:
    metrics = [{"code": "x=1", "status": str(CacheStatus.RESTORED),
                "total_time": 0.01, "saved_time": 0.5}]
    text = render_text(build_interactive_badge(metrics))
    assert text.startswith("[Cash]")
    assert "CACHED" in text
    assert "RESTORED" in text
    assert "saved 0.50s" in text


def test_upstream_section_label_and_indent() -> None:
    metrics = [
        {"code": "setup()", "status": str(CacheStatus.RESTORED), "is_upstream": True,
         "total_time": 0.01, "saved_time": 0.5},
        {"code": "compute()", "status": str(CacheStatus.COMPUTED), "total_time": 0.3},
    ]
    text = render_text(build_interactive_badge(metrics))
    assert "Upstream:" in text
    assert "⬆️" in text
    assert "COMPUTED" in text


def test_iteration_context_stripped() -> None:
    metrics = [{
        "code": "# __iteration_context__: deadbeef\nprocess(x)",
        "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
    }]
    text = render_text(build_interactive_badge(metrics))
    assert "process(x)" in text
    assert "deadbeef" not in text
    assert "__iteration_context__" not in text


def test_decorator_summary_section() -> None:
    metrics = [{
        "code": "f()", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
        "decorator_calls": [
            {"func_name": "myf", "cache_hit": True, "execution_time": 0.001},
            {"func_name": "myf", "cache_hit": False, "execution_time": 0.05},
        ],
    }]
    text = render_text(build_interactive_badge(metrics))
    assert "@cash.cache:" in text
    assert "myf(): 1/2 cached" in text
