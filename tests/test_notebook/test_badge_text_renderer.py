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
    # The header and the row under it use the SAME word for the same state.
    # They used to disagree -- header "CACHED", row "RESTORED" -- which is what
    # sent four docs pages describing the wrong label (CAS-272).
    assert text.count("CACHED") == 2, text
    assert "RESTORED" not in text, "internal vocabulary leaked into the badge"
    assert "saved 0.50s" in text


def test_restored_row_saved_is_avoided_compute_not_restore_time() -> None:
    """A RESTORED statement row's "saved" is the compute we skipped
    (``saved_time``), NOT the tiny deserialise wall-clock (``total_time``).

    The row line used to print ``time_s`` under a "saved" label, so a 0.02s
    restore of a statement whose true saving was 0.44s read as "saved 0.02s".
    total_time and saved_time are made distinct here to catch that regression.
    """
    metrics = [{"code": "df = make_frame()", "status": str(CacheStatus.RESTORED),
                "total_time": 0.02, "saved_time": 0.44}]
    text = render_text(build_interactive_badge(metrics))
    assert "saved 0.44s" in text            # header AND the statement row
    assert "saved 0.02s" not in text        # the restore time is never "saved"


def test_upstream_section_label_and_indent() -> None:
    metrics = [
        {"code": "setup()", "status": str(CacheStatus.RESTORED), "is_upstream": True,
         "total_time": 0.01, "saved_time": 0.5},
        {"code": "compute()", "status": str(CacheStatus.COMPUTED), "total_time": 0.3},
    ]
    text = render_text(build_interactive_badge(metrics))
    assert "Upstream:" in text
    assert "^CACHED" in text  # ASCII upstream marker; see below for why
    assert "EXECUTED" in text
    # The text badge feeds headless/agent runs, so it is read by a different
    # process than wrote it. Any character a cp1252 console cannot encode
    # crashes that reader instead of showing it the badge.
    text.encode("cp1252")


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


def test_text_badge_is_ascii_across_every_status():
    """No status may emit a character a legacy-codepage reader cannot decode.

    ``%cash_badge print`` is documented as required for headless / agent runs,
    so its output is consumed by a DIFFERENT process than the kernel that wrote
    it -- nbconvert, a log scraper, an agent parsing the .ipynb. The kernel's
    stdout is always UTF-8, so an emoji is written happily and then raises
    UnicodeEncodeError in the reader: a traceback instead of the badge, for
    precisely the audience the mode exists for.

    Shipped 0.1.0 emitted U+2192, U+2699, U+2705 and U+FE0F here.
    """
    from cash.notebook.badge_renderer.view import BadgeStatus

    metrics = []
    for i, status in enumerate(BadgeStatus):
        metrics.append({
            "code": f"stmt_{i}()",
            "status": str(status.value),
            "total_time": 0.25,
            "time_saved": 0.5,
            "storage_tiers": ["RAM", "DISK"],
        })
    text = render_text(build_interactive_badge(metrics))
    assert text, "expected badge output"
    try:
        text.encode("cp1252")
    except UnicodeEncodeError as exc:
        bad = text[exc.start:exc.end]
        raise AssertionError(
            f"text badge emitted {bad!r} (U+{ord(bad[0]):04X}), which crashes a "
            f"cp1252 reader. The text renderer must stay ASCII -- put the glyph "
            f"in the HTML renderer instead."
        ) from None


def test_a_multiline_statement_stays_one_line_in_the_text_badge() -> None:
    """The text badge documents one line per statement.

    It renders ``row.code.splitlines()[0]``, so handing it the display source
    would silently print ``x = (`` and drop the rest. It keeps the unparsed
    form on purpose; `_agent_guide.py` reproduces this shape verbatim and
    `test_agent_guide_sync` pins it.
    """
    metrics = [{
        "code": "x = a + 1",
        "display_code": "x = (\n    a\n    + 1\n)",
        "status": str(CacheStatus.COMPUTED),
        "total_time": 0.5,
    }]
    text = render_text(build_interactive_badge(metrics))
    body = [ln for ln in text.splitlines() if "x = " in ln]
    assert len(body) == 1, f"expected one statement line, got {body}"
    assert "x = a + 1" in body[0], "the text badge lost the statement's tail"
