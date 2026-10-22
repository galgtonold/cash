"""An upstream restore whose trace looked code up by a string (marker
``string_lookup``) is never restored: at a lookup such a statement is served
on an agreeing trace only. Other unusable markers keep the old rule
(restored on the key alone)."""

import pytest

from cash.notebook import shadow as shadow_mod


@pytest.fixture
def sh(tmp_path):
    s = shadow_mod.Shadow()
    if not s.ok:
        pytest.skip(f"no monitoring tool: {s.reason}")

    class Backend:
        local_dir = str(tmp_path)

    s.attach({"__name__": "__main__"}, {}, Backend())
    return s


def entry(*markers, items=()):
    return {"items": [list(i) for i in items], "markers": list(markers)}


def test_string_lookup_trace_is_not_restored(sh):
    assert sh.restore_vouched("stmt:k", "t = globals()[name](3)", entry("string_lookup")) is False
    assert sh.counts["upstream:miss:string_lookup"] == 1
    assert sh.counts["upstream:hit:unusable"] == 0


def test_string_lookup_with_other_markers_is_not_restored(sh):
    assert sh.restore_vouched("stmt:k", "t = g(3)", entry("memo", "string_lookup", "jit")) is False
    assert sh.counts["upstream:miss:string_lookup"] == 1


def test_string_lookup_with_agreeing_items_is_still_not_restored(sh):
    # Items that would agree do not rescue it: the marker alone decides.
    assert sh.restore_vouched("stmt:k", "t = 1", entry("string_lookup", items=[("gap", "m", "q", "x")])) is False


@pytest.mark.parametrize("marker", sorted(shadow_mod.UNUSABLE - {"string_lookup"}))
def test_other_unusable_markers_are_restored_as_before(sh, marker):
    assert sh.restore_vouched("stmt:k", "t = g(3)", entry(marker)) is True
    assert sh.counts["upstream:hit:unusable"] == 1


def test_agreeing_trace_is_restored_and_no_trace_is_not(sh):
    assert sh.restore_vouched("stmt:k", "t = 1", entry()) is True
    assert sh.restore_vouched("stmt:k", "t = 1", None) is False
    assert sh.counts["upstream:miss:no_trace"] == 1
