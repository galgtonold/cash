"""CAS-195: ``%cash_badge print`` crashes on a NESTED control cell.

CONFIRMED on the working tree (post-6dc2540), with two corrections to the
ticket:

1. The ticket's stated minimal repro (a FLAT ``for i in range(4): acc += i``)
   does NOT crash — see the two negative-control tests below, which pass.
   The crash needs a control structure nested inside another one.
2. The concrete class reaching ``_row_line`` is ``ForLoopGroup``, not
   ``ControlGroupSingle`` (the ticket's static guess). Same line, same root
   cause.

Root cause (``badge_renderer``):
  * ``view_builder._section_item_from_grouped`` builds ``ControlGroup.rows``
    from ``sub_items`` via a recursive dispatch, so ``rows`` may hold
    ``ForLoopGroup`` / ``ControlGroupSingle`` / ``ControlGroup`` -- NOT only
    ``StatementRow``.
  * ``renderers/text.py:126`` (``_item_lines``, ControlGroup branch) does
    ``[_row_line(r, ...) for r in item.rows]``, and ``_row_line`` at :84 does
    ``row.code`` -> AttributeError.
  * ``_iter_rows`` at :164 has the same flat assumption (``yield from
    item.rows``), so the guard summary would fault on the same shape.

Impact is worse than the ticket records: the AttributeError escapes into the
IPKernel message handler, so the cell never sends an execute_reply and the
client BLOCKS until its timeout -- not merely "the badge is silently dropped".

The badge output IS the artifact under test here, so capturing it verbatim is
the correct evidence (unlike caching claims, which need an external counter).
Non-ASCII is stripped before printing to survive the Windows console.
"""
import pytest

pytestmark = [pytest.mark.timeout(400)]

SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    "import time"
)

CRASH = "AttributeError"


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def _check(nb_runner, label: str, cell_num: int) -> str:
    out = nb_runner.get_raw_output(cell_num)
    print(f"\n===== {label} (cell {cell_num}) =====\n{_ascii(out)}\n=====")
    return out


# ---------------------------------------------------------------------------
# THE BUG: a control structure nested inside another one.
# ---------------------------------------------------------------------------

def test_for_nested_in_if_badge_print_crashes(nb_runner):
    """``for`` inside ``if`` -> ControlGroup.rows holds a ForLoopGroup, which
    is handed straight to ``_row_line``.

    Fails today with, in the kernel's stderr::

        File ".../badge_renderer/renderers/text.py", line 126, in _item_lines
          return [_row_line(r, is_upstream=is_upstream) for r in item.rows]
        File ".../badge_renderer/renderers/text.py", line 84, in _row_line
          code = row.code.splitlines()[0][...] if row.code else ""
        AttributeError: 'ForLoopGroup' object has no attribute 'code'

    and the execute_reply never arrives (client times out at 120s).
    """
    nb_runner.create_notebook([
        SETUP,
        "flag = True\n"
        "total = 0\n"
        "if flag:\n"
        "    for i in range(4):\n"
        "        time.sleep(0.02)\n"
        "        total += i",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = _check(nb_runner, "for-in-if", 2)
    assert CRASH not in out, f"CAS-195 reproduced (for-in-if):\n{_ascii(out)}"


def test_nested_if_in_if_badge_print(nb_runner):
    """``if`` inside ``if`` with a single-statement inner body -- the shape
    most likely to put a ControlGroupSingle in ControlGroup.rows (the ticket's
    original static hypothesis)."""
    nb_runner.create_notebook([
        SETUP,
        "flag = True\n"
        "n = 4\n"
        "total = 0\n"
        "if flag:\n"
        "    if n > 2:\n"
        "        time.sleep(0.05)\n"
        "        total = n * 10",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = _check(nb_runner, "if-in-if", 2)
    assert CRASH not in out, f"CAS-195 reproduced (if-in-if):\n{_ascii(out)}"


# ---------------------------------------------------------------------------
# Negative controls: these PASS. They pin that the ticket's stated minimal
# repro is not the trigger, so a fix must not be validated against them.
# ---------------------------------------------------------------------------

def test_flat_loop_cell_badge_print_is_fine(nb_runner):
    """The ticket's claimed minimal repro. Renders cleanly -- ticket is wrong
    about the trigger."""
    nb_runner.create_notebook([
        SETUP,
        "acc = 0\n"
        "for i in range(4):\n"
        "    time.sleep(0.02)\n"
        "    acc += i",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = _check(nb_runner, "flat loop", 2)
    nb_runner.run_cell(2)
    out2 = _check(nb_runner, "flat loop (warm re-run)", 2)
    for o in (out, out2):
        assert CRASH not in o, _ascii(o)


def test_if_nested_in_for_badge_print_is_fine(nb_runner):
    """``if`` inside ``for`` -- the body groups per-iteration, so no
    ControlGroup wraps a group. Renders cleanly."""
    nb_runner.create_notebook([
        SETUP,
        "total = 0\n"
        "for i in range(4):\n"
        "    if i % 2 == 0:\n"
        "        time.sleep(0.02)\n"
        "        total += i",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = _check(nb_runner, "if-in-for", 2)
    nb_runner.run_cell(2)
    out2 = _check(nb_runner, "if-in-for (warm re-run)", 2)
    for o in (out, out2):
        assert CRASH not in o, _ascii(o)
