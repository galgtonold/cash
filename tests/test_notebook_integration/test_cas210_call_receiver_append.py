"""CAS-210: a call-expression receiver must not make an append re-fire.

``open(p, 'a').write(x)`` spells its receiver as a Call, not a name. Receiver
resolution walked through the Call to the CALLEE and returned ``open``, so the
statement was booked as an in-place mutation of ``open`` -- a builtin that no
statement can mutate. That bumped "lineage" forced the writer to re-execute
during upstream reconstruction, and because the write is a ``mode='a'`` append,
re-execution appended the line a SECOND time. The file on disk diverged from
what a plain kernel produces, and the damage is not recoverable by restarting.

The scope gate (CAS-193/196/200) does not catch this: it keys on *relevance*
("does any relevant consumer read this file?"), not on *repeatability* ("is
re-running this write safe?"). Here the file IS read, so the gate steps aside.

The oracle is a no-cash kernel: cash ON must leave the file byte-identical to
cash OFF, both after ``run_all`` and after an upstream edit.
"""
import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(180)]


def _p(path):
    return str(path).replace("\\", "/")


def _lines(path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _cells(audit_path, *, cash_on):
    """The repro notebook, with the cash magics included only when *cash_on*."""
    ap = _p(audit_path)
    first = "import cash\n%cash_on\n%cash_badge print" if cash_on else "pass"
    return [
        first,                                                                # 1
        "scale = 2",                                                          # 2
        "payload = scale * 21",                                               # 3
        # writer: APPEND mode, payload-dependent, call-expression receiver
        f"open(r'{ap}', 'a').write(f'entry-{{payload}}\\n')\nwrote = True",   # 4
        # reader of the SAME file -> puts the writer in the consumer's scope
        f"# @cash:no-cache\nn_lines = len(open(r'{ap}').read().splitlines())\n"
        f"print('LINES', n_lines)",                                           # 5
    ]


def test_append_with_call_receiver_matches_a_plain_kernel(nb_runner, tmp_path):
    """cash ON must write the same bytes as cash OFF -- run_all and after an edit."""
    audit = tmp_path / "audit.log"

    nb_runner.create_notebook(_cells(audit, cash_on=True))
    nb_runner.start_kernel()
    nb_runner.run_all()

    after_runall = _lines(audit)
    assert after_runall == ["entry-42"], (
        f"run_all duplicated the append: {after_runall!r}. The writer re-fired "
        f"during reconstruction because its receiver was attributed to `open`."
    )

    # Edit the payload upstream, then run ONLY the reader. Reconstruction must
    # not re-fire the writer: the user did not run the writer's own cell.
    nb_runner.set_cell_source(2, "scale = 3")
    nb_runner.run_cell(5)

    after_edit = _lines(audit)
    assert after_edit == after_runall, (
        f"upstream edit + running the READER re-fired the append: "
        f"{after_runall!r} -> {after_edit!r}"
    )


def test_plain_kernel_oracle_writes_exactly_one_line(nb_runner, tmp_path):
    """Pin the oracle itself, so the assertion above cannot drift silently."""
    audit = tmp_path / "audit.log"

    nb_runner.create_notebook(_cells(audit, cash_on=False))
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert _lines(audit) == ["entry-42"]
