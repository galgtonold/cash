"""Reassignment-accumulator loop trust (CAS-120 / CAS-91b).

A loop that accumulates by *reassignment* (``total = total + b`` or
``total += b``) must be trusted on a no-change downstream re-run exactly like an
in-place accumulator (``results.append(...)``) already is.  If it is not, the
loop is marked "broken" and re-executed on every downstream read; when the
iterable is a ONE-SHOT consumable (a stored generator), the re-execution drains
an already-exhausted source and produces the WRONG value.

The critical guard is :func:`test_reassign_accumulator_reexecutes_on_upstream_edit`
which proves the trust does NOT leak into genuine upstream edits (no
under-invalidation).
"""
import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations]


@pytest.mark.timeout(90)
def test_reassign_accumulator_generator_not_redrained(nb_runner):
    """Core fix: ``total = total + v`` over a one-shot generator survives a
    plain downstream re-run (no edits) — the generator is not re-drained."""
    nb_runner.create_notebook([
        "g = (i for i in range(6))",              # sum(0..5) = 15, one-shot
        "total = 0\nfor v in g:\n    total = total + v",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=15" in nb_runner.get_output(3), nb_runner.get_output(3)

    # Plain re-run of the downstream reader, NO edits anywhere. A plain kernel
    # would just read total=15. Cash must not re-execute the loop (which would
    # drain the now-exhausted generator down to total=0).
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "total=15" in out, f"one-shot generator was re-drained: {out}"


@pytest.mark.timeout(90)
def test_augmented_accumulator_generator_not_redrained(nb_runner):
    """Control (augmented ``+=``): behaves the same as the plain-reassign case."""
    nb_runner.create_notebook([
        "g = (i for i in range(6))",
        "total = 0\nfor v in g:\n    total += v",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=15" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "total=15" in out, f"one-shot generator was re-drained (+=): {out}"


@pytest.mark.timeout(90)
def test_reassign_accumulator_reexecutes_on_upstream_edit(nb_runner):
    """CRITICAL under-invalidation guard.

    A reassignment accumulator that genuinely depends on an upstream input
    (``total = total + factor * b`` with ``factor`` from an upstream cell) must
    STILL re-execute when ``factor`` is edited — the trust must not serve a
    stale ``total``.  Uses a re-usable list iterable so the correct re-execution
    is observable.
    """
    nb_runner.create_notebook([
        "factor = 2",
        "src = [1, 2, 3, 4]",
        "total = 0\nfor b in src:\n    total = total + factor * b",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=20" in nb_runner.get_output(4), nb_runner.get_output(4)  # 2*10

    # Genuine upstream edit: factor 2 -> 3. The loop MUST re-run and reflect it.
    nb_runner.set_cell_source(1, "factor = 3")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "total=30" in out, f"under-invalidation: stale total served: {out}"  # 3*10


@pytest.mark.timeout(90)
def test_inplace_append_accumulator_generator_not_redrained(nb_runner):
    """Control (in-place ``.append``): the existing trust path is not regressed
    — a one-shot generator consumed by an append accumulator stays correct."""
    nb_runner.create_notebook([
        "g = (i for i in range(6))",
        "results = []\nfor v in g:\n    results.append(v)",
        "print(f'results={results}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "results=[0, 1, 2, 3, 4, 5]" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "results=[0, 1, 2, 3, 4, 5]" in out, f"append accumulator regressed: {out}"
