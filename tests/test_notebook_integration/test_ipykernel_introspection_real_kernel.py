"""CAS-134 in a real kernel: cash must not lie to ipykernel's introspection.

Companion to ``tests/test_notebook/test_ipykernel_signature_introspection.py``.
That one proves the property against stubs in-process; this one proves it inside
a **real ipykernel process**, where the real ``do_execute`` really does call
``_accepts_parameters(shell.run_cell_async, ["cell_id"])`` before dispatching.

Two independent assertions, neither of which trusts the badge:

1.  The cell's work is verified by an **out-of-band side effect** — the cell
    writes a file, and we assert the file exists. The badge is not evidence:
    CAS-134 shipped precisely because the badge reported ``EXECUTED`` for cells
    that never ran.
2.  Introspection **parity** is computed *inside the kernel*, against the true
    original that cash stashes on ``shell._cash_hooks``. That is what fails on
    the unfixed wrapper.

Note on reach: the *user-visible* hang needs IPython < 8.3 (the last release
whose ``run_cell_async`` lacks ``cell_id``), which no dev environment pins, so a
real kernel here cannot be made to hang. The parity assertion is what makes the
defect observable on any IPython — see the unit test's module docstring.
"""
import os
import textwrap

import pytest

pytestmark = [pytest.mark.timeout(120)]

PROBE = ["cell_id", "cell_meta", "_cash_definitely_not_a_real_parameter"]


def _kernel_runs_cash_under_test(nb_runner) -> bool:
    """True when the integration kernel imports the same cash tree as this suite.

    The ``python3`` kernelspec is resolved from the Jupyter search path, not from
    the interpreter running pytest, so it can point at a *different* environment
    holding a *different* (e.g. released) cash. When it does, this test would be
    grading code that is not the working tree — we skip instead of reporting a
    green or a red that means nothing.
    """
    import cash

    nb_runner.run_cells([1])
    reported = nb_runner.get_output(1).strip()
    expected = os.path.realpath(os.path.dirname(cash.__file__))
    return bool(reported) and os.path.realpath(reported) == expected


def test_patched_hooks_do_not_lie_to_ipykernel_in_a_real_kernel(nb_runner, tmp_path):
    side_effect = tmp_path / "cell_actually_ran.txt"

    verdict_cell = textwrap.dedent("""\
        from ipykernel.kernelbase import _accepts_parameters as _ap
        _ip = get_ipython()
        _probe = {probe!r}
        _hooks = getattr(_ip, '_cash_hooks', None)
        if not _hooks or 'original_run_cell_async' not in _hooks:
            print('NO_ASYNC_HOOK')
        else:
            _before = _ap(_hooks['original_run_cell_async'], _probe)
            _after = _ap(_ip.run_cell_async, _probe)
            print('PARITY_OK' if _before == _after
                  else f'PARITY_FAIL before={{_before}} after={{_after}}')
    """).format(probe=PROBE)

    nb_runner.create_notebook([
        "import cash, os; print(os.path.dirname(cash.__file__))",
        "%cash_on",
        # A magic-containing cell whose real work is observable off-badge.
        "%cash_on\n"
        f"open({str(side_effect)!r}, 'w').write('RAN')\n"
        "print('side effect written')",
        verdict_cell,
    ])
    nb_runner.start_kernel()

    if not _kernel_runs_cash_under_test(nb_runner):
        pytest.skip(
            "the 'python3' kernelspec resolves to an environment whose cash is not "
            "the tree under test, so this test would not grade the working tree"
        )

    nb_runner.run_all()

    # 1. Out-of-band proof the cell really executed. NOT the badge, NOT the output.
    assert side_effect.exists(), (
        "the magic-containing cell never actually executed — no file was written. "
        "(Do not relax this into a badge/output check: the badge reports EXECUTED "
        "for cells that never ran, which is how CAS-134 shipped.)"
    )
    assert side_effect.read_text() == "RAN"

    # 2. ipykernel's real verdict on cash's hook == its verdict on the original.
    verdict = nb_runner.get_output(4)
    assert "NO_ASYNC_HOOK" not in verdict, "cash did not patch run_cell_async at all"
    assert "PARITY_OK" in verdict, (
        "inside a real kernel, ipykernel's introspection of cash's patched "
        f"run_cell_async disagrees with its introspection of the original: {verdict}"
    )
