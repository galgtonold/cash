"""Fixtures for the notebook integration tests.

The runner behind them, and every helper a test imports, is in the
``tests/_nbharness`` package.
"""

import os
import sys
import tempfile

import pytest

from tests._nbharness.kernels import DEFAULT_KERNEL_NAME, kernelspec_mismatch
from tests._nbharness.runner import NotebookTestRunner
from tests._nbharness.trace import TraceResult

# Crash visibility (faulthandler) is installed in the ROOT conftest
# (tests/conftest.py) so it covers every worker, not just notebook workers.


@pytest.fixture(scope="session", autouse=True)
def assert_kernelspec_is_this_interpreter():
    """Fail loudly if the kernelspec is not the interpreter under test.

    Every runner here boots ``kernel_name='python3'``, which
    :class:`KernelSpecManager` resolves from the user/system Jupyter search
    paths -- NOT from ``sys.executable``. So a stray ``ipykernel install --user``
    run from any other environment silently repoints the whole integration
    suite at a different interpreter. The suite then grades code that is not
    the editable install under test and still reports green, which makes every
    result quietly worthless. That failure is invisible by construction: there
    is no assertion anywhere that the kernel is us.

    Session-scoped and static (it reads the spec, boots nothing), so it costs
    one file read per worker and reports the problem before any test runs.

    Conservative: only a PROVABLE mismatch fails. A missing spec, a bare
    ``python``/``python3`` argv[0] with no path, or any resolution error leaves
    the suite alone -- this guard exists to catch a specific silent
    misconfiguration, not to become a new source of flakiness.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        argv = KernelSpecManager().get_kernel_spec(DEFAULT_KERNEL_NAME).argv
    except Exception:  # no spec / unreadable -> stay out of the way
        return
    problem = kernelspec_mismatch(argv, sys.executable)
    if problem:
        raise RuntimeError(problem)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def nb_runner(tmp_path, request):
    """
    Primary fixture for notebook integration tests.

    Provides a NotebookTestRunner instance that:
    - Uses real notebook files (no mocking)
    - Copies notebooks to tmp_path for isolation
    - Supports cell modification via file rewrites
    - Reuses one warm kernel per xdist worker (see ``tests/_nbharness/kernels.py``)

    Example:
        def test_example(nb_runner):
            nb_runner.create_notebook([
                "x = 10",
                "y = x * 2",
                "print(f'Result: {y}')"
            ])
            nb_runner.start_kernel()  # with_cash=True by default
            nb_runner.run_all()
            assert "Result: 20" in nb_runner.get_output(3)

            # Modify a cell and re-run
            nb_runner.set_cell_source(1, "x = 100")
            nb_runner.run_cells([1, 2, 3])
            assert "Result: 200" in nb_runner.get_output(3)
    """
    runner = NotebookTestRunner(work_dir=tmp_path)
    # A test marked `fresh_kernel` opts OUT of warm-kernel reuse. Under
    # reuse, shutdown()+start_kernel() is a namespace reset on the SAME
    # process -- measured, pid unchanged -- so a test whose subject IS the
    # restart stops testing anything and passes vacuously.
    runner._force_fresh_kernel = request.node.get_closest_marker("fresh_kernel") is not None
    yield runner
    runner.shutdown()


@pytest.fixture
def upstream_trace(nb_runner):
    """Capture the upstream-checker decision trace for a scenario.

    Returns a callable ``capture(cells, actions, *, with_cash=True) -> TraceResult``.
    ``actions(nb_runner)`` performs the re-run (e.g. ``lambda r: r.run_cell(2)``)
    after an initial ``run_all``. Example::

        def test_x(upstream_trace):
            t = upstream_trace(
                ["log = []", "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))"],
                lambda r: r.run_cell(2),
            )
            assert "log = []" not in t.scheduled()   # producer must not re-run
    """
    import json

    created: list[str] = []

    # The trace file is named by an environment variable the kernel PROCESS
    # reads at boot, so this fixture can only ever work on a kernel booted
    # after `_capture` sets it. Under CASH_TEST_REUSE_KERNEL=1 the warm kernel
    # boots once per worker, before any test runs -- so the trace file stays
    # empty and every assertion about the trace fails on an empty record list.
    #
    # That failure is order-dependent in the most misleading way: the FIRST
    # trace test on a worker passes (its warm kernel had not been claimed yet),
    # and every later one fails, which reads as one test poisoning another.
    # Measured: test_nocache_inplace_does_not_reexecute_producer passes alone
    # and fails behind ANY predecessor, whatever that predecessor does.
    #
    # Opting out of reuse here, rather than marking each calling test, keeps
    # the requirement with the fixture that has it.
    nb_runner._force_fresh_kernel = True

    def _capture(cells, actions, *, with_cash: bool = True) -> "TraceResult":
        fd, path = tempfile.mkstemp(suffix=".cashtrace.jsonl")
        os.close(fd)
        created.append(path)
        prev = os.environ.get("CASH_TRACE_FILE")
        os.environ["CASH_TRACE_FILE"] = path
        try:
            nb_runner.create_notebook(cells)
            nb_runner.start_kernel(with_cash=with_cash)
            nb_runner.run_all()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write('{"event": "__run_all_done__"}\n')
            actions(nb_runner)
            with open(path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            return TraceResult(records)
        finally:
            if prev is None:
                os.environ.pop("CASH_TRACE_FILE", None)
            else:
                os.environ["CASH_TRACE_FILE"] = prev

    yield _capture

    for p in created:
        try:
            os.unlink(p)
        except OSError:
            pass
