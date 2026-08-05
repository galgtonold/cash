"""Regression guard: editing a cell must not trigger a notebook-read storm.

Before memoization, a single ``run_all`` re-read and re-parsed the notebook file
~30 times (the magic reads once per cell; the upstream checker reads twice per
cell), and each read paid a ``_wait_for_notebook_save`` sleep on the freshly
saved file — adding >1s of pure overhead per edit even though no user code was
recomputed.

This test pins the fix: across a ``run_all`` after an edit, the notebook is
actually read (the cache-miss path, which is the only place
``_wait_for_notebook_save`` runs) only a small bounded number of times.
"""
import pytest

pytestmark = [pytest.mark.upstream]


def _exec(nb_runner, code: str) -> str:
    out = {}

    def hook(msg):
        if msg['msg_type'] == 'stream':
            out['t'] = out.get('t', '') + msg['content']['text']

    nb_runner._run_async(
        nb_runner.client.kc._async_execute_interactive(
            code, store_history=False, output_hook=hook,
        )
    )
    return out.get('t', '')


def test_edit_does_not_trigger_notebook_read_storm(nb_runner):
    nb_runner.create_notebook([
        "a = 1",
        "b = a + 1",
        "c = b + 1\nprint(c)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Instrument the actual-read path (the cache MISS path) in the kernel.
    _exec(nb_runner, (
        "from cash.notebook import server_discovery as _sd\n"
        "_sd._READS = 0\n"
        "_orig = _sd._wait_for_notebook_save\n"
        "def _w(p):\n    _sd._READS += 1\n    return _orig(p)\n"
        "_sd._wait_for_notebook_save = _w\n"
    ))

    try:
        # Edit one cell and re-run everything.
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        reads = _exec(nb_runner, (
            "from cash.notebook import server_discovery as _sd\n"
            "print(getattr(_sd, '_READS', -1))"
        ))
        n = int(reads.strip().splitlines()[-1])
    finally:
        # RESTORE, and that is not tidiness. The patch above replaces a
        # function inside cash's OWN module, and prepare_for_test's module
        # purge deliberately spares cash's modules -- so under
        # CASH_TEST_REUSE_KERNEL=1 the wrapper survives into every later test
        # on this worker. A fresh kernel hides it, because the process dies
        # with the patch. Measured: this one file was enough to fail all three
        # tests in test_numpy_out_rerun.py ("306.0" not in "36.0" -- the edit
        # to an upstream cell never reached the downstream one), reproduced in
        # 5s from a two-file run.
        _exec(nb_runner, (
            "from cash.notebook import server_discovery as _sd\n"
            "_sd._wait_for_notebook_save = _orig\n"
        ))

    # Per file state we parse at most twice (the plain + include_ids variants).
    # A small bound (<= 4) leaves slack while still catching the ~30x storm.
    assert 0 < n <= 4, f"notebook read {n} times during one run_all (read storm regressed)"
