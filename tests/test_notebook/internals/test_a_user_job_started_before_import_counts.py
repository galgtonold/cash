"""Review of claude/speed-model-once 42fef514: the exact kernel list holds for
threads started after ``import cash``, but a thread alive at import is still
trusted by its package (`_the_kernels`): an IPython background job or a
tornado loop the user started in an earlier cell, before ``import cash`` /
``%cash_on``, is taken for the kernel's for the whole session.

Put this file in tests/test_notebook/internals/.

Suggested fix: use `_KERNEL_THREADS` for the threads alive at import too
(the kernel's threads alive then are on that list), and drop the package
rule.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tornado")
pytest.importorskip("IPython")

_SCRIPT = textwrap.dedent(
    """
    import threading, time
    from tornado.ioloop import IOLoop
    loop = IOLoop(make_current=False)
    t = threading.Thread(target=loop.start, daemon=True)
    t.start()                       # cell 1: a small web app / bokeh server
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.instance()
    from IPython.lib.backgroundjobs import BackgroundJobManager
    go = threading.Event()
    jobs = BackgroundJobManager()
    jobs.new(lambda: go.wait(30))   # cell 2: a background job
    time.sleep(0.1)
    import cash                     # cell 3: import cash; %cash_on
    from cash.tracking import trace_recorder as tr
    print(int(tr.other_thread_alive({"loop": loop})))
    loop.add_callback(loop.stop); t.join(10)
    print(int(tr.other_thread_alive({"jobs": jobs})))
    go.set()
    """
)


def test_a_tornado_loop_and_a_background_job_started_before_import_count():
    out = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=120, check=True)
    tornado, job = out.stdout.split()
    assert tornado == "1", "the user's tornado loop, started before import, is taken for the kernel's"
    assert job == "1", "the user's background job, started before import, is taken for the kernel's"
