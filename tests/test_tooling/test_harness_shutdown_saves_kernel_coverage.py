"""The integration harness saves a kernel's coverage data before killing it.

``NotebookTestRunner.shutdown`` runs ``Coverage.current().save()`` in the
kernel when ``COVERAGE_PROCESS_START`` is set, because the kernel exits through
``os._exit`` and skips coverage's own atexit save. A test that called
``shutdown()`` itself was shut down again by the fixture's teardown, and that
second call built the save coroutine, failed to hand it to a ``_run_async``
that was already gone, and swallowed the error: the only trace was a
"coroutine ... was never awaited" warning. These tests drive ``shutdown`` with
a stand-in kernel client, no real kernel.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from types import SimpleNamespace

import pytest

from tests._nbharness import runner as runner_module
from tests._nbharness.runner import NotebookTestRunner


class _KernelClient:
    def __init__(self):
        self.executed: list[str] = []

    async def _async_execute_interactive(self, code, **kwargs):
        self.executed.append(code)

    def stop_channels(self):
        pass


@pytest.fixture
def started(tmp_path, monkeypatch):
    """A runner as start_kernel leaves it, over a stand-in kernel client."""
    monkeypatch.setenv("COVERAGE_PROCESS_START", str(tmp_path / ".coveragerc"))
    killed = []
    monkeypatch.setattr(runner_module, "_force_kill_kernel", killed.append)
    monkeypatch.setattr(runner_module, "_close_async_runner", lambda loop: None)
    runner = NotebookTestRunner(work_dir=tmp_path)
    km = SimpleNamespace(has_kernel=True)
    runner.client = SimpleNamespace(kc=_KernelClient(), km=km)
    runner._run_async = asyncio.run
    runner._kernel_started = True
    return runner, killed, km


def _shutdown_recording_warnings(runner) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.shutdown()
        gc.collect()  # an unawaited coroutine warns when it is collected
    return [str(w.message) for w in caught]


def test_shutdown_saves_the_kernels_coverage_then_kills_it(started):
    runner, killed, km = started
    kc = runner.client.kc
    assert _shutdown_recording_warnings(runner) == []
    assert any("_c.save()" in code for code in kc.executed), kc.executed
    assert killed == [km]


def test_a_second_shutdown_does_nothing(started):
    runner, killed, km = started
    runner.shutdown()
    kc = runner.client.kc
    kc.executed.clear()
    assert _shutdown_recording_warnings(runner) == []
    assert kc.executed == [] and killed == [km], "the second shutdown acted on a kernel already gone"


def test_a_coverage_save_that_cannot_run_says_so(started):
    runner, killed, km = started

    def loop_closed(coro):
        raise RuntimeError("Event loop is closed")

    runner._run_async = loop_closed
    messages = _shutdown_recording_warnings(runner)
    assert not any("never awaited" in m for m in messages), messages
    assert any("could not save the kernel's coverage data" in m and "Event loop is closed" in m for m in messages), (
        messages
    )
    assert killed == [km], "the kernel must still be killed"


def test_a_kernel_that_already_exited_is_not_asked_to_save(started):
    """A test that shut the kernel down through its manager leaves no kernel
    to answer: asking would wait out the whole run timeout for nothing."""
    runner, killed, km = started
    km.has_kernel = False
    kc = runner.client.kc
    assert _shutdown_recording_warnings(runner) == []
    assert kc.executed == []
    assert killed == [km]
