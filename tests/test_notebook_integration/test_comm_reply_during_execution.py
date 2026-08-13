"""Can a kernel receive a comm message WHILE it is executing a cell?

This is the one fact a "ask the frontend for the live notebook" design turns
on, and it is worth its own test because the answer decides an architecture
rather than a detail.

Background. cash reads the cells it did not execute from the saved ``.ipynb``,
so an unsaved upstream edit is invisible and a downstream cell can restore a
stale value while the screen shows new code. Colab is exempt because its
frontend implements ``get_ipynb``. Every route to the equivalent elsewhere was
measured and closed: JupyterLab 4 / Notebook 7 expose no application object, no
Lumino back-reference from the DOM, and window the notebook so unrendered cells
are not in the DOM at all; VS Code runs output JS inside a ``vscode-webview://``
sandbox that cannot see the workbench. What remains is a frontend EXTENSION,
and an extension talks to the kernel over a **comm**.

So: cash would ask, mid-statement, "what does the editor currently hold?" and
would have to BLOCK until the answer arrives -- it needs the value at the point
it checks upstream state, not afterwards. ipykernel normally processes shell
messages one at a time, and an execute_request occupies that slot for the whole
cell, so an inbound comm would ordinarily wait until the cell finishes. Which
would be useless.

``jupyter_ui_poll`` exists precisely to re-enter the kernel's event loop from
inside a running cell and service pending frontend messages, and on current
ipykernel it is the ONLY route: ``IPythonKernel.do_one_iteration`` -- the hand
-rolled pump every older recipe reaches for -- no longer exists on ipykernel
7.2. So this test also decides whether cash would take a dependency on
``jupyter_ui_poll``, since there is nothing to fall back to.

If this passes, a request/response extension is viable and cash can read live
state at the moment it needs it. If it fails, the extension must instead PUSH
cell contents to the kernel on every change, and cash reads a local cache --
a different design, decided here rather than discovered late.

The test plays the frontend itself, so it needs raw shell-channel access and
manages its own kernel rather than using the ``nb_runner`` fixture.
"""
from __future__ import annotations

import json
import queue

import pytest

pytest.importorskip("jupyter_client")

# The kernel-side probe needs a pump, and on current ipykernel this is the only
# one there is. Without it the probe cannot even attempt the thing under test,
# and its wait loop would exit immediately -- a pass/fail that means nothing.
# The conftest asserts the kernelspec is THIS interpreter, so importing it here
# is a sound proxy for it being importable inside the kernel.
pytest.importorskip(
    "jupyter_ui_poll",
    reason="no way to pump the kernel loop mid-execution; the probe would be vacuous",
)

# Registered by the kernel, opened by us. Any name works; this one is
# recognisable in a message dump.
TARGET = "cash_q2_probe"

# Bounds the kernel-side wait. Comfortably longer than a comm round-trip on a
# loaded CI box, comfortably shorter than the test's own read timeouts, so a
# negative result is "the kernel never saw it" and not "we gave up first".
KERNEL_WAIT_S = 10.0

_KERNEL_CODE = r'''
import json, time
from IPython import get_ipython

_R = {"received": False, "waited": None, "pump": None, "pump_error": None}
_shell = get_ipython()


def _on_open(comm, msg):
    _R["received"] = True
    try:
        comm.close()
    except Exception:
        pass


_shell.kernel.comm_manager.register_target("%(target)s", _on_open)

# The test must not send the comm before the target exists, or the kernel
# rejects it and we would be measuring our own race instead of the mechanism.
print("PROBE_READY", flush=True)

_t0 = time.monotonic()
try:
    from jupyter_ui_poll import ui_events
    _R["pump"] = "jupyter_ui_poll"
    with ui_events() as poll:
        while not _R["received"] and time.monotonic() - _t0 < %(wait)s:
            poll(10)          # service up to 10 pending frontend messages
            time.sleep(0.02)
except Exception as exc:
    # Report the fallback route taken rather than silently degrading: which
    # mechanism works is itself the finding.
    _R["pump_error"] = f"{type(exc).__name__}: {exc}"
    _kernel = _shell.kernel
    _fn = getattr(_kernel, "do_one_iteration", None)
    _R["pump"] = "do_one_iteration" if _fn else "none available"
    while _fn is not None and not _R["received"] and time.monotonic() - _t0 < %(wait)s:
        try:
            _res = _fn()
            if hasattr(_res, "__await__"):
                _loop = getattr(_kernel, "io_loop", None)
                if _loop is not None and hasattr(_loop, "run_sync"):
                    _loop.run_sync(lambda: _fn())
                    _R["pump"] = "io_loop.run_sync(do_one_iteration)"
                else:
                    import asyncio
                    asyncio.get_event_loop().run_until_complete(_res)
                    _R["pump"] = "asyncio.run_until_complete(do_one_iteration)"
        except Exception as exc2:
            _R["pump_error"] = f"{_R['pump_error']} | pump: {type(exc2).__name__}: {exc2}"
            break
        time.sleep(0.02)

_R["waited"] = round(time.monotonic() - _t0, 3)
print("PROBE_RESULT " + json.dumps(_R), flush=True)
'''


@pytest.fixture
def kernel():
    """A real kernel with raw channel access, torn down whatever happens."""
    from jupyter_client import KernelManager

    km = KernelManager(kernel_name="python3")
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=60)
    except RuntimeError:  # pragma: no cover - a dead kernel is an env problem
        kc.stop_channels()
        km.shutdown_kernel(now=True)
        pytest.skip("kernel did not become ready")
    try:
        yield kc
    finally:
        try:
            kc.stop_channels()
        finally:
            km.shutdown_kernel(now=True)


def _stdout_until(kc, marker: str, timeout: float = 30.0) -> str:
    """Read iopub until a stdout line starts with *marker*; return that line."""
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=1.0)
        except queue.Empty:
            continue
        if msg["msg_type"] == "error":
            raise AssertionError(
                "kernel raised while probing:\n" + "\n".join(msg["content"]["traceback"])
            )
        if msg["msg_type"] != "stream":
            continue
        for line in msg["content"].get("text", "").splitlines():
            if line.startswith(marker):
                return line
    raise AssertionError(f"never saw {marker!r} within {timeout}s")


def _send_comm_open(kc) -> None:
    """Act as the frontend: open a comm on the shell channel.

    ``session.send`` wants the zmq socket, not the channel wrapper -- passing
    the channel raises ``'ZMQSocketChannel' object has no attribute
    'send_multipart'``.
    """
    kc.session.send(
        kc.shell_channel.socket,
        "comm_open",
        content={"comm_id": "q2-probe-comm", "target_name": TARGET, "data": {}},
    )


def _run_probe(kc, *, send_while_busy: bool) -> dict:
    code = _KERNEL_CODE % {"target": TARGET, "wait": KERNEL_WAIT_S}
    kc.execute(code)

    # Wait until the target is registered, so the comm cannot be rejected for
    # arriving too early -- that would look like a negative result.
    _stdout_until(kc, "PROBE_READY")

    if send_while_busy:
        _send_comm_open(kc)

    line = _stdout_until(kc, "PROBE_RESULT", timeout=KERNEL_WAIT_S + 30)
    return json.loads(line[len("PROBE_RESULT "):])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ANSWERED, NO -- and pinned so a future ipykernel that changes it says "
        "so loudly. On ipykernel 7.2 a comm sent while a cell is executing is "
        "queued and delivered only after the cell ends, even with "
        "jupyter_ui_poll actively pumping, and IPythonKernel.do_one_iteration "
        "no longer exists as a fallback. Delivery itself is fine -- "
        "test_the_comm_is_delivered_once_the_cell_finishes proves the same "
        "message lands once the cell is done -- so this is deferral, not a "
        "malformed probe. Design consequence: a frontend extension cannot "
        "ANSWER cash mid-statement; it must PUSH cell contents on change while "
        "the kernel is idle, and cash reads a locally-held copy."
    ),
)
def test_a_comm_sent_while_the_kernel_is_busy_is_received_before_the_cell_ends(kernel):
    """THE question. A frontend extension can only answer cash synchronously if
    the kernel can take delivery mid-execution."""
    r = _run_probe(kernel, send_while_busy=True)

    assert r["received"], (
        "The kernel did not process an inbound comm while executing a cell "
        f"(pump={r['pump']!r}, pump_error={r['pump_error']!r}, "
        f"waited {r['waited']}s).\n\n"
        "Consequence: a request/response frontend extension cannot serve cash a "
        "live read at the moment it checks upstream state. The extension would "
        "have to PUSH cell contents to the kernel on every change, with cash "
        "reading a locally-held copy instead of asking."
    )
    # The pump route is version-sensitive, so record which one carried it --
    # a future ipykernel that silently changes this should be legible here.
    assert r["pump"] in {
        "jupyter_ui_poll",
        "do_one_iteration",
        "io_loop.run_sync(do_one_iteration)",
        "asyncio.run_until_complete(do_one_iteration)",
    }, f"unexpected pump route: {r['pump']!r}"


def test_the_comm_is_delivered_once_the_cell_finishes(kernel):
    """The control that makes the result above mean anything.

    "Not received during execution" and "never sent anything the kernel could
    parse" look identical from inside the probe. So: send the same message the
    same way, let the cell end, then ask the kernel again. If the flag flipped
    after the cell finished, delivery works and was merely DEFERRED -- which is
    the real finding. If it never flips, this test file is measuring a
    malformed message and proves nothing.
    """
    r = _run_probe(kernel, send_while_busy=True)

    # The probe's own loop has ended; the shell queue can now drain.
    kernel.execute('import json; print("AFTER " + json.dumps(_R))')
    after = json.loads(_stdout_until(kernel, "AFTER ")[len("AFTER "):])

    assert after["received"], (
        "the comm never reached the kernel at all, not even after the cell "
        f"ended (during-execution result was {r['received']!r}). The message "
        "this test sends is malformed or unroutable, so the headline result "
        "above is not evidence about mid-execution delivery."
    )


def test_the_probe_reports_nothing_received_when_no_comm_is_sent(kernel):
    """The control, and it is load-bearing.

    Without it, a probe that reported ``received`` unconditionally -- or a
    ``_on_open`` fired by some unrelated comm -- would pass the test above and
    we would build on a mechanism that never worked. Same code path, same
    kernel, only the frontend's message withheld.
    """
    r = _run_probe(kernel, send_while_busy=False)

    assert not r["received"], (
        "the probe reported a comm it was never sent -- the positive test above "
        "proves nothing"
    )
    assert r["waited"] >= KERNEL_WAIT_S - 1, (
        f"the wait loop exited early ({r['waited']}s), so the positive result "
        "may just be a loop that never really waited"
    )
