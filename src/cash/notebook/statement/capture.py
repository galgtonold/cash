"""Capturing a statement's output as it runs, and showing it afterwards.

``capture_output`` is the only IPython name this module needs at import time,
and it resolves it on first use: the decorator does not need IPython, and
this module is on the ``import cash`` chain.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from typing import TYPE_CHECKING, Any

from .._tee import TeeWriter

if TYPE_CHECKING:
    from cash.notebook.statement.results import ProcessResult

logger = logging.getLogger(__name__)

__all__ = [
    "NoCapture",
    "TeeWriter",
    "capture_output",
    "display_execution_output",
    "make_capture_ctx",
    "replay_outputs",
    "tee_output",
]


class NoCapture:
    """The captured output of a run that captured nothing."""

    def __init__(self) -> None:
        self.stdout = ""
        self.stderr = ""
        self.outputs: list = []


@contextmanager
def tee_output() -> Generator[Any, None, None]:
    """Context manager that tees stdout/stderr to both the real stream and a buffer.

    Yields a ``CapturedOutput``-compatible object whose ``.stdout`` and
    ``.stderr`` attributes contain the recorded text, while everything
    written during the block also appears on the real streams immediately.

    Performance: uses batched flushes (~100 ms) so that 50 000+ print calls
    complete in roughly the same time as native Python/Jupyter output.
    """

    class TeedOutput:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.outputs = []  # rich outputs not supported in tee mode

    teed = TeedOutput()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    old_stdout, old_stderr = sys.stdout, sys.stderr

    sys.stdout = TeeWriter(old_stdout, stdout_chunks)
    sys.stderr = TeeWriter(old_stderr, stderr_chunks)
    try:
        yield teed
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        teed.stdout = "".join(stdout_chunks)
        teed.stderr = "".join(stderr_chunks)
        sys.stdout, sys.stderr = old_stdout, old_stderr


# ``capture_output`` (below) falls back to this when IPython is missing,
# because it is a genuine working equivalent (it really does capture
# stdout/stderr), not a silent drop.
#
# ``display`` / ``publish_display_data`` deliberately get no fallback: there is
# no honest way to "render rich output" without IPython, and a no-op stub would
# make a display call silently vanish -- the cell appears to succeed while
# producing nothing. ``replay_outputs`` imports them when it needs them, so a
# genuine display attempt fails loudly with a clear ImportError.
@contextmanager
def _fallback_capture_output(
    stdout: bool = True, stderr: bool = True, display: bool = True
) -> Generator[Any, None, None]:
    """Fallback capture_output for when IPython is not available."""

    class CapturedOutput:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.outputs = []

        def show(self):
            if self.stdout:
                print(self.stdout, end="")
            if self.stderr:
                print(self.stderr, end="", file=sys.stderr)

    captured = CapturedOutput()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    if stdout:
        sys.stdout = StringIO()
    if stderr:
        sys.stderr = StringIO()

    try:
        yield captured
        if stdout:
            captured.stdout = sys.stdout.getvalue()
        if stderr:
            captured.stderr = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


_CAPTURE_OUTPUT = None


#: Methods ipykernel expects on ``shell.display_pub`` that IPython's
#: ``CapturingDisplayPublisher`` does not implement. ``set_parent`` is the one
#: that bites: ``zmqshell.set_parent`` calls it for EVERY shell message, from
#: the very top of ``dispatch_shell`` -- before the busy status is published
#: and before the message type is even read.
_KERNEL_PUBLISHER_METHODS = ("set_parent", "register_hook", "unregister_hook")


def _attach_kernel_publisher_api(pub: Any, real: Any) -> None:
    """Give the capturing publisher the methods ipykernel calls on it.

    ``capture_output(display=True)`` swaps ``shell.display_pub`` for a
    ``CapturingDisplayPublisher``, and that swap is **process-wide** and stays
    installed for the whole of a statement's execution. ipykernel dispatches
    shell messages during that window -- its shell channel runs in its own
    thread, and an ``await`` inside an async statement yields to the event loop
    besides -- and every dispatch begins with
    ``zmqshell.set_parent -> self.display_pub.set_parent(parent)``.

    ``CapturingDisplayPublisher`` has no such method, so the message died with
    ``AttributeError`` before it was handled: no busy status, no reply, and for
    an ``execute_request`` a cell that silently never ran. Observed on Binder.

    Forwards to the *real* publisher rather than swallowing the call. ipykernel
    tracks the parent so later output is attributed to the cell that caused it;
    a no-op would stop the exception and leave the real publisher holding a
    stale parent the moment capture exits, which trades a loud failure for a
    quiet mis-attribution.

    Bound per instance rather than patched onto IPython's class: the class is
    shared with every other ``capture_output`` user in the process, and cash
    does not get to change their behaviour.
    """
    for name in _KERNEL_PUBLISHER_METHODS:
        if hasattr(pub, name):
            continue
        target = getattr(real, name, None)
        if target is None:
            # A publisher that never had this method (a plain terminal
            # ``DisplayPublisher``) is not one ipykernel calls it on either, so
            # accepting and dropping the call is right here -- it only has to
            # be survivable, not meaningful.
            def _noop(*_args: Any, __name: str = name, **_kwargs: Any) -> None:
                logger.debug("[CAPTURE] %s() ignored: no real publisher", __name)

            setattr(pub, name, _noop)
        else:
            setattr(pub, name, target)


def _kernel_safe_capture(cls: Any) -> Any:
    """``capture_output`` that leaves ``shell.display_pub`` usable by the kernel.

    Subclasses rather than wraps so ``__exit__`` -- which restores the real
    publisher and the display hook -- is inherited untouched, exception
    propagation included.
    """

    class _KernelSafeCaptureOutput(cls):  # type: ignore[misc, valid-type]
        def __enter__(self):
            captured = super().__enter__()
            # ``self.display`` goes False when there is no shell to swap on, in
            # which case nothing was installed and there is nothing to repair.
            if getattr(self, "display", False) and self.shell is not None:
                pub = getattr(self.shell, "display_pub", None)
                if pub is not None:
                    _attach_kernel_publisher_api(pub, self.save_display_pub)
            return captured

    _KernelSafeCaptureOutput.__name__ = cls.__name__
    _KernelSafeCaptureOutput.__qualname__ = cls.__qualname__
    return _KernelSafeCaptureOutput


def capture_output(stdout: bool = True, stderr: bool = True, display: bool = True):
    """Resolve IPython's ``capture_output`` on FIRST USE, not at import.

    Importing this one name pulls in the whole of IPython, which measured at
    3.4s of the ~10s ``import cash`` took -- and ``import cash`` sits in front
    of every kernel start and every subprocess a test spawns. A test running
    three subprocesses therefore spent 30s+ on imports alone and tripped the
    30s per-test timeout, which reads as a hang rather than as slowness.

    Still resolved for real when a capture is actually requested, so a genuine
    display attempt keeps working; the stub below is used only when IPython is
    genuinely absent (the decorator runs without IPython).
    """
    global _CAPTURE_OUTPUT
    if _CAPTURE_OUTPUT is None:
        try:
            from IPython.utils.io import capture_output as _ipy_capture

            # See `_kernel_safe_capture`: IPython's publisher is missing methods
            # ipykernel calls on `shell.display_pub` while the swap is live.
            _CAPTURE_OUTPUT = _kernel_safe_capture(_ipy_capture)
        except ImportError:
            # The fallback never touches `display_pub`, so it has nothing to fix.
            _CAPTURE_OUTPUT = _fallback_capture_output
    return _CAPTURE_OUTPUT(stdout=stdout, stderr=stderr, display=display)


def make_capture_ctx(stream_output: bool, skip_capture: bool) -> Any:
    """Return the output-capture context manager for an execution.

    Streaming output that will not be cached skips capture entirely, so
    it reaches the real streams with no interception.
    """
    if stream_output and skip_capture:
        return contextlib.nullcontext(NoCapture())
    if stream_output:
        return tee_output()
    return capture_output(stdout=True, stderr=True, display=True)


def replay_outputs(stdout: str = "", stderr: str = "", rich: list | None = None) -> None:
    """Show captured output again: *stdout*, *stderr*, then the *rich*
    display outputs (display-data dicts, or objects to ``display``).

    Raises ImportError without IPython when there is rich output to show --
    see the module-header note: a caller replaying rich output expects it to
    render, so failing loudly beats silently dropping it. With no rich output
    IPython is never imported.
    """
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if not rich:
        return

    from IPython.display import display, publish_display_data

    for output in rich:
        if isinstance(output, dict) and "data" in output:
            publish_display_data(data=output["data"], metadata=output.get("metadata", {}))
        else:
            display(output)


def display_execution_output(captured: Any, silent: bool, stream_output: bool, metrics: ProcessResult) -> None:
    """Display captured stdout/stderr/rich outputs after execution."""
    if stream_output:
        # User already saw output in real-time via TeeWriter.
        metrics["_output_flushed"] = True
        if not silent:
            replay_outputs(rich=captured.outputs)
    elif not silent:
        replay_outputs(captured.stdout, captured.stderr, captured.outputs)
