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

from .._tee import StreamTaken, TeeWriter

if TYPE_CHECKING:
    from cash.notebook.statement.results import ProcessResult

logger = logging.getLogger(__name__)

__all__ = [
    "NoCapture",
    "TeeWriter",
    "capture_output",
    "display_execution_output",
    "make_capture_ctx",
    "REPLAY_RECORD_CAP",
    "replay_outputs",
    "show_unshown_output",
    "unshown_output",
    "replay_record",
    "tee_output",
]


#: The most output, in bytes, an entry's metadata keeps for a step a cell run
#: may skip (``replay_record``). Over it, the step runs instead.
REPLAY_RECORD_CAP = 64 * 1024


def replay_record(captured: Any) -> dict | None:
    """What *captured* showed, for an entry's metadata to replay when a cell
    run skips its statement (``NotebookSimulator.plan_cell_run``).

    None when it showed nothing. ``{"complete": False}`` alone when it
    showed more than ``REPLAY_RECORD_CAP`` bytes or a rich output that is
    not plain display data: that statement is run rather than skipped.
    """
    stdout = getattr(captured, "stdout", "") or ""
    stderr = getattr(captured, "stderr", "") or ""
    outputs = list(getattr(captured, "outputs", None) or [])
    if not (stdout or stderr or outputs):
        return None
    rich: list[dict] = []
    size = len(stdout) + len(stderr)
    for output in outputs:
        data = getattr(output, "data", None) if not isinstance(output, dict) else output.get("data")
        metadata = getattr(output, "metadata", None) if not isinstance(output, dict) else output.get("metadata")
        if not isinstance(data, dict) or not all(isinstance(v, (str, dict, list)) for v in data.values()):
            return {"complete": False}
        size += sum(len(str(v)) for v in data.values())
        rich.append({"data": dict(data), "metadata": dict(metadata or {})})
    if size > REPLAY_RECORD_CAP:
        return {"complete": False}
    return {"complete": True, "stdout": stdout, "stderr": stderr, "rich": rich}


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
        """What was written so far, read at any time (``CodeRunner`` asks
        before it shows the cell's value)."""

        def __init__(self, stdout_chunks: list[str], stderr_chunks: list[str]):
            self._stdout_chunks = stdout_chunks
            self._stderr_chunks = stderr_chunks
            self.outputs = []  # rich outputs not supported in tee mode

        @property
        def stdout(self) -> str:
            return "".join(self._stdout_chunks)

        @property
        def stderr(self) -> str:
            return "".join(self._stderr_chunks)

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    teed = TeedOutput(stdout_chunks, stderr_chunks)
    with (
        StreamTaken("stdout", lambda real: TeeWriter(real, stdout_chunks)) as out,
        StreamTaken("stderr", lambda real: TeeWriter(real, stderr_chunks)) as err,
    ):
        try:
            yield teed
        finally:
            out.sink.flush()
            err.sink.flush()


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


class _Captured:
    """What a statement wrote and displayed: ``.stdout``, ``.stderr`` and
    ``.outputs``, as IPython's ``CapturedIO`` gives them."""

    def __init__(self, shown: Any, out: StreamTaken | None, err: StreamTaken | None) -> None:
        self._shown = shown
        self._out = out
        self._err = err

    @property
    def stdout(self) -> str:
        return self._out.getvalue() if self._out is not None else ""

    @property
    def stderr(self) -> str:
        return self._err.getvalue() if self._err is not None else ""

    @property
    def outputs(self) -> list:
        return list(getattr(self._shown, "outputs", None) or [])


@contextmanager
def _capture_in_place(
    cls: Any, stdout: bool, stderr: bool, display: bool, tee: bool = False
) -> Generator[Any, None, None]:
    """IPython's capture for the displays; the streams taken in place.

    IPython's capture puts back the ``sys.stdout`` and ``sys.stderr`` it
    found on exit, even when it took neither: a statement that rebinds one
    (``sys.stdout = buf``) had it undone, and what the cell printed next went
    to the notebook instead. What the statement left there is kept.

    *tee*: what is printed also goes on to the stream as it is written
    (`TeeWriter`), as in plain; the displays are still held."""
    sink = (lambda real: TeeWriter(real)) if tee else (lambda _real: StringIO())
    with contextlib.ExitStack() as stack:
        left: dict[str, Any] = {}
        stack.callback(lambda: [setattr(sys, name, stream) for name, stream in left.items()])
        shown = stack.enter_context(cls(stdout=False, stderr=False, display=display))
        stack.callback(lambda: left.update(stdout=sys.stdout, stderr=sys.stderr))
        out = stack.enter_context(StreamTaken("stdout", sink)) if stdout else None
        err = stack.enter_context(StreamTaken("stderr", sink)) if stderr else None
        if tee:
            stack.callback(lambda: [taken.sink.flush() for taken in (out, err) if taken is not None])
        yield _Captured(shown, out, err)


def capture_output(stdout: bool = True, stderr: bool = True, display: bool = True, tee: bool = False):
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
    if _CAPTURE_OUTPUT is _fallback_capture_output:
        if tee:
            return tee_output()
        return _CAPTURE_OUTPUT(stdout=stdout, stderr=stderr, display=display)
    return _capture_in_place(_CAPTURE_OUTPUT, stdout, stderr, display, tee)


def make_capture_ctx(stream_output: bool, skip_capture: bool, keep_displays: bool = False) -> Any:
    """Return the output-capture context manager for an execution.

    Streaming output that will not be cached skips capture entirely, so
    it reaches the real streams with no interception. *keep_displays*: what
    is printed streams, and the displays are captured as without streaming.
    """
    if stream_output and skip_capture:
        return contextlib.nullcontext(NoCapture())
    if stream_output and keep_displays:
        return capture_output(stdout=True, stderr=True, display=True, tee=True)
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
        elif _is_captured_display(output):
            # Sent again as it was sent. ``display(output)`` formats the capture
            # as an object, and a bundle with no ``text/plain`` (a plotly
            # figure's) gets the object's repr there: ``<RichOutput at 0x..>``.
            output.display()
        else:
            display(output)


def _is_captured_display(output: Any) -> bool:
    """A display that ``capture_output`` recorded (IPython's ``RichOutput``):
    the mime bundle it was sent with, and a ``display()`` that sends it again."""
    return isinstance(getattr(output, "data", None), dict) and callable(getattr(output, "display", None))


#: What an interrupted statement printed or displayed and nobody has shown
#: yet, on the interrupt (`unshown_output`).
_UNSHOWN = "_cash_unshown_output"


def unshown_output(exc: BaseException, captured: Any, silent: bool, stream_output: bool) -> None:
    """Show what a statement *exc* interrupts printed and displayed, or keep
    it on *exc* for the cell to show (*silent*: the cell shows a statement's
    output once it ends, and this one does not end).

    Streamed, the prints have been shown as they came; the displays have not.
    An upstream replay's statement keeps it on *exc* too, and nothing shows
    it: plain never ran that statement."""
    stdout = "" if stream_output else (getattr(captured, "stdout", "") or "")
    stderr = "" if stream_output else (getattr(captured, "stderr", "") or "")
    rich = list(getattr(captured, "outputs", None) or ())
    if not (stdout or stderr or rich):
        return
    if not silent:
        replay_outputs(stdout, stderr, rich)
        return
    try:
        setattr(exc, _UNSHOWN, (stdout, stderr, rich))
    except (AttributeError, TypeError):  # an exception class with __slots__
        replay_outputs(stdout, stderr, rich)


def show_unshown_output(exc: BaseException) -> None:
    """Show, once, what the statement *exc* interrupted printed and displayed."""
    held = getattr(exc, _UNSHOWN, None)
    if held is None:
        return
    try:
        delattr(exc, _UNSHOWN)
    except AttributeError:
        pass
    replay_outputs(*held)


def display_execution_output(captured: Any, silent: bool, stream_output: bool, metrics: ProcessResult) -> None:
    """Display captured stdout/stderr/rich outputs after execution."""
    if stream_output:
        # User already saw output in real-time via TeeWriter.
        metrics["_output_flushed"] = True
        if not silent:
            replay_outputs(rich=captured.outputs)
    elif not silent:
        replay_outputs(captured.stdout, captured.stderr, captured.outputs)
