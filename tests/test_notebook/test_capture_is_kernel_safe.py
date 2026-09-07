"""The publisher cash installs while capturing must satisfy ipykernel's interface.

Reported from a live Binder session:

    AttributeError: 'CapturingDisplayPublisher' object has no attribute 'set_parent'
      ipykernel/kernelbase.py  dispatch_shell -> self.set_parent(...)
      ipykernel/zmqshell.py    set_parent     -> self.display_pub.set_parent(parent)

`capture_output(display=True)` swaps `shell.display_pub` for IPython's
`CapturingDisplayPublisher`, which does not implement `set_parent`. The swap is
process-wide and stays installed for the whole of a statement's execution, so any
shell message the kernel dispatches in that window hits it.

**This is not cosmetic.** `set_parent` is called at the very top of
`dispatch_shell` -- before the busy status is published and before the message
type is even read -- so an exception there drops the message unhandled. For an
`execute_request` that means a cell silently never runs and never replies.

`ZMQDisplayPublisher` also has `register_hook`/`unregister_hook`, which
`CapturingDisplayPublisher` lacks; those are called by user code rather than by
dispatch, but they are the same defect and are covered here too.
"""
from __future__ import annotations

import pytest

pytest.importorskip("IPython")

from IPython.core.displaypub import DisplayPublisher

from cash.notebook.statement.processor import StatementProcessor


class _RecordingPublisher(DisplayPublisher):
    """Stands in for ipykernel's ZMQDisplayPublisher, which does implement these."""

    def __init__(self):
        super().__init__()
        self.parents = []
        self.hooks = []

    def set_parent(self, parent):
        self.parents.append(parent)

    def register_hook(self, hook):
        self.hooks.append(hook)

    def unregister_hook(self, hook):
        self.hooks.remove(hook)
        return True


@pytest.fixture
def shell_with_recording_pub(monkeypatch):
    """A real InteractiveShell whose display_pub records what ipykernel would do."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    real = _RecordingPublisher()
    monkeypatch.setattr(shell, "display_pub", real, raising=False)
    return shell, real


def test_set_parent_during_capture_does_not_raise(shell_with_recording_pub):
    """The reported crash. ipykernel calls this for EVERY shell message, so it
    fires whenever one arrives while a statement is running."""
    shell, real = shell_with_recording_pub
    parent = {"header": {"msg_id": "abc", "msg_type": "execute_request"}}

    with StatementProcessor._make_capture_ctx(stream_output=False, skip_capture=False):
        shell.display_pub.set_parent(parent)


def test_set_parent_during_capture_reaches_the_real_publisher(shell_with_recording_pub):
    """Not merely swallowed. ipykernel tracks the parent so later output is
    attributed to the right cell; a no-op that only stops the exception would
    leave the real publisher holding a stale parent once capture exits."""
    shell, real = shell_with_recording_pub
    parent = {"header": {"msg_id": "abc", "msg_type": "execute_request"}}

    with StatementProcessor._make_capture_ctx(stream_output=False, skip_capture=False):
        shell.display_pub.set_parent(parent)

    assert real.parents == [parent], (
        "the parent set during capture never reached the real publisher"
    )


def test_display_hooks_survive_capture(shell_with_recording_pub):
    """`register_hook`/`unregister_hook` are the same defect, reached from user
    code rather than from dispatch."""
    shell, real = shell_with_recording_pub

    def hook(msg):
        return msg

    with StatementProcessor._make_capture_ctx(stream_output=False, skip_capture=False):
        shell.display_pub.register_hook(hook)
        shell.display_pub.unregister_hook(hook)

    assert real.hooks == []


def test_capture_still_captures(shell_with_recording_pub):
    """The control. Making the publisher kernel-safe must not stop it capturing
    -- that is the whole reason cash installs it, and a fix that quietly
    published straight through would pass every assertion above."""
    from IPython.display import display

    shell, real = shell_with_recording_pub
    with StatementProcessor._make_capture_ctx(
        stream_output=False, skip_capture=False
    ) as captured:
        print("to stdout")
        display({"text/plain": "rich"}, raw=True)

    assert captured.stdout == "to stdout\n"
    assert len(captured.outputs) == 1, f"display output was not captured: {captured.outputs}"
