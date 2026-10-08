"""A badge is published without flushing streams that hold nothing.

ipykernel's publisher flushes stdout and stderr before every display, two
round trips to its IO thread: ~0.5 ms of each badge render when the cell had
printed nothing, a third of drawing a trivial cell's badge. The flush is what
keeps a print ahead of the display that follows it, so it still happens when
a write is waiting.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("IPython")

from cash.notebook.ipython import badges


class _Publisher:
    """ipykernel's ZMQDisplayPublisher, as far as a badge sees it."""

    def __init__(self) -> None:
        self.flushes = 0
        self.published = 0

    def _flush_streams(self) -> None:
        self.flushes += 1

    def publish(self, data, metadata=None, *, transient=None, update=False) -> None:
        self._flush_streams()
        self.published += 1


class _Stream:
    def __init__(self, pending: bool) -> None:
        self._flush_pending = pending
        self._subprocess_flush_pending = False

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


@pytest.fixture
def presenter(monkeypatch):
    pub = _Publisher()
    shell = SimpleNamespace(display_pub=pub, execution_count=1, user_ns={})
    p = badges.BadgePresenter(shell, SimpleNamespace(backend=None))
    monkeypatch.setattr(
        badges,
        "display",
        lambda obj, display_id=None, update=False: pub.publish({"text/html": obj.data}, transient={}, update=update),
    )
    return p, pub


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_pending_write_is_flushed_before_the_badge(presenter, monkeypatch, stream):
    p, pub = presenter
    monkeypatch.setattr(sys, "stdout", _Stream(stream == "stdout"))
    monkeypatch.setattr(sys, "stderr", _Stream(stream == "stderr"))

    p._publish_html("<b>badge</b>", display_id="d1")

    assert (pub.published, pub.flushes) == (1, 1)


def test_idle_streams_are_not_flushed(presenter, monkeypatch):
    p, pub = presenter
    monkeypatch.setattr(sys, "stdout", _Stream(False))
    monkeypatch.setattr(sys, "stderr", _Stream(False))

    p._publish_html("<b>badge</b>", display_id="d1")
    p._publish_html("<b>badge</b>", display_id="d1", update_existing=True)

    assert (pub.published, pub.flushes) == (2, 0)
    # The publisher is left as it was: the next display flushes again.
    assert "_flush_streams" not in vars(pub)
    pub.publish({})
    assert pub.flushes == 1


def test_streams_that_are_not_ipykernels_are_flushed(presenter, monkeypatch):
    """A stream that does not say whether it holds anything is flushed."""
    p, pub = presenter
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(write=len, flush=lambda: None))

    p._publish_html("<b>badge</b>", display_id="d1")

    assert pub.flushes == 1
