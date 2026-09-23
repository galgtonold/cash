"""A stand-in for a user's own project module, imported by
``test_a_writer_is_seen_however_it_is_spelled``.

It has to be a real file on disk: the writer analysis reads the callee's
source with ``inspect``, so a synthesised module object would not do.
"""

from __future__ import annotations

import json


def export_summary(data, path):
    """The reported shape: a project-module export that replaces a file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def save_chart(fig, path):
    fig.savefig(path)
    return path


def tidy(values):
    """No write at all -- the control."""
    return sorted(values)


def note(msg):
    """An APPEND, which a cache hit is understood to skip, like a print."""
    with open("run.log", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
