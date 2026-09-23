"""cash's I/O wrappers must look like the functions they wrap.

While cash watches reads and side effects it replaces ``json.load``,
``pandas.read_csv``, ``socket.socket.connect`` and friends with wrappers. A
wrapper that does not carry its original's metadata turns
``inspect.signature(json.load)`` into ``(*args, **kwargs)`` and empties
``help(pd.read_csv)`` and Jupyter's shift-tab tooltip, for the whole process.
"""

from __future__ import annotations

import builtins
import inspect
import json
import os
import pickle
import socket
import subprocess
import sys
import textwrap

import pytest

from cash.effect_observer import EffectObserver
from cash.tracking.file_tracker import FileAccessTracker


def _targets():
    import glob
    import io
    import linecache
    import pathlib

    import numpy
    import pandas

    targets = [
        (builtins, "open"),
        (io, "open"),
        (json, "load"),
        (pickle, "load"),
        (os, "listdir"),
        (os, "scandir"),
        (os.path, "exists"),
        (glob, "glob"),
        (linecache, "getlines"),
        (pandas, "read_csv"),
        (pandas, "read_parquet"),
        (numpy, "load"),
        (numpy, "loadtxt"),
        (pathlib.Path, "stat"),
        (socket.socket, "connect"),
        (subprocess.Popen, "__init__"),
    ]
    try:
        import pyarrow.parquet

        targets.append((pyarrow.parquet, "read_table"))
    except ImportError:
        pass
    return targets


def _metadata(fn):
    try:
        signature = str(inspect.signature(fn))
    except (TypeError, ValueError):
        signature = None
    return {
        "signature": signature,
        "doc": fn.__doc__,
        "name": getattr(fn, "__name__", None),
        "qualname": getattr(fn, "__qualname__", None),
    }


@pytest.mark.parametrize(
    "index", range(len(_targets())), ids=[f"{getattr(o, '__name__', o)}.{n}" for o, n in _targets()]
)
def test_a_wrapped_callable_keeps_its_signature_and_docs(index):
    owner, name = _targets()[index]
    with FileAccessTracker(), EffectObserver():
        current = getattr(owner, name)
        original = getattr(current, "_original_func", None)
        if original is None:
            pytest.skip(f"{name} is not wrapped in this configuration")
        assert _metadata(current) == _metadata(original)
        assert getattr(current, "__wrapped__", None) is original


def test_help_and_signature_are_unchanged_by_cash_in_a_fresh_process():
    """The user's view, end to end: the same answers before and after cash watches."""
    script = textwrap.dedent(
        """
        import inspect, json, pydoc
        import pandas as pd

        def view():
            return [
                str(inspect.signature(json.load)),
                str(inspect.signature(pd.read_csv)),
                pydoc.render_doc(json.load, renderer=pydoc.plaintext),
            ]

        before = view()

        import cash
        from cash.effect_observer import EffectObserver
        from cash.tracking.file_tracker import FileAccessTracker

        with FileAccessTracker(), EffectObserver():
            during = view()
        after = view()
        assert before == during == after, (before, during, after)
        assert "fp" in before[0] and "filepath_or_buffer" in before[1]
        """
    )
    env = dict(os.environ)
    import cash

    src = os.path.dirname(os.path.dirname(os.path.abspath(cash.__file__)))  # the cash under test
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
