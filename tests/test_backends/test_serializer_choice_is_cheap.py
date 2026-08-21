"""Choosing a serializer must not import pandas.

``get_serializer`` runs on EVERY store. It used to ``import pandas`` just to
ask whether the value was a DataFrame, so the FIRST cached call in any
process paid the whole pandas import -- measured at 731ms for a function
whose body was ``return n``, essentially all of it module loading.

A DataFrame cannot exist unless pandas is already imported, so its absence
from ``sys.modules`` answers the question for free. These tests pin both
halves: the cheap path stays cheap, and a real DataFrame still reaches the
parquet serializer.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(body: str) -> str:
    """Run *body* in a FRESH interpreter and return its stdout.

    A subprocess is the only honest instrument here: pytest has almost
    certainly imported pandas already, so an in-process check of
    ``sys.modules`` would pass no matter what the code does.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


def test_storing_a_non_dataframe_does_not_import_pandas():
    out = _run(
        """
        import sys
        from cash.backends.serialization import get_serializer

        assert "pandas" not in sys.modules, "pandas imported merely by importing cash"
        for value in (42, [1, 2, 3], {"a": 1}, "text", None):
            get_serializer(value)
        print("pandas" in sys.modules)
        """
    )
    assert out == "False", "get_serializer imported pandas for a non-DataFrame value"


def test_a_cached_call_does_not_import_pandas():
    """End to end: the decorator's first store must not drag pandas in."""
    out = _run(
        """
        import sys, tempfile
        import cash

        c = cash.Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)

        @c.cache
        def f(n):
            return n + 1

        f(1)
        print("pandas" in sys.modules)
        """
    )
    assert out == "False", "a cached call imported pandas"


def test_a_dataframe_still_uses_the_parquet_serializer():
    """The control: the cheap check must not cost the feature."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from cash.backends.serialization import ParquetSerializer, get_serializer

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert isinstance(get_serializer(frame), ParquetSerializer)


def test_a_dataframe_round_trips():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from cash.backends.serialization import get_serializer

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    serializer = get_serializer(frame)
    restored = serializer.deserialize(serializer.serialize(frame))
    pd.testing.assert_frame_equal(frame, restored)
