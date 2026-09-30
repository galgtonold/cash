"""Storing a result must not import pandas.

Choosing how to store a result once imported pandas just to ask whether the
value was a DataFrame, so the FIRST cached call in any process paid the whole
pandas import -- measured at 731ms for a function whose body was
``return n``, essentially all of it module loading.
"""

import subprocess
import sys
import textwrap


def _run(body: str) -> str:
    """Run *body* in a FRESH interpreter and return its stdout.

    A subprocess is the only honest instrument here: pytest has almost
    certainly imported pandas already, so an in-process check of
    ``sys.modules`` would pass no matter what the code does.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


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
