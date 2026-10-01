"""Storing a result must not import pandas.

Choosing how to store a result once imported pandas just to ask whether the
value was a DataFrame, so the FIRST cached call in any process paid the whole
pandas import -- measured at 731ms for a function whose body was
``return n``, essentially all of it module loading.
"""

import textwrap

from tests._scripts import run_python


def _run(body: str, cwd) -> str:
    """Run *body* in a FRESH interpreter and return its stdout.

    A subprocess is the only honest instrument here: pytest has almost
    certainly imported pandas already, so an in-process check of
    ``sys.modules`` would pass no matter what the code does.
    """
    proc = run_python("-c", textwrap.dedent(body), cwd=cwd)
    return proc.stdout.strip()


def test_a_cached_call_does_not_import_pandas(tmp_path):
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
        """,
        tmp_path,
    )
    assert out == "False", "a cached call imported pandas"
