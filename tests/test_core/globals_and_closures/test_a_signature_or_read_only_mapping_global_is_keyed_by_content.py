"""A ``Signature`` or a ``MappingProxyType`` is keyed by what it holds.

Reported against scanpy checked out as the user's code: ``neighbors`` takes
``metric_kwds=MappingProxyType({})`` as a default, and a helper captures an
``inspect.Signature`` of it. Neither has a pickle, so the default refused
every call (KEY-UNHASHABLE-DEFAULT) and a global holding the signature was
left out of the key (KEY-UNHASHABLE-GLOBAL), an edit to the default it
shows then served the result from before.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HELPERS = textwrap.dedent("""
    import inspect
    from types import MappingProxyType


    def scale(x, opts=MappingProxyType({"k": {K}})):
        return x * opts["k"]


    SIG = inspect.signature(scale)
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    import helpers


    @cash.cache
    def by_default(x):
        print("[RUN default]", file=sys.stderr)  # @cash:assume-safe
        return helpers.scale(x)


    @cash.cache
    def by_signature(x):
        print("[RUN signature]", file=sys.stderr)  # @cash:assume-safe
        return x * helpers.SIG.parameters["opts"].default["k"]


    print(by_default(3), by_signature(3))
""")


def _run(tmp_path, k):
    (tmp_path / "helpers.py").write_text(HELPERS.replace("{K}", str(k)), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path, env={"PYTHONDONTWRITEBYTECODE": "1"})
    return proc.stdout.strip(), proc.stderr


def test_both_are_cached_and_an_edit_to_what_they_hold_recomputes(tmp_path):
    out, err = _run(tmp_path, 2)
    assert out == "6 6" and "UNHASHABLE" not in err, err
    out, err = _run(tmp_path, 2)
    assert out == "6 6" and "[RUN" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, 5)
    assert out == "15 15", f"served the result from before the edit:\n{err}"
