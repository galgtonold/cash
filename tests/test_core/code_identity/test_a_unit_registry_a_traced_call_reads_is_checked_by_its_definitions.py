"""A pint unit registry that traced code reads as a global is checked by its
definitions.

xclim's indicators read the module global ``units``, a pint
``UnitRegistry``. The trace of a call keys each global the code that ran
reads by its value, and a registry had none it could read: the trace was
refused ("reads the global units, whose value cannot be checked"), so a
call that needs its trace ran uncached every time. A registry is now read
by its definitions, as an argument holding one is (`cash.pint_values`):
the same registry in another process checks the same, and one where a unit
means something else does not.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytest.importorskip("pint")

pytestmark = [pytest.mark.timeout(300)]

HANDLERS = textwrap.dedent("""
    import os

    import pint

    ureg = pint.UnitRegistry()
    ureg.define(f"widget = {os.environ['WIDGET']} * meter")


    def to_meters(n):
        return ureg.Quantity(n, "widget").to("meter").magnitude
""")

MAIN = textwrap.dedent("""
    import cash
    import handlers

    c = cash.Cash(cache_dir=".cash")


    @c.cache
    def convert(name, n):
        # Picks its code by name: served only on a trace of what it ran.
        return getattr(handlers, name)(n)


    info = convert.cache_info
    print(convert("to_meters", 5.0), info()["hits"], [w["code"] for w in info()["warnings"]])
""")

DIGEST = textwrap.dedent("""
    import sys

    import pint

    from cash.pint_values import registry_digest
    from cash.tracking.trace_items import value_digest

    ureg = pint.UnitRegistry()
    if len(sys.argv) > 1:
        ureg.define(sys.argv[1])
    before = registry_digest(ureg)
    digest = value_digest(ureg)
    print(digest, registry_digest(ureg) == before)
""")


def _run(tmp_path, widget):
    proc = run_python("main.py", cwd=tmp_path, env={"WIDGET": widget})
    value, hits, warnings = proc.stdout.strip().split(" ", 2)
    return float(value), int(hits), warnings


def test_the_next_process_hits_and_another_definition_misses(tmp_path):
    (tmp_path / "handlers.py").write_text(HANDLERS, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    assert _run(tmp_path, "2") == (10.0, 0, "[]")
    assert _run(tmp_path, "2") == (10.0, 1, "[]")
    assert _run(tmp_path, "3") == (15.0, 0, "[]")
    assert _run(tmp_path, "3") == (15.0, 1, "[]")


def _digest(tmp_path, seed, *define):
    (tmp_path / "digest.py").write_text(DIGEST, encoding="utf-8")
    proc = run_python("digest.py", *define, cwd=tmp_path, env={"PYTHONHASHSEED": str(seed)})
    digest, unchanged = proc.stdout.split()
    # Reading it adds nothing to it: an attribute read of a registry
    # (``nbytes``) parses the name as a unit and defines that unit.
    assert unchanged == "True"
    return digest


def test_a_registry_digests_by_its_definitions_in_every_process(tmp_path):
    digests = {_digest(tmp_path, seed) for seed in (1, 2)}
    assert len(digests) == 1
    assert not next(iter(digests)).startswith("opaque:")
    assert _digest(tmp_path, 3, "smoot = 1.7 * meter") not in digests
