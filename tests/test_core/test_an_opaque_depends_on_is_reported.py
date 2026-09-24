"""A ``depends_on=`` target cash can only key on by name is reported.

The warning used to sit behind ``if snapshot is None``, but the source hash
never returns None: for a builtin or a compiled-extension function it falls
back to the ``module.qualname`` string, which a rebuilt extension does not
change. So KEY-DEPENDS-ON-OPAQUE never fired, and the case docs/warnings.md
describes -- a declared dependency that can never invalidate anything -- went
unreported.
"""

from __future__ import annotations

import json
import math
import warnings

import pytest

from cash import Cash

pytestmark = pytest.mark.core


def _codes_for(c, dep, name):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ns: dict = {"c": c, "dep": dep}
        exec(f"@c.cache(depends_on=[dep])\ndef {name}(x):\n    return x\n", ns)
        ns[name](1)
    return [getattr(w.message, "code", None) for w in rec]


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.mark.parametrize("dep", [len, math.sqrt], ids=["builtin", "c-extension"])
def test_a_callable_without_source_or_bytecode_warns(c, dep):
    assert "KEY-DEPENDS-ON-OPAQUE" in _codes_for(c, dep, "f")


def _helper(x):
    return x


@pytest.mark.parametrize("dep", [json.loads, _helper], ids=["stdlib-python", "local"])
def test_a_readable_callable_does_not(c, dep):
    assert "KEY-DEPENDS-ON-OPAQUE" not in _codes_for(c, dep, "f")


def test_it_does_not_hide_another_warning_on_the_same_function(c):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        @c.cache(depends_on=[len], cache_if=lambda r: True, chunk_max_items=2)
        def gen(n):
            yield from range(n)

        assert list(gen(10)) == list(range(10))
    codes = {getattr(w.message, "code", None) for w in rec}
    assert {"KEY-DEPENDS-ON-OPAQUE", "CACHE-IF-BYPASSED"} <= codes
