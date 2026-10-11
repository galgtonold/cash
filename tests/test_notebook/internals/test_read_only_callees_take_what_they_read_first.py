"""The fast path in `call_arguments` that skips resolving a library call
with fewer than two positional arguments rests on every read-only function
taking what it reads first, never a write parameter."""

from __future__ import annotations

import sys

import pytest

from cash.analysis.read_only_callees import _READ_ONLY_CALLEES, _write_positions

pytest.importorskip("numpy")
pytest.importorskip("pandas")


def test_no_read_only_function_takes_a_write_parameter_first():
    for module_name, names in _READ_ONLY_CALLEES.items():
        module = sys.modules.get(module_name)
        for name in names if module else ():
            positions = _write_positions(getattr(module, name))
            assert positions is None or 0 not in positions, f"{module_name}.{name}"
