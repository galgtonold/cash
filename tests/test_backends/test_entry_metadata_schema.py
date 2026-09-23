"""Backends read only the metadata keys `EntryMetadata` documents.

Metadata is a plain dict, so nothing else stops a backend from growing a new
ad-hoc key that no one can find. This reads every string-keyed access on a
metadata dict in ``cash/backends`` and checks it against the schema.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import cash.backends
from cash.backends._base import EntryMetadata

BACKENDS = Path(cash.backends.__file__).parent

#: Expressions that hold entry metadata in the backends' code.
_METADATA_HOLDER = re.compile(r"meta|^entry$|^existing$|^known$|\[0\]$")


def _metadata_keys_read() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(BACKENDS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                holder, key = ast.unparse(node.func.value), node.args[0].value
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                holder, key = ast.unparse(node.value), node.slice.value
            else:
                continue
            if isinstance(key, str) and _METADATA_HOLDER.search(holder) and "__dict__" not in holder:
                found.setdefault(key, set()).add(path.name)
    return found


def test_every_metadata_key_a_backend_reads_is_documented():
    documented = set(EntryMetadata.__annotations__)
    undocumented = {k: sorted(v) for k, v in _metadata_keys_read().items() if k not in documented}
    assert not undocumented, f"add these to EntryMetadata, with what they mean: {undocumented}"


def test_the_scan_sees_the_keys_it_is_meant_to():
    """Guards the scan itself: an access pattern it stopped recognising would
    make the test above pass on nothing."""
    keys = _metadata_keys_read()
    assert {"ttl", "execution_time", "force_persist", "referenced", "version_slot"} <= set(keys)
