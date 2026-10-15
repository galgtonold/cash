"""Without the C extension every value is hashed, and every digest and key
is the one the extension's fast path gives.

The extension is an accelerator: its absence (a pure-Python install,
``CASH_PURE_PYTHON=1``, a free-threaded build) only means no value is ever
"provably unchanged". A cache written by one install serves the other.
"""

from __future__ import annotations

import json

import pytest

from cash import value_watch
from cash.value_hash import compute_hash
from tests._scripts import run_python


def test_without_the_extension_every_digest_is_read_from_the_value(monkeypatch, counted):
    monkeypatch.setattr(value_watch, "_cw", None)
    monkeypatch.setattr(value_watch, "WATCHING", False)
    value = {"a": [1, 2, 3]}
    for _ in range(4):
        value_watch.digest(value, counted)
    assert counted.calls == 4
    assert value_watch.peek(value) is None
    assert value_watch.stats()["enabled"] is False
    monkeypatch.setattr(value_watch, "REASON", "CASH_PURE_PYTHON is set")
    value_watch.forget(value)
    value_watch.reset()


def test_compute_hash_without_the_extension_reads_the_value(monkeypatch):
    monkeypatch.setattr(value_watch, "WATCHING", False)
    calls = []
    real = value_watch.digest
    monkeypatch.setattr(value_watch, "digest", lambda *a, **k: calls.append(1) or real(*a, **k))
    compute_hash({"a": [1]})
    assert not calls


_SCRIPT = r"""
import json, sys
import cash
from cash import value_watch
from cash.value_hash import compute_hash
from cash.decorator.arg_hashing import plain_key_part

class P:
    def __init__(self, x):
        self.x = x

value = {"rows": [{"v": i, "s": str(i)} for i in range(3000)], "p": P([1, 2]), "t": (1, 2.5, None)}
plain = {"rows": [{"v": i} for i in range(3000)]}
out = []
for step in range(4):
    out.append(compute_hash(value))
    out.append(list(plain_key_part(plain)))
    if step == 2:
        value["rows"][5]["v"] = -1
        value["p"].x.append(3)
        plain["rows"].append({"v": 0})

c = cash.Cash()

@c.cache
def total(d):
    return len(d["rows"])

for _ in range(3):
    out.append(total.explain(plain).cache_key)
    total(plain)
print(json.dumps({"digests": out, "fast": value_watch.enabled(), "hits": value_watch.stats().get("hits", 0)}))
"""


@pytest.mark.skipif(not value_watch.enabled(), reason="compares with the fast path on")
def test_the_same_digests_and_keys_come_out_with_and_without_it(tmp_path):
    fast = json.loads(run_python("-c", _SCRIPT, cwd=tmp_path, env={"CASH_CWATCH_MIN_BYTES": "0"}).stdout)
    pure = json.loads(run_python("-c", _SCRIPT, cwd=tmp_path, env={"CASH_PURE_PYTHON": "1"}).stdout)
    assert fast["fast"] and not pure["fast"]
    assert fast["hits"] > 0, "the fast path was never taken"
    assert fast["digests"] == pure["digests"]
