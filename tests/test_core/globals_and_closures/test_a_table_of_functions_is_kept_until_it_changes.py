"""A table of functions read on every call is keyed once per version of
it, and every change to what its functions run is still seen.

Each function in a dispatch table or step list was analysed again, its code
identified and what it reads folded, on every hit; past 500 functions the
analysis memo emptied itself on every call: 0.3 s to 1.6 s a hit for a body
of a microsecond. A hit now keeps what was built from the table while the
table, each function's code and defaults and every global those functions
name are the same objects (`decorator.code_tables.CodeTable`) -- counted in
bytes hashed, not timed. Rebinding a constant the functions read, putting
another function in the table or giving one new code each recompute, and
the key of a kept hit is the key built afresh.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cash import Cash
from tests._work_counts import hashed_bytes


def _module(tmp_path, monkeypatch, name, n):
    lines = ["K = 2"]
    lines += [f"def h{i}(x):\n    return x * K + {i}" for i in range(n)]
    lines.append("def other(x):\n    return -x")
    lines.append("HANDLERS = {" + ", ".join(f"'h{i}': h{i}" for i in range(n)) + "}")
    lines.append("STEPS = [" + ", ".join(f"h{i}" for i in range(n)) + "]")
    (tmp_path / f"{name}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _reader(module, shape):
    handlers, steps = module.HANDLERS, module.STEPS

    def by_closure(name, x):
        return handlers[name](x)

    def by_module_attr(name, x):
        return module.HANDLERS[name](x)

    def by_step_list(name, x):
        return steps[int(name[1:])](x)

    return {"closure": by_closure, "module_attr": by_module_attr, "step_list": by_step_list}[shape]


def _hit_bytes(f):
    with hashed_bytes() as hashed:
        f("h3", 1)
    return hashed.bytes


_SHAPES = ["closure", "module_attr", "step_list"]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("shape", _SHAPES)
def test_a_hit_does_not_grow_with_the_table(cash_instance, tmp_path, monkeypatch, shape):
    def hit_bytes(n):
        module = _module(tmp_path, monkeypatch, f"_cash_tbl_size_{shape}_{n}", n)
        f = cash_instance.cache(_reader(module, shape))
        f("h3", 1)
        f("h3", 1)
        return _hit_bytes(f)

    small, big = hit_bytes(20), hit_bytes(700)
    assert big < 2 * small + 4096, (small, big)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("shape", _SHAPES)
def test_a_kept_table_still_recomputes_on_every_change(cash_instance, tmp_path, monkeypatch, shape):
    module = _module(tmp_path, monkeypatch, f"_cash_tbl_change_{shape}", 300)
    ran = []
    reader = _reader(module, shape)

    def counted(name, x):
        ran.append(name)
        return reader(name, x)

    f = cash_instance.cache(counted)
    for _ in range(3):
        assert f("h3", 1) == 5
    assert ran == ["h3"]
    # Kept: a hit no longer hashes the 300 functions' code.
    assert _hit_bytes(f) < 8192

    # The key of a kept hit is the key built afresh: another Cash over the
    # same entries, with nothing kept yet, finds the same entry.
    assert Cash(backend=cash_instance.backend, register_magic=False).cache(counted)("h3", 1) == 5
    assert ran == ["h3"]

    monkeypatch.setattr(module, "K", 10)
    assert f("h3", 1) == 13
    assert f("h3", 1) == 13

    monkeypatch.setattr(module.h3, "__code__", module.other.__code__)
    assert f("h3", 1) == -1

    monkeypatch.setattr(module.h3, "__code__", module.h4.__code__)
    if shape == "step_list":
        module.STEPS[3] = module.other
    else:
        module.HANDLERS["h3"] = module.other
    assert f("h3", 1) == -1
    assert ran == ["h3"] * 4
