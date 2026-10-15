"""`cash.analysis.bytecode` reads ``co_code`` itself instead of calling
``dis.get_instructions``, about ten times faster. It must give what ``dis``
gives: the same instructions in the same order, with the same ``argval``,
on every Python cash supports (``EXTENDED_ARG`` prefixes, 3.11+ inline
caches, the 3.12 and 3.13 name and pair encodings, 3.14's new loads).
Compared here over the code of a slice of the standard library, and over
code built to need ``EXTENDED_ARG``; and the scans cash makes with it give
what the same scans made with ``dis`` gave.
"""

from __future__ import annotations

import dis
import importlib
import inspect
import math
import types

import pytest

from cash.analysis import bytecode

_MODULES = (
    "argparse",
    "ast",
    "asyncio.tasks",
    "collections",
    "dataclasses",
    "decimal",
    "dis",
    "email.message",
    "enum",
    "fractions",
    "functools",
    "inspect",
    "json.decoder",
    "pathlib",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
    "zipfile",
)

#: Code that needs what a module may not hold: EXTENDED_ARG on names,
#: constants, locals and jumps; closures; comprehensions; ``super()``;
#: f-strings; ``match``; ``except*``; async; augmented and chained compares.
_SYNTHETIC = (
    "def many():\n"
    + "".join(f"    v{i} = g{i} + {i}.5 + c.a{i}\n" for i in range(300))
    + "    return v299\n"
    + """
def outer(x):
    k = 1
    def inner(y):
        nonlocal k
        k += y
        del x
        return [k * z for z in range(y) if z > k], {z: str(z) for z in 'ab'}
    return inner

class C(dict):
    def m(self, *a, **kw):
        return super().get(*a, **kw), f"{self!r:>10} {a!s} {kw!a}", 1 < len(a) <= 3

def matcher(v):
    match v:
        case {"k": [1, *rest]} if rest:
            return rest
        case C(a=1) | None:
            return 0
        case _:
            return -1

async def agen(xs):
    async with xs as s:
        async for x in s:
            yield x
    try:
        pass
    except* ValueError as e:
        raise e from None

def loop(n):
    i = 0
    while i < n:
        i += 1
        if i % 3 is not None and i not in (2, 4):
            continue
        for j in range(i):
            try:
                break
            finally:
                i -= 0
    return i
"""
)


def _codes(code: types.CodeType) -> list[types.CodeType]:
    found = [code]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found.extend(_codes(const))
    return found


def _all_codes() -> list[types.CodeType]:
    codes: list[types.CodeType] = []
    for name in _MODULES:
        module = importlib.import_module(name)
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            continue
        codes.extend(_codes(compile(source, module.__file__ or name, "exec")))
    try:
        codes.extend(_codes(compile(_SYNTHETIC, "<synthetic>", "exec")))
    except SyntaxError:  # 3.10: no except*
        codes.extend(_codes(compile(_SYNTHETIC.replace("except*", "except"), "<synthetic>", "exec")))
    return codes


CODES = _all_codes()


def _same(a: object, b: object) -> bool:
    if a is b:
        return True
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def test_every_instruction_and_argval_is_what_dis_gives():
    assert len(CODES) > 1000
    extended = 0
    for code in CODES:
        expected = [(ins.opname, ins.argval) for ins in dis.get_instructions(code)]
        got = bytecode.instructions(code)
        assert len(got) == len(expected), code
        for (name, value), (want_name, want_value) in zip(got, expected):
            assert name == want_name, code
            assert _same(value, want_value), (code, name, value, want_value)
        extended += sum(name == "EXTENDED_ARG" for name, _ in got)
    assert extended > 100  # the prefixes were exercised


@pytest.mark.parametrize(
    "wanted",
    [
        bytecode.NAME_LOADS,
        bytecode.GLOBAL_WRITES,
        bytecode.DEREF_WRITES,
        frozenset({"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR"}),
        frozenset({"LOAD_CONST", "LOAD_FAST", "JUMP_BACKWARD", "COMPARE_OP"}),
    ],
)
def test_argvals_are_the_argvals_dis_gives_for_those_instructions(wanted):
    for code in CODES:
        expected = [ins.argval for ins in dis.get_instructions(code) if ins.opname in wanted]
        got = bytecode.argvals(code, wanted)
        assert len(got) == len(expected)
        assert all(_same(a, b) for a, b in zip(got, expected)), code


def test_opnames_are_the_opnames_dis_gives():
    for code in CODES:
        assert bytecode.opnames(code) == {ins.opname for ins in dis.get_instructions(code)}, code


def test_the_scans_cash_makes_give_what_they_gave_with_dis():
    from cash.notebook import cache_key, call_key, callee_reach

    for code in CODES[:1500]:
        names = {ins.argval for ins in dis.get_instructions(code) if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
        for nested in _codes(code)[1:]:
            names |= {ins.argval for ins in dis.get_instructions(nested) if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
        assert call_key._code_names(code) == frozenset(names)
        loaded = frozenset(
            ins.argval
            for ins in dis.get_instructions(code)
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str)
        )
        assert callee_reach._loaded_globals(code) == loaded
        global_names = frozenset(
            ins.argval
            for ins in dis.get_instructions(code)
            if ins.opcode in frozenset(dis.hasname)
            and ins.opname not in cache_key._ATTRIBUTE_OPS
            and isinstance(ins.argval, str)
        )
        assert cache_key._global_names(code) == global_names
