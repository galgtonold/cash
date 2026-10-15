"""A code object's instructions, read straight from ``co_code``.

``dis.get_instructions`` builds an ``Instruction`` per instruction, with a
label map, line numbers, positions and a printable ``argrepr`` nobody here
reads: about 2.5 us an instruction, and cash scans the code of every
statement it runs and of every function a call reaches. `instructions`
reads the same bytes in one loop, about a tenth of that, and gives each
instruction's ``(opname, argval)`` as ``dis`` does: the same instructions
in the same order (``EXTENDED_ARG`` included, inline ``CACHE`` entries
skipped), every ``EXTENDED_ARG`` prefix folded into the argument after
it, and the same ``argval`` -- the constant, the name (``LOAD_GLOBAL``'s
and ``LOAD_ATTR``'s index shifted as 3.11 and 3.12 store them), the
variable or pair of variables, the comparison, the jump target.

The decoding is built once from ``dis``'s own tables (which opcodes take an
argument, how many cache entries each has, which table the argument
indexes), so it follows the running Python. A test compares it with
``dis.get_instructions`` over the code of the standard library.
"""

from __future__ import annotations

import dis
import opcode
import sys
import types
from typing import Any

_V = sys.version_info[:2]

#: Instructions callers ask `argvals` for.
NAME_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
GLOBAL_WRITES = frozenset({"STORE_GLOBAL", "DELETE_GLOBAL"})
DEREF_WRITES = frozenset({"STORE_DEREF", "DELETE_DEREF"})

#: How an argument is resolved into ``argval`` (`_KIND`).
_RAW, _CONST, _CONST_UNKNOWN, _NAME, _JREL, _JABS, _LOCAL, _FREE, _PAIR, _COMPARE, _FORMAT, _CONVERT = range(12)

_ALL_OPNAME: list[str] = list(getattr(dis, "_all_opname", dis.opname))
_ALL_OPMAP: dict[str, int] = dict(getattr(dis, "_all_opmap", dis.opmap))
_DEOPTMAP: dict[str, str] = dict(getattr(dis, "deoptmap", {}) or {})
_EXTENDED_ARG = dis.EXTENDED_ARG
_INT_OVERFLOW = getattr(dis, "_INT_OVERFLOW", 2**32)


def _deop(op: int) -> int:
    name = _ALL_OPNAME[op]
    return _ALL_OPMAP[_DEOPTMAP[name]] if name in _DEOPTMAP else op


def _cache_entries(op: int) -> int:
    if _V < (3, 11):
        return 0
    size = getattr(dis, "_get_cache_size", None)
    if size is not None:
        return size(_ALL_OPNAME[op])
    return dis._inline_cache_entries[op]  # type: ignore[attr-defined]


def _has_arg(op: int) -> bool:
    if _V >= (3, 12):
        return op in opcode.hasarg
    return op >= dis.HAVE_ARGUMENT


def _kind(op: int) -> tuple[int, int]:
    """``(kind, shift)`` of *op*'s argument, in the order ``dis`` asks."""
    name = _ALL_OPNAME[op]
    if op in dis.hasconst:
        if _V == (3, 11) and name != "LOAD_CONST":
            return _CONST_UNKNOWN, 0  # 3.11 resolves only LOAD_CONST's
        return _CONST, 0
    if op in dis.hasname:
        if name == "LOAD_GLOBAL" and _V >= (3, 11):
            return _NAME, 1
        if name == "LOAD_ATTR" and _V >= (3, 12):
            return _NAME, 1
        if name == "LOAD_SUPER_ATTR":
            return _NAME, 2
        return _NAME, 0
    if _V >= (3, 13):
        if op in dis.hasjump or op in dis.hasexc:
            return (_JABS, 0) if op in dis.hasjabs else (_JREL, 0) if op in dis.hasjrel else (_RAW, 0)
        if name in (
            "LOAD_FAST_LOAD_FAST",
            "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
            "STORE_FAST_LOAD_FAST",
            "STORE_FAST_STORE_FAST",
        ):
            return _PAIR, 0
        if op in dis.haslocal or op in dis.hasfree:
            return _LOCAL, 0
        if op in dis.hascompare:
            return _COMPARE, 5
        if name == "CONVERT_VALUE":
            return _CONVERT, 0
        return _RAW, 0
    if op in dis.hasjabs:
        return _JABS, 0
    if op in dis.hasjrel:
        return _JREL, 0
    if _V >= (3, 11):
        if op in dis.haslocal or op in dis.hasfree:
            return _LOCAL, 0
        if op in dis.hascompare:
            return _COMPARE, 4 if _V >= (3, 12) else 0
    else:
        if op in dis.haslocal:
            return _LOCAL, 0
        if op in dis.hascompare:
            return _COMPARE, 0
        if op in dis.hasfree:
            return _FREE, 0
    if name == "FORMAT_VALUE":
        return _FORMAT, 0
    return _RAW, 0


def _backward(op: int) -> bool:
    is_backward = getattr(dis, "_is_backward_jump", None)
    return bool(is_backward(op)) if is_backward is not None else False


# Per byte value: what ``co_code`` can hold. Opcodes above 255 are pseudo
# instructions the compiler removes before the bytes are written.
_OPNAME = [_ALL_OPNAME[op] for op in range(256)]
_DEOP = [_deop(op) for op in range(256)]
_CACHES = [_cache_entries(_DEOP[op]) for op in range(256)]
_HASARG = [_has_arg(_DEOP[op]) for op in range(256)]
_KINDS = [_kind(_DEOP[op]) for op in range(256)]
_BACKWARD = [_backward(_DEOP[op]) for op in range(256)]
#: Bytes a relative jump counts from past its own unit: the cache entries
#: after it on 3.12 and later, none before.
_JUMP_CACHES = [2 * _CACHES[op] if _V >= (3, 12) else 0 for op in range(256)]
_CMP_OP = tuple(dis.cmp_op)
_FORMAT_CONVERTERS = getattr(dis, "FORMAT_VALUE_CONVERTERS", ())
_CONVERTED = (None, str, repr, ascii)
_UNKNOWN = getattr(dis, "UNKNOWN", None)


def instructions(code: types.CodeType) -> list[tuple[str, Any]]:
    """``[(ins.opname, ins.argval) for ins in dis.get_instructions(code)]``,
    read from ``co_code`` in one loop. *code* only, not its nested code."""
    return _scan(code, None)


def _scan(code: types.CodeType, keep: frozenset[int] | None) -> list[tuple[str, Any]]:
    """`instructions`, only those whose byte is in *keep* when it is given."""
    raw = code.co_code
    consts = code.co_consts
    names = code.co_names
    if _V >= (3, 11):
        varname = code._varname_from_oparg  # type: ignore[attr-defined]
        cells: tuple = ()
    else:
        varnames = code.co_varnames
        varname = varnames.__getitem__
        cells = code.co_cellvars + code.co_freevars
    opnames, deops, caches_of, hasarg, kinds = _OPNAME, _DEOP, _CACHES, _HASARG, _KINDS
    out: list[tuple[str, Any]] = []
    append = out.append
    extended = 0
    caches = 0
    for i in range(0, len(raw), 2):
        if caches:
            caches -= 1
            continue
        op = raw[i]
        deop = deops[op]
        caches = caches_of[deop]
        if not hasarg[deop]:
            extended = 0
            if keep is None or op in keep:
                append((opnames[op], None))
            continue
        arg = raw[i + 1] | extended
        if deop == _EXTENDED_ARG:
            extended = arg << 8
            if extended >= _INT_OVERFLOW:
                extended -= 2 * _INT_OVERFLOW
        else:
            extended = 0
        if keep is not None and op not in keep:
            continue
        kind, shift = kinds[deop]
        if kind == _NAME:
            argval: Any = names[arg >> shift]
        elif kind == _RAW:
            argval = arg
        elif kind == _LOCAL:
            argval = varname(arg)
        elif kind == _CONST:
            argval = consts[arg]
        elif kind == _JREL:
            argval = i + 2 + (-arg if _BACKWARD[deop] else arg) * 2 + _JUMP_CACHES[deop]
        elif kind == _PAIR:
            argval = (varname(arg >> 4), varname(arg & 15))
        elif kind == _COMPARE:
            argval = _CMP_OP[arg >> shift]
        elif kind == _JABS:
            argval = arg * 2
        elif kind == _FREE:
            argval = cells[arg]
        elif kind == _FORMAT:
            argval = (_FORMAT_CONVERTERS[arg & 0x3][0], bool(arg & 0x4))
        elif kind == _CONVERT:
            argval = _CONVERTED[arg]
        else:  # _CONST_UNKNOWN
            argval = _UNKNOWN
        append((opnames[op], argval))
    return out


def opnames(code: types.CodeType) -> set[str]:
    """The names of the instructions *code* holds (not its nested code)."""
    raw = code.co_code
    found: set[int] = set()
    caches = 0
    deops, caches_of = _DEOP, _CACHES
    for i in range(0, len(raw), 2):
        if caches:
            caches -= 1
            continue
        op = raw[i]
        found.add(op)
        caches = caches_of[deops[op]]
    return {_OPNAME[op] for op in found}


def argvals(code: types.CodeType, wanted: frozenset[str]) -> list[Any]:
    """The ``argval`` of each instruction of *code* whose opname is in
    *wanted*, in order: ``[ins.argval for ins in dis.get_instructions(code)
    if ins.opname in wanted]``, resolving only those."""
    keep = _BYTES.get(wanted)
    if keep is None:
        keep = _BYTES[wanted] = frozenset(op for op in range(256) if _OPNAME[op] in wanted)
    return [argval for _name, argval in _scan(code, keep)]


#: ``opnames -> the byte values named so`` (`argvals`): a few constant sets.
_BYTES: dict[frozenset[str], frozenset[int]] = {}
