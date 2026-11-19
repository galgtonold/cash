"""What a library method reads off ``self`` is read off its raw bytecode.

A traced miss asks, for each library function it runs the first time in a
process, which attributes it loads off its first argument
(`library_reads.self_names`). Read through ``dis.get_instructions``, which
builds an object per instruction, that cost 0.24 ms per pandas method: 0.1 s
on top of a 0.02 s GeoDataFrame constructor in the first cached call of a
process. The names found are the ones ``dis`` finds, and a load whose name
index needs an ``EXTENDED_ARG`` is found too.
"""

from __future__ import annotations

import dis
import email.message
import json.decoder
import logging
import sys
import types

import pytest

from cash.decorator import library_reads

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="the raw bytecode read is 3.12+")


def _by_dis(code):
    """The names ``self_names`` stands for, read with ``dis``."""
    names = set()
    first = code.co_varnames[0]
    on_first = False
    for ins in dis.get_instructions(code):
        if on_first and ins.opname == "LOAD_ATTR" and not ins.argval.startswith("__"):
            names.add(ins.argval)
        if ins.opname.startswith("LOAD_FAST"):
            val = ins.argval
            on_first = (val[-1] if isinstance(val, tuple) else val) == first
        elif ins.opname != "EXTENDED_ARG":
            on_first = False
    return tuple(sorted(names))


def _codes():
    seen = {}

    def walk(code):
        if id(code) not in seen:
            seen[id(code)] = code
            for const in code.co_consts:
                if isinstance(const, types.CodeType):
                    walk(const)

    for module in (json.decoder, email.message, logging, dis):
        for value in vars(module).values():
            for fn in [value, *vars(value).values()] if isinstance(value, type) else [value]:
                if isinstance(fn, types.FunctionType):
                    walk(fn.__code__)
    return [code for code in seen.values() if code.co_argcount or code.co_posonlyargcount]


class Box:
    def pair(self, other):
        return other.width + self.height  # LOAD_FAST_LOAD_FAST on 3.13+: self on top

    def other_first(self, other):
        return self.width + other.depth


def _many_names_method():
    """A method whose ``self.target`` load needs an ``EXTENDED_ARG``: more
    than 127 names come before ``target`` in ``co_names``."""
    loads = "\n".join(f"    x.n{i}" for i in range(200))
    namespace = {}
    exec(f"def method(self, x):\n{loads}\n    return self.target\n", namespace)
    return namespace["method"].__code__


def test_the_names_are_the_ones_dis_finds():
    codes = _codes()
    assert len(codes) > 200
    for code in codes:
        assert library_reads.self_names(code) == _by_dis(code), code.co_qualname


def test_the_first_argument_on_top_of_a_packed_pair_is_followed():
    assert library_reads.self_names(Box.pair.__code__) == ("height",)
    assert library_reads.self_names(Box.other_first.__code__) == ("width",)


def test_a_load_whose_name_needs_an_extended_arg_is_found():
    code = _many_names_method()
    assert "EXTENDED_ARG" in {ins.opname for ins in dis.get_instructions(code)}
    assert library_reads.self_names(code) == ("target",)


def test_no_instruction_objects_are_built(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("dis.get_instructions was called")

    code = _many_names_method().replace(co_name="fresh")  # not memoised yet
    monkeypatch.setattr(dis, "get_instructions", refused)
    assert library_reads.self_names(code) == ("target",)
