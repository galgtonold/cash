"""The signature summary in docs/api/cash.md matches the real signatures.

It is a ``text`` fence, so nothing executes it, and a claim anchor only says
that *something* in the function changed. This compares what the reader is
told -- every parameter, in order, with its default -- against ``inspect``.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import cash

pytestmark = pytest.mark.core

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "api" / "cash.md"


def _documented() -> dict[str, list[tuple[str, str]]]:
    text = _PAGE.read_text(encoding="utf-8")
    section = text.split("## Signatures at a glance", 1)[1]
    fence = re.search(r"```text\n(.*?)```", section, re.S).group(1)
    out = {}
    for name, body in re.findall(r"^([\w.]+)\((.*?)\)\s*$", fence, re.S | re.M):
        call = ast.parse(f"f({body.replace('*, ', '')})", mode="eval").body
        params = [("**" + k.value.id, "") if k.arg is None else
                  (k.arg, repr(ast.literal_eval(k.value))) for k in call.keywords]
        out[name] = [p for p in params if p[0]]
    return out


def _actual(fn) -> list[tuple[str, str]]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        if p.name == "self":
            continue
        if p.kind is p.VAR_KEYWORD:
            out.append(("**" + p.name, ""))
        else:
            out.append((p.name, repr(p.default)))
    return out


@pytest.mark.parametrize("name, fn", [
    ("Cash", cash.Cash.__init__),
    ("Cash.cache", cash.Cash.cache),
])
def test_documented_signature_matches(name, fn):
    documented = _documented()
    assert name in documented, f"{name} is missing from the summary"
    assert documented[name] == _actual(fn)
