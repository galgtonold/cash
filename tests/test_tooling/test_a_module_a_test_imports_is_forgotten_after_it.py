"""A test takes a module out of ``sys.modules`` through ``forget_module``.

``monkeypatch.delitem(sys.modules, name, raising=False)`` undoes nothing when
the name was not there, so the module the test then imports stays loaded for
every later test on the worker (`tests._modules.forget_module`). Deleted after
the import, it is put back at the end. This keeps the pattern from coming back.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

# What the check looks for; `_modules.py` and this file name it on purpose.
LEAKY = re.compile(r"\.delitem\(\s*sys\." + "modules")


def test_no_test_deletes_from_sys_modules_but_forget_module():
    found = [
        f"{path.relative_to(TESTS)}:{n}"
        for path in sorted(TESTS.rglob("*.py"))
        if path.name not in ("_modules.py", Path(__file__).name)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if LEAKY.search(line)
    ]
    assert not found, f"use tests._modules.forget_module instead: {found}"


def test_the_pattern_catches_the_leaky_call():
    assert LEAKY.search("    monkeypatch.delitem(sys.modules, name, raising=False)")
    assert not LEAKY.search("    forget_module(monkeypatch, name)")
