"""cash's dataclasses behave the same whether their methods were compiled
in this process or kept from an earlier one.

``import cash`` hands `dataclasses` an ``exec`` that keeps the compiled
method code in a file and runs it in later processes (`cash._dataclass_code`).
Three fresh processes import cash: with the file switched off (every method
compiled as `dataclasses` does), with no file yet (the file is written), and
with the file (methods come from it). For every cash dataclass the three
agree on each generated method's code and on what equality, hash, repr and
assignment to a frozen instance do.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

FINGERPRINT = textwrap.dedent(
    """
    import dataclasses, json, re, sys, types
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.instance().run_cell("import cash")  # as in a notebook: the whole engine loads

    def same(text):  # an address differs from process to process
        return re.sub(r" at 0x[0-9a-f]+", " at ADDRESS", text)

    def code(c):
        return [
            c.co_qualname if hasattr(c, "co_qualname") else c.co_name,
            c.co_code.hex(), c.co_filename, c.co_firstlineno, c.co_flags,
            c.co_argcount, c.co_kwonlyargcount, list(c.co_names), list(c.co_varnames),
            list(c.co_freevars), list(c.co_cellvars),
            [code(k) if isinstance(k, types.CodeType) else sorted(k) if isinstance(k, frozenset) else repr(k)
             for k in c.co_consts],
        ]

    def method(cls, name):
        f = cls.__dict__.get(name, "absent")
        if not isinstance(f, types.FunctionType):
            return repr(f) if f is None or f == "absent" else type(f).__name__
        return [f.__qualname__, same(repr(f.__defaults__)), same(repr(f.__kwdefaults__)),
                [type(c.cell_contents).__name__ for c in f.__closure__ or ()], code(f.__code__)]

    def outcome(call):
        try:
            return ["ok", call()]
        except Exception as e:
            return ["raised", type(e).__name__]

    def instance(cls, offset):
        obj = object.__new__(cls)
        for i, f in enumerate(dataclasses.fields(cls)):
            object.__setattr__(obj, f.name, i + offset)
        return obj

    def behaviour(cls):
        a, b, c = instance(cls, 0), instance(cls, 0), instance(cls, 1)
        first = dataclasses.fields(cls)[0].name if dataclasses.fields(cls) else "x"
        return {
            "equal": outcome(lambda: a == b),
            "unequal": outcome(lambda: a == c),
            "hash": outcome(lambda: hash(a) == hash(b)),
            "repr": outcome(lambda: same(repr(a))),
            "assign": outcome(lambda: setattr(a, first, 99)),
            "delete": outcome(lambda: delattr(b, first)),
        }

    out = {}
    for mod in list(sys.modules.values()):
        if not (getattr(mod, "__name__", "") or "").startswith("cash"):
            continue
        for cls in list(vars(mod).values()):
            if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
                continue
            if not cls.__module__.startswith("cash"):
                continue
            name = cls.__module__ + "." + cls.__qualname__
            params = cls.__dataclass_params__
            out[name] = {
                "params": repr(params),
                "match_args": repr(getattr(cls, "__match_args__", None)),
                "methods": {m: method(cls, m) for m in (
                    "__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__",
                    "__lt__", "__le__", "__gt__", "__ge__", "__getstate__", "__setstate__")},
                "behaviour": behaviour(cls),
            }
    print(json.dumps(out, sort_keys=True))
    """
)


def _fingerprint(cache: Path, *, kept: bool) -> dict:
    env = dict(os.environ)
    # The per-user cache root on every platform (see `per_user_cache_root`).
    env.update(HOME=str(cache), XDG_CACHE_HOME=str(cache), LOCALAPPDATA=str(cache))
    env["CASH_DATACLASS_CODE_CACHE"] = "1" if kept else "0"
    done = subprocess.run(
        [sys.executable, "-c", FINGERPRINT], env=env, capture_output=True, text=True, encoding="utf-8", timeout=300
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def _kept_files(cache: Path) -> list[Path]:
    return sorted(cache.rglob("dataclass-code-*.marshal"))


def test_kept_methods_match_freshly_compiled_ones(tmp_path):
    compiled = _fingerprint(tmp_path / "off", kept=False)
    assert not _kept_files(tmp_path / "off"), "switched off, no file is written"
    assert len(compiled) > 80, "cash's dataclasses were not found"
    frozen = [name for name, cls in compiled.items() if "frozen=True" in cls["params"]]
    assert frozen and all(compiled[n]["behaviour"]["assign"] == ["raised", "FrozenInstanceError"] for n in frozen)

    first = _fingerprint(tmp_path / "kept", kept=True)
    (written,) = _kept_files(tmp_path / "kept")
    stamp = written.stat().st_mtime_ns
    second = _fingerprint(tmp_path / "kept", kept=True)
    assert written.stat().st_mtime_ns == stamp, "every method came from the file, yet it was written again"

    for name in compiled:
        assert first.get(name) == compiled[name], f"{name}: compiling while writing the file changed it"
        assert second.get(name) == compiled[name], f"{name}: methods from the file differ"
    assert set(first) == set(second) == set(compiled)


def test_an_unreadable_file_is_compiled_past_and_rewritten(tmp_path):
    _fingerprint(tmp_path, kept=True)
    (written,) = _kept_files(tmp_path)
    written.write_bytes(b"not marshal data")
    compiled = _fingerprint(tmp_path / "off", kept=False)
    assert _fingerprint(tmp_path, kept=True) == compiled
    assert written.read_bytes() != b"not marshal data"
