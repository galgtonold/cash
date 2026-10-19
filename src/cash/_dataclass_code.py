"""The methods of cash's dataclasses, compiled once and kept across processes.

``@dataclass`` writes each class's ``__init__``, ``__repr__``, ``__eq__``,
``__hash__`` (and, frozen, ``__setattr__`` and ``__delattr__``) as source
text and compiles it with ``exec``. For the ~100 dataclasses ``import cash``
builds that compiling was about 0.05 s of every fresh process, more than a
quarter of the import.

While cash imports, `dataclasses` is handed an ``exec`` that compiles a given
text once and keeps the code object, by the text, in a file beside cash's own
bytecode (its ``__pycache__``, or under ``PYTHONPYCACHEPREFIX``), the way
Python keeps a module's ``.pyc``: nothing is written where cash keeps
results, and nothing when Python writes no bytecode. A later process runs
the kept code object. The text is the one `dataclasses` wrote for this
Python and these fields, and the code object is what compiling it gives
(same file name, no inherited flags), so every method is the one
``@dataclass`` makes, by construction. A text not kept yet is compiled as
before. The file is named by the Python's bytecode magic, so another Python
never reads it; an unreadable file is ignored and rewritten.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import importlib.util
import marshal
import os
import sys
import types
from pathlib import Path

__all__ = ["compiled_dataclass_methods"]

#: Set to ``0`` to compile every dataclass method as ``dataclasses`` does
#: (the behaviour test compares the two).
ENV_SWITCH = "CASH_DATACLASS_CODE_CACHE"

#: More texts than this in the file is not cash's own classes alone: it is
#: rewritten from what this import used.
_MAX_ENTRIES = 4096


def _cache_file() -> Path | None:
    """Beside this module's ``.pyc``; None when it has no source file to
    place one by (a sourceless install)."""
    try:
        pyc = importlib.util.cache_from_source(__file__)
    except (NotImplementedError, ValueError):  # no cache_tag, or a .pyc-only install
        return None
    tag = importlib.util.MAGIC_NUMBER.hex() + "-" + (sys.implementation.cache_tag or "py")
    return Path(pyc).parent / f"dataclass-code-{tag}.marshal"


class _KeptCode:
    """An ``exec`` for `dataclasses` that compiles each text once."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kept: dict[bytes, types.CodeType] = {}
        self.used: set[bytes] = set()
        self.added = False
        with contextlib.suppress(Exception):  # absent or unreadable: compiled as before
            loaded = marshal.loads(path.read_bytes())
            if isinstance(loaded, dict) and all(isinstance(code, types.CodeType) for code in loaded.values()):
                self.kept = loaded

    def __call__(self, source, globals=None, locals=None, /, **kwargs):
        if not isinstance(source, str) or globals is None or kwargs:
            return builtins.exec(source, globals, locals, **kwargs)
        key = hashlib.blake2b(source.encode("utf-8"), digest_size=16).digest()
        code = self.kept.get(key)
        if code is None:
            # As ``exec(text)`` from `dataclasses` compiles it: file
            # "<string>", and no future flags (that module has none).
            code = compile(source, "<string>", "exec", dont_inherit=True)
            self.kept[key] = code
            self.added = True
        self.used.add(key)
        return builtins.exec(code, globals, locals)

    def save(self) -> None:
        """Write the file when this import compiled a text it lacked."""
        if not self.added or sys.dont_write_bytecode:
            return
        kept = self.kept
        if len(kept) > _MAX_ENTRIES:
            kept = {key: code for key, code in kept.items() if key in self.used}
        with contextlib.suppress(Exception):  # read-only or full: the next process compiles again
            self.path.parent.mkdir(parents=True, exist_ok=True)
            import tempfile  # only to write the file: not on every import

            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".dataclass-code-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(marshal.dumps(kept))
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise


@contextlib.contextmanager
def compiled_dataclass_methods():
    """While inside, `dataclasses` runs kept code objects for the method
    texts it has seen before (see the module docstring)."""
    import dataclasses

    path = _cache_file()
    if path is None or os.environ.get(ENV_SWITCH) == "0" or "exec" in vars(dataclasses):
        yield
        return
    kept = _KeptCode(path)
    dataclasses.exec = kept  # found before the builtin by the module's own code
    try:
        yield
    finally:
        with contextlib.suppress(AttributeError):
            del dataclasses.exec
        kept.save()
