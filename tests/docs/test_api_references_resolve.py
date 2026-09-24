"""Every ``%cash_*`` magic and ``Cash.<name>`` the docs and docstrings name exists.

A docstring once told users to run ``%cash_clear`` and ``Cash().clear_cache()``,
neither of which had ever existed, so following the instructions failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("IPython")

from cash.backends import InMemoryBackend  # noqa: E402
from cash.core import Cash  # noqa: E402
from cash.notebook.ipython.magics import CashMagics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: ``%cash_on``, or a ``%%`` cell-magic form; not the ``%cash_*`` family wildcard.
_MAGIC = re.compile(r"(?<![%\w])%{1,2}(cash\w*)(?![\w*])")
#: ``Cash.clear_all``, ``Cash().register_magic``.
_METHOD = re.compile(r"\bCash(?:\(\))?\.([A-Za-z_]\w*)")

#: Stale references still to fix. Drop each one from here once it is fixed;
#: none of them is an exception to keep.
#: The decorator's internals moved off `Cash` onto the objects it wires
#: together, while these pages were being rewritten elsewhere; the prose
#: is renamed with that rewrite.
_KNOWN_STALE: set[tuple[str, str]] = {
    ("docs/how-it-works/cache-keys-and-lineage.md", "Cash._hash_arg_payload"),
    ("docs/how-it-works/decorator-path.md", "Cash._decorator_call_log"),
    ("docs/tutorials/feature-guides/controlling-cache-behavior.md", "Cash._entry_expired"),
    ("docs/tutorials/feature-guides/custom-file-sources.md", "Cash._explain_call"),
    ("docs/tutorials/feature-guides/debugging-and-monitoring.md", "Cash._explain_call"),
    ("docs/tutorials/feature-guides/dynamic-dependencies.md", "Cash._resolve_dynamic_dependencies"),
    ("docs/tutorials/feature-guides/dynamic-dependencies.md", "Cash._build_key"),
    ("docs/tutorials/feature-guides/iterator-caching.md", "Cash._stream_and_store"),
    ("docs/tutorials/feature-guides/iterator-caching.md", "Cash._wrap_iterator_hit"),
    ("docs/tutorials/feature-guides/iterator-caching.md", "Cash._chunks_are_intact"),
    ("docs/tutorials/feature-guides/iterator-caching.md", "Cash._write_one_chunk"),
    ("docs/tutorials/feature-guides/iterator-caching.md", "Cash._attach_lineage"),
    ("docs/tutorials/feature-guides/thread-safety.md", "Cash._reread"),
    ("docs/tutorials/feature-guides/thread-safety.md", "Cash._warn_lock_failed"),
}

#: A magic named in prose as an example of a *different* name.
_NOT_A_REFERENCE: set[tuple[str, str]] = set()


def _sources() -> list[Path]:
    return sorted(
        [*(ROOT / "src" / "cash").rglob("*.py"), *(ROOT / "docs").rglob("*.md"), ROOT / "README.md"],
    )


def _unresolved() -> list[str]:
    magics = set(CashMagics.magics["line"]) | set(CashMagics.magics["cell"])
    instance = Cash(backend=InMemoryBackend(), register_magic=False)
    bad = []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            for m in _MAGIC.finditer(line):
                if m[1] not in magics and (rel, m[0]) not in _KNOWN_STALE | _NOT_A_REFERENCE:
                    bad.append(f"{rel}:{lineno}: {m[0]}")
            for m in _METHOD.finditer(line):
                if not hasattr(instance, m[1]) and (rel, f"Cash.{m[1]}") not in _KNOWN_STALE:
                    bad.append(f"{rel}:{lineno}: Cash.{m[1]}")
    return bad


def test_every_named_magic_and_cash_method_exists():
    assert _unresolved() == []
