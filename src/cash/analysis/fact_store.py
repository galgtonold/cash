"""Facts read off one function's tree, kept on disk, so a fresh process does
not work them out again for every helper it walks.

The helper walk asks of every body it audits, and `purity_flow.returns_fresh`
of every user helper a body calls, which of its calls may build a new object
(`fresh_call_candidates`), which names and calls hold a fresh object where
they are evaluated (`fresh_flow_nodes`, given which calls do), which names it
hands to other code (`_escaped_names`), its locals and its ``return``
statements. Each is a function of the function's tree alone, and of the
fresh calls handed in for the flow pass; what the names are bound to is
resolved live, on every walk, and only its outcome (which calls are fresh)
is part of the key. `tree_memo` keeps them for the process; this keeps them
across processes, in the same place and under the same rules as the module
scans of `scan_store`:

* the source TEXT the tree was parsed from is digested in full, so an edit
  (or a revert) is a different text;
* the Python and cash's own code are the store's folder
  (`scan_store.store_dir`);
* a node is named by its index in ``ast.walk`` order below the function's
  node, which one Python parsing one text gives the same in every process.

Only the trees `tree_memo.parse` hands out are stored, and nothing changes
them: a body the walk rewrites (`spell_static_dispatch` spells
``sys.modules["m"].g`` as ``m.g``) is a new tree, whose facts are worked
out for it alone. One file per source file (the path only names the file; the
text decides) holds the facts of each text of its functions, most recently
used last, at most `_MAX_TEXTS` of them. Written at the end of each walk
(`flush`); a file that cannot be read or does not hold what it should is a
miss. ``CASH_ANALYSIS_MEMO_VERIFY=1`` works every stored fact out again and
raises `StoredFactMismatch` when it differs.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from .. import _weak
from . import scan_store

__all__ = ["STATS", "StoredFactMismatch", "flush", "kept", "register", "status"]

T = TypeVar("T")

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 1
_KIND = "tree-facts"
#: Texts kept per file: the functions of one module, and their recent edits.
_MAX_TEXTS = 2048

STATS = {"loaded": 0, "computed": 0, "stored": 0, "refused": 0, "verified": 0}


class StoredFactMismatch(AssertionError):
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: a stored tree fact is not what working
    it out again gives."""


class _Place(NamedTuple):
    """Where a function's facts are stored: the file named for *path*, under
    the digest of the text it was parsed from, at *where* in that text."""

    path: str
    digest: str
    where: str


#: A function node of a registered tree -> its place.
_PLACES: _weak.WeakKeyDictionary = _weak.WeakKeyDictionary()
#: Trees already registered.
_REGISTERED: _weak.WeakKeyDictionary = _weak.WeakKeyDictionary()
#: path -> {digest: {key: encoded}}: what this process read or worked out.
_SLOTS: dict[str, dict[str, dict[str, Any]]] = {}
#: path -> digests with facts not written yet.
_PENDING: dict[str, set[str]] = {}
#: path -> digests used in this process (kept first when a file is full).
_USED: dict[str, set[str]] = {}


def status() -> dict:
    return dict(STATS)


def _store_dir() -> Path | None:
    return scan_store.store_dir()


def register(tree: ast.AST, src: str, path: str) -> None:
    """Store the facts of the functions in *tree*, parsed from *src* as read
    from the file at *path*. Once per tree; nothing when the store is off."""
    if path in ("", None) or tree in _REGISTERED or _store_dir() is None:
        return
    digest = hashlib.blake2b(src.encode("utf-8", "surrogatepass"), digest_size=20).hexdigest()
    try:
        _REGISTERED[tree] = digest
    except TypeError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _PLACES[node] = _Place(path, digest, f"{node.lineno}:{node.col_offset}")


def kept(
    origin: ast.AST,
    node: ast.AST,
    key: str | Callable[[], str],
    compute: Callable[[ast.AST], T],
    encode: Callable[[ast.AST, T], Any],
    decode: Callable[[ast.AST, Any], T],
) -> T:
    """``compute(node)``, from the store when an earlier process stored it
    for the text *origin* was parsed from. *node* is *origin* or the copy of
    it `tree_memo.stripped_def` made; *key* names the fact, and everything
    else *compute* reads (a callable is asked only when the fact is stored,
    and raises ValueError when it cannot be spelled: then it is not).
    *decode* raises ValueError (or a lookup error) on an answer that does
    not fit the tree: it is worked out again."""
    place = _PLACES.get(origin)
    if place is None:
        return compute(node)
    if callable(key):
        try:
            key = key()
        except ValueError:
            STATS["refused"] += 1
            return compute(node)
    full = f"{place.where}|{int(node is origin)}|{key}"
    facts = _slot(place.path).get(place.digest)
    if facts is not None and full in facts:
        try:
            answer = decode(node, facts[full])
        except (ValueError, TypeError, LookupError, AttributeError):
            pass
        else:
            STATS["loaded"] += 1
            _USED.setdefault(place.path, set()).add(place.digest)
            from . import tree_memo

            if tree_memo.VERIFY:
                STATS["verified"] += 1
                if compute(node) != answer:
                    raise StoredFactMismatch(f"stored tree fact {key!r} of {place.path} differs from working it out")
            return answer
    answer = compute(node)
    STATS["computed"] += 1
    try:
        encoded = encode(node, answer)
    except (ValueError, TypeError, LookupError):
        return answer
    _slot(place.path).setdefault(place.digest, {})[full] = encoded
    _PENDING.setdefault(place.path, set()).add(place.digest)
    _USED.setdefault(place.path, set()).add(place.digest)
    return answer


def _file(directory: Path, path: str) -> Path:
    slot = hashlib.blake2b(f"{_KIND}\0{path}".encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
    return directory / f"{slot}.json"


def _read_file(path: str) -> dict[str, dict[str, Any]]:
    """The texts stored for *path*: only well-formed ones."""
    directory = _store_dir()
    if directory is None:
        return {}
    data = scan_store.read_json(_file(directory, path))
    if not isinstance(data, dict) or data.get("kind") != _KIND or data.get("format") != _FORMAT:
        return {}
    if data.get("path") != path:
        return {}
    texts = data.get("texts")
    if not isinstance(texts, dict):
        return {}
    return {d: f for d, f in texts.items() if isinstance(d, str) and isinstance(f, dict)}


def _slot(path: str) -> dict[str, dict[str, Any]]:
    slot = _SLOTS.get(path)
    if slot is None:
        slot = _SLOTS[path] = _read_file(path)
    return slot


def flush() -> None:
    """Write what this process worked out since the last flush: merged into
    what is on disk now, most recently used texts kept. Best effort."""
    if not _PENDING:
        return
    pending = dict(_PENDING)
    _PENDING.clear()
    directory = _store_dir()
    if directory is None:
        return
    for path, digests in pending.items():
        mine = _SLOTS.get(path, {})
        merged = _read_file(path)
        for digest in digests:
            merged.setdefault(digest, {}).update(dict(mine.get(digest, {})))
        used = _USED.get(path, set())
        # Most recently used last, so a full file drops the ones no
        # process has met for longest.
        ordered = {d: f for d, f in merged.items() if d not in used}
        ordered.update((d, merged[d]) for d in merged if d in used)
        texts = dict(list(ordered.items())[-_MAX_TEXTS:])
        data = {"kind": _KIND, "format": _FORMAT, "path": path, "texts": texts}
        if scan_store.write_json(directory, _file(directory, path), data):
            STATS["stored"] += 1


# -- encodings: a node as its index in ``ast.walk`` order below the function --


def node_index(node: ast.AST) -> dict[int, int]:
    """id of each node below *node* (itself included) -> its index in
    ``ast.walk`` order."""
    return {id(n): i for i, n in enumerate(ast.walk(node))}


def indices_of(index: dict[int, int], chosen_ids: Any) -> list[int]:
    """The indices (`node_index`) of the nodes whose ids are *chosen_ids*, in
    their order; raises ValueError when one is not below the function."""
    try:
        return [index[i] for i in chosen_ids]
    except KeyError:
        raise ValueError("a node outside the function") from None


def at_indices(nodes: list[ast.AST], indices: Any, kind: type | tuple[type, ...] = ast.AST) -> list[ast.AST]:
    """The nodes at *indices*, each a *kind*; raises ValueError otherwise."""
    if not isinstance(indices, list):
        raise ValueError("not a list")
    out = []
    for i in indices:
        if type(i) is not int or not 0 <= i < len(nodes) or not isinstance(nodes[i], kind):
            raise ValueError("not a node of this function")
        out.append(nodes[i])
    return out


def names_decoded(_node: ast.AST, value: Any) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("not a list of names")
    return frozenset(value)


def names_encoded(_node: ast.AST, value: frozenset[str]) -> list[str]:
    return sorted(value)
