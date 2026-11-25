"""Library code a rule was read in, recognised by its text.

A few rules rest on how a library pickles its objects: scikit-learn's
``BaseEstimator``, numpy's ``MaskedArray``. Each was written after reading
the functions that do it, and holds while they say what was read. Asking
for the release they were read in (``__version__ == "1.9.1"``) turned the
rule off on every other release, the next patch release included, although
the functions had not changed, and nothing said so.

So a rule asks for the functions themselves: the SHA-256 of their source
text, against the digests of the texts that were read (`RULES`). A release
that leaves them alone is accepted unseen; one that edits a line of them
turns the rule off until someone reads the new text and adds its digest
(``tools/library_code_read.py`` prints it, and fails the nightly run that
installs the newest releases). Off is the safe side: the statement that
needed the rule runs every time. ``%cash_status`` says which rules are on.

What a digest does not cover is the code those functions call. Each rule's
functions are the whole of the Python between pickle and the object; below
them is ``object``'s and ``ndarray``'s own pickling, which the rules that
use this one already rest on for every other object.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cash._user_attrs import attribute_or
from cash.loaded_modules import loaded_module

__all__ = ["RULES", "Rule", "functions_read", "is_read", "source_digest", "status"]


def _estimator_functions() -> list[Any] | None:
    base = attribute_or(loaded_module("sklearn.base"), "BaseEstimator", None)
    if base is None:
        return None
    return [vars(base).get(name) for name in ("__getstate__", "__setstate__")]


def _masked_array_functions() -> list[Any] | None:
    core = loaded_module("numpy.ma.core")
    masked = attribute_or(core, "MaskedArray", None)
    if masked is None:
        return None
    methods = [vars(masked).get(name) for name in ("__reduce__", "__getstate__", "__setstate__")]
    return [*methods, attribute_or(core, "_mareconstruct", None)]


@dataclass(frozen=True)
class Rule:
    """One rule that rests on library code: the *library* whose code it
    is (its distribution's module, for the version shown), the functions
    read (*functions* returns them, or None while the library is not
    loaded), the digests *accepted* with the releases each was read in,
    and what the statement does without the rule (*without*)."""

    library: str
    functions: Callable[[], list[Any] | None]
    accepted: Mapping[str, str]
    without: str


#: Add a digest only after reading the functions it stands for and finding
#: that they still do what the rule's own text (in `cash.notebook.anchors`)
#: says. ``python tools/library_code_read.py`` prints the installed digest.
RULES: dict[str, Rule] = {
    "estimator_pickling": Rule(
        library="sklearn",
        functions=_estimator_functions,
        accepted={"4ed9a186c438a4efa06144182ef7b57b4c387485437f34ba6e045ee508f14517": "scikit-learn 1.9.1"},
        without="a fitted estimator that keeps a slice of its training data runs every time",
    ),
    "masked_array_pickling": Rule(
        library="numpy",
        functions=_masked_array_functions,
        accepted={"9ff51462d3857607bdfdf7b037313685447855de725d216df97432eba5eefb41": "numpy 2.5.4 and 2.5.5"},
        without="an output holding a masked array beside a slice it was made from runs every time",
    ),
}

#: ``rule -> (the functions judged, their digest)``: the text of a function
#: is read from its file once, and again only when the function is another
#: object (a library reloaded, a method replaced).
_JUDGED: dict[str, tuple[tuple[Any, ...], str | None]] = {}


def source_digest(functions: list[Any]) -> str | None:
    """The SHA-256 of the functions' source text, in order, with the
    indentation they share and the spaces at line ends removed; None when
    one of them is missing or has no text to read (compiled, or its file
    is gone)."""
    digest = hashlib.sha256()
    for function in functions:
        try:
            text = inspect.getsource(function)
        except (OSError, TypeError):
            return None
        lines = inspect.cleandoc("\n" + text).splitlines() if text.strip() else []
        digest.update("\n".join(line.rstrip() for line in lines).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _digest_of(name: str) -> str | None:
    """The digest of the functions of rule *name* as they are now; None
    while its library is not loaded or their text cannot be read."""
    functions = RULES[name].functions()
    if functions is None:
        return None
    judged = _JUDGED.get(name)
    if judged is not None and len(judged[0]) == len(functions) and all(a is b for a, b in zip(judged[0], functions)):
        return judged[1]
    digest = source_digest(functions)
    _JUDGED[name] = (tuple(functions), digest)
    return digest


def is_read(name: str) -> bool:
    """Whether the library code rule *name* rests on is loaded and is a
    text that was read."""
    digest = _digest_of(name)
    return digest is not None and digest in RULES[name].accepted


def functions_read(name: str) -> list[Any]:
    """The functions of rule *name* when `is_read`, else none."""
    return list(RULES[name].functions() or ()) if is_read(name) else []


def status() -> dict[str, str]:
    """``rule -> line`` for ``%cash_status``: ``"on (...)"``, ``"off:
    ..."`` with what the statement does instead, or ``"not loaded"``."""
    lines = {}
    for name, rule in RULES.items():
        if rule.functions() is None:
            lines[name] = "not loaded"
            continue
        digest = _digest_of(name)
        version = attribute_or(loaded_module(rule.library), "__version__", "?")
        if digest in rule.accepted:
            lines[name] = f"on (as read in {rule.accepted[digest]}; installed: {version})"
        else:
            lines[name] = f"off: {rule.library} {version} has pickling code cash has not read; {rule.without}"
    return lines
