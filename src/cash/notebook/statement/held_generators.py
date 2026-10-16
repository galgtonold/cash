"""Generators a statement draws from through an attribute (``m.rng.normal()``).

cash follows a generator held in a variable, and one held in an object the
statement changes, whose rebuild makes a new generator. One reached through
an attribute of something cash does not rebuild (a class attribute read
through an instance, a module, a function) is not followed: a cell re-run on
its own draws from wherever such a generator stands now.
:class:`HeldGenerators` notices that it moved since the statement last
started and warns RANDOM-HELD-GENERATOR.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
import types
from collections.abc import Mapping
from typing import Any

from cash.control_markers import strip_markers
from cash.diagnostics import warn_diagnostic
from cash.tracking.randomness import CashRandomnessWarning
from cash.tracking.randomness.detect import attribute_draws
from cash.tracking.randomness.state import object_carrier_state, same_carrier_state

from ...exceptions import SOURCE_RETRIEVAL_ERRORS
from ...install_paths import is_user_code_file
from ...source_norm import exact_source_digest
from ...source_reading import own_source

__all__ = ["HeldGenerators"]

logger = logging.getLogger(__name__)

#: How many statement starts are kept; past it, the oldest go.
_KEPT = 512
_MISSING = object()


def _attribute(obj: Any, name: str) -> Any:
    """``obj.name`` as stored, without running a property or ``__getattr__``."""
    try:
        return inspect.getattr_static(obj, name)
    except AttributeError:
        return _MISSING


def _follow(obj: Any, attrs: tuple[str, ...]) -> Any:
    for name in attrs:
        obj = _attribute(obj, name)
        if obj is _MISSING:
            return _MISSING
    return obj


def _method_draws(obj: Any, method: str) -> tuple[tuple[str, ...], ...]:
    """The attribute chains from ``self`` that the method *method* of *obj*'s
    own class (code of the user's) draws from: ``("rng",)`` for a body that
    calls ``self.rng.normal()``."""
    if isinstance(obj, (type, types.ModuleType)):
        return ()
    fn = _attribute(type(obj), method)
    if not isinstance(fn, types.FunctionType) or not is_user_code_file(fn.__code__.co_filename):
        return ()
    known = _METHOD_CHAINS.get(fn.__code__)
    if known is None:
        known = _self_chains(fn)
        if len(_METHOD_CHAINS) < 4096:
            _METHOD_CHAINS[fn.__code__] = known
    return known


#: A method's code -> what `_self_chains` found in it.
_METHOD_CHAINS: dict[types.CodeType, tuple[tuple[str, ...], ...]] = {}


def _self_chains(fn: types.FunctionType) -> tuple[tuple[str, ...], ...]:
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if not params:
        return ()
    try:
        source = textwrap.dedent(own_source(fn))
    except SOURCE_RETRIEVAL_ERRORS:
        return ()
    return tuple(draw.attrs for draw in attribute_draws(source, params[0]))


def held_generators(code: str, namespace: Mapping[str, Any]) -> dict[str, tuple[Any, int]]:
    """``{"m.rng": (generator, line)}`` for each generator *code* draws from
    through an attribute, directly or in a method of the user's it calls on
    an object."""
    found: dict[str, tuple[Any, int]] = {}
    for draw in attribute_draws(strip_markers(code)):
        root = namespace.get(draw.root, _MISSING)
        if root is _MISSING:
            continue
        chains = [draw.attrs] if draw.method is None else _method_draws(root, draw.method)
        for attrs in chains:
            obj = _follow(root, attrs)
            if obj is not _MISSING and object_carrier_state(obj) is not None:
                found.setdefault(".".join((draw.root, *attrs)), (obj, draw.line))
    return found


class HeldGenerators:
    """Where each attribute-held generator stood when a statement last
    started, to see one a re-run finds moved."""

    def __init__(self) -> None:
        #: ``(statement digest, label) -> (generator, its state)``. The
        #: generator is held so that ``is`` answers: a numpy generator takes
        #: no weak reference.
        self._starts: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._warned: set[tuple[str, str]] = set()

    def before_run(self, code: str, namespace: Mapping[str, Any], cell: int | None, rerun: bool) -> None:
        """Record where the generators *code* draws from through an attribute
        stand, about to be computed; warn when cell *cell* runs on its own
        (*rerun*: not right after the cell above it) and one is the generator
        the statement started from last time, moved on since."""
        try:
            generators = held_generators(code, namespace)
        except Exception:  # advisory: a probe of arbitrary objects must not break the statement
            logger.debug("[RANDOMNESS] held-generator probe failed", exc_info=True)
            return
        if not generators:
            return
        digest = exact_source_digest(code)
        for label, (generator, line) in generators.items():
            state = object_carrier_state(generator)
            key = (digest, label)
            last = self._starts.pop(key, None)
            self._starts[key] = (generator, state)
            if len(self._starts) > _KEPT:
                self._starts.pop(next(iter(self._starts)))
            if not rerun or cell is None or last is None or last[0] is not generator:
                continue
            if same_carrier_state(last[1], state) or key in self._warned:
                continue
            self._warned.add(key)
            self._warn(code, label, line, cell)

    @staticmethod
    def _warn(code: str, label: str, line: int, cell: int) -> None:
        first = strip_markers(code).strip().splitlines()[0] if strip_markers(code).strip() else code
        if len(first) > 60:
            first = first[:57] + "..."
        warn_diagnostic(
            CashRandomnessWarning,
            "RANDOM-HELD-GENERATOR",
            f"cell {cell + 1} runs on its own, and `{first}` draws from "
            f"`{label}`, a generator held by an object. cash does not put such a generator back where "
            "a run from the top has it, so this draw continues from where the last run left it.",
            "draw from the generator in the cell that creates it, or hold it in a variable and draw "
            "from that; for the numbers a run from the top gives, run the notebook from the top.",
            location=("<cash>", line),
        )
