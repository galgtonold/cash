"""The user code a piece of code references: the functions and classes its
code objects load by name (or spell as a string), and its annotations,
followed transitively."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterator
from typing import Any

from .._annotation_refs import annotation_referents
from .._memo import CODE_OBJECTS, LruMemo
from ..install_paths import is_user_module
from .arg_hashing import is_opaque
from .call_state import KeyBuildFailed
from .user_code import cash_wrapped, is_user_code_object, user_layers

#: How many user-code objects one reference walk (`CodeRefs.reached`)
#: may reach. Not a depth or a count that real code meets: the walk follows
#: every reference, however deep, and its seen set ends cycles. What can pass
#: it is code that makes a NEW object on every read (a module ``__getattr__``
#: building a function per lookup), where the walk would never end. Past it
#: the key would leave code out, so the call runs uncached instead
#: (KEY-HELPERS-UNWALKABLE), as the helper walk does
#: (``PurityAnalyzer._WALK_LIMIT``).
MAX_CODE_REF_TARGETS = 5_000


def walk_nested_code(code: types.CodeType, glb: dict):
    """Yield *code* and the code objects nested in its constants, however deep.

    A comprehension, a lambda, or a nested ``def`` compiles to its own
    code object stored in ``co_consts``; the names IT references do not
    appear in the parent's ``co_names``. ``field(default_factory=lambda:
    B(0))`` is exactly that shape -- ``B`` is reachable only through the
    lambda -- so a walk that stopped at the top level would miss the case
    this exists for.
    """
    stack = [code]
    while stack:
        current = stack.pop()
        yield current, glb
        stack.extend(const for const in reversed(current.co_consts) if isinstance(const, types.CodeType))


class CodeRefs:
    """The user-code objects code references, resolved fresh on every call
    (`CodeRefs.targets`), and the walk over everything reachable that way
    (`CodeRefs.reached`)."""

    def __init__(self) -> None:
        # object -> tuple of (code object, globals dict) it carries. Static for
        # as long as that object exists (a redefinition makes a new one), so it
        # is safe to memo; the NAMES those code objects reference are resolved
        # fresh per call, because what a name is bound to can change.
        self._code_refs_cache: LruMemo[Any, tuple] = LruMemo(CODE_OBJECTS)

    def _code_and_globals(self, obj: Any):
        """Yield ``(code object, globals)`` for the code *obj* carries."""
        carriers: list[Any] = []
        if isinstance(obj, type):
            for base in obj.__mro__:
                if base is object or is_opaque(base):
                    continue
                if not is_user_code_object(base):
                    continue
                carriers.extend(vars(base).values())
                # Same blind spot as class_surface_parts: a field declaring
                # default_factory has no class attribute to find in vars().
                fields_map = getattr(base, "__dataclass_fields__", None)
                if isinstance(fields_map, dict):
                    for fld in fields_map.values():
                        factory = getattr(fld, "default_factory", None)
                        if factory is not None and factory is not dataclasses.MISSING:
                            carriers.append(factory)
        else:
            carriers.append(obj)

        for member in carriers:
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if isinstance(member, property):
                accessors = [a for a in (member.fget, member.fset, member.fdel) if a]
            else:
                accessors = [member]
            for accessor in accessors:
                # Every layer: a function under two decorators was read one
                # layer down, and what its body names was never reached.
                for layer in (accessor, *user_layers(accessor)):
                    code = getattr(layer, "__code__", None)
                    glb = getattr(layer, "__globals__", None)
                    if isinstance(code, types.CodeType) and isinstance(glb, dict):
                        yield from walk_nested_code(code, glb)

    def targets(self, obj: Any) -> list[Any]:
        """User-code objects that *obj*'s code references by global name.

        Resolution happens on every call, deliberately. Only the (code,
        globals) pairs are memoized -- those are fixed for as long as *obj*
        exists -- because what a NAME is bound to can change underneath us,
        and that change is precisely what must invalidate.

        Names come from ``co_names``, i.e. what the code actually LOADS, and
        from *obj*'s annotations. An annotation is not always a hint that never
        runs: pydantic runs ``B``'s validators for a field ``b: B``, and a
        ``typing.get_type_hints`` builder constructs ``B`` from ``A``'s hints
        (see ``cash._annotation_refs``). A hint that really is inert costs a
        recompute when its class is edited, never a stale value.
        """
        pairs = None
        try:
            pairs = self._code_refs_cache.get(obj)
        except TypeError:
            pass  # unhashable - recompute each time rather than fail
        if pairs is None:
            pairs = tuple(self._code_and_globals(obj))
            try:
                self._code_refs_cache[obj] = pairs
            except TypeError:
                pass

        targets: list[Any] = []
        seen_names: set[str] = set()

        def consider(value: Any) -> None:
            value = cash_wrapped(value)
            if value is None or value is obj:
                return
            if not (isinstance(value, type) or callable(value)):
                return
            try:
                if is_opaque(value) or not is_user_code_object(value):
                    return
            except Exception:  # noqa: BLE001 - never break a call
                return
            targets.append(value)

        for code, glb in pairs:
            for name in code.co_names:
                if name in seen_names:
                    continue
                seen_names.add(name)
                consider(glb.get(name))
            # A name spelled as a string: `getattr(MOD, "fun1")()`,
            # `globals()["fun1"]`. The string is a constant, not a loaded name,
            # so `co_names` does not have it. Resolved in the code's
            # module and in the user modules it loads; a string that only
            # happens to match a function costs a needless recompute, never a
            # stale value.
            names = [c for c in code.co_consts if isinstance(c, str) and c.isidentifier() and c not in seen_names]
            if not names:
                continue
            modules = [glb.get(n) for n in code.co_names]
            modules = [m for m in modules if isinstance(m, types.ModuleType) and is_user_module(m)]
            for name in names:
                seen_names.add(name)
                consider(glb.get(name))
                for module in modules:
                    consider(getattr(module, name, None))
        if isinstance(obj, type) or callable(obj):
            seen_ids = {id(t) for t in targets}
            for value in annotation_referents(obj, is_user_code_object):
                if id(value) not in seen_ids:
                    seen_ids.add(id(value))
                    consider(value)
        return targets

    def reached(self, obj: Any) -> Iterator[Any]:
        """Every user-code object reachable from *obj*'s code, each once.

        Breadth-first with an identity ``seen`` set, so a mutually-referential
        pair (``A.make`` returns ``B``, ``B.make`` returns ``A``) terminates
        instead of recursing forever. Reached objects are held in *keep* for
        the duration: ``id()`` is only unique among LIVE objects, and a
        collected one could otherwise let a later object reuse its id and be
        skipped as already-seen.

        Lazy: each object is yielded as the walk reaches it. Past
        `MAX_CODE_REF_TARGETS` it raises `KeyBuildFailed`.
        """
        seen: set[int] = {id(obj)}
        keep: list[Any] = [obj]
        frontier: list[Any] = [obj]
        while frontier:
            following: list[Any] = []
            for source in frontier:
                for target in self.targets(source):
                    if id(target) in seen:
                        continue
                    seen.add(id(target))
                    keep.append(target)
                    if len(keep) > MAX_CODE_REF_TARGETS:
                        name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
                        raise KeyBuildFailed(
                            "KEY-HELPERS-UNWALKABLE",
                            f"cash cannot key the code {name} reaches: it does not end (over "
                            f"{MAX_CODE_REF_TARGETS} functions and classes; code that makes a new "
                            f"function on every read can cause this), so the call ran uncached.",
                            "Name what the result depends on with depends_on=[...] instead of "
                            "creating it on every read.",
                        )
                    yield target
                    following.append(target)
            frontier = following
