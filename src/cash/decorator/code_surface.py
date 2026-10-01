"""The code surface of what a call reaches: the bytecode-level identity
of a class's members, a callable's code and the user code it runs, and
the user classes behind an instance."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pickle
import re
import types
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, LruMemo
from ..code_digest import source_digest, unwrap_partials
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_norm import code_consts_without_docstring
from .arg_hashing import is_opaque
from .call_state import KeyBuildFailed
from .code_refs import CodeRefs
from .function_identity import hash_callable_source
from .key_values import SYNC_TYPES, is_immutable_capture, iter_contained
from .user_code import cash_wrapped, is_user_class, is_user_code_object, user_layers

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher

logger = logging.getLogger(__name__)


#: Pydantic v2 compiles these onto every model class. They are derived from the
#: field declarations and their digest differs in every process, so folding them
#: made a pydantic spec un-cacheable across runs. `CodeSurface._pydantic_field_parts`
#: folds the declarations they were standing in for.
PYDANTIC_COMPILED = frozenset(
    {
        "__pydantic_core_schema__",
        "__pydantic_serializer__",
        "__pydantic_validator__",
    }
)


def _defaults_pin(functions: list[Any]) -> tuple | None:
    """The defaults of *functions*, when every one is immutable; else None."""
    pin = []
    for fn in functions:
        pos = getattr(fn, "__defaults__", None) or ()
        kwd = getattr(fn, "__kwdefaults__", None) or {}
        if not (all(is_immutable_capture(v) for v in pos) and all(is_immutable_capture(v) for v in kwd.values())):
            return None
        pin.append((pos, dict(kwd)))
    return tuple(pin)


#: A memory address in a repr (``<object object at 0x7f...>``).
_ADDRESS_REPR = re.compile(r"\bat 0x[0-9a-fA-F]+")


class CodeSurface:
    """What identifies code reached through arguments and globals: the code
    surface of classes, instances and functions, and of the user code that
    code references (`CodeRefs`)."""

    def __init__(self, args: ArgHasher) -> None:
        self._args = args
        # user class -> source hash. A class's source cannot change within a
        # running interpreter, so it is hashed once and reused; see
        # `user_class_source_hash` / `instance_class_source_parts`.
        self._user_class_src_cache: LruMemo[type, str] = LruMemo(CODE_OBJECTS)
        # user class or function -> code-surface digest (bytecode-based, class-
        # aware); see `code_surface_hash`. Keyed on the object itself, not
        # id(), so a redefinition (a new object) is a distinct memo entry.
        self._code_surface_cache: LruMemo[Any, tuple[tuple | None, list, str]] = LruMemo(CODE_OBJECTS)
        self._refs = CodeRefs()

    def _code_identity(self, fn: Any) -> tuple:
        """The bytecode-level identity of a callable, or ``()`` if it has none.

        Bytecode rather than source because a class defined in a notebook cell
        has no retrievable source at all: ``inspect.getsource`` resolves a class
        through ``sys.modules[cls.__module__].__file__``, and a notebook
        ``__main__`` has none. A function escapes this via ``co_filename``,
        which is why ``hash_callable_source`` works for helpers and not here.

        Comments and formatting are absent from bytecode, so they do not
        invalidate -- strictly better than source hashing. Docstrings live in
        ``co_consts`` and are masked out, so they do not either.

        Instance method (not static) because defaults/kwdefaults go through
        ``_value_identity`` -> ``self._args.hash_payload``: a default like
        ``def m(self, x=_MISSING)`` reprs as ``<object object at 0x...>``,
        the same address leak as a nested code object, just one layer up.
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return ()
        return (
            self._code_object_identity(code),
            self._value_identity(getattr(fn, "__defaults__", None)),
            self._value_identity(getattr(fn, "__kwdefaults__", None)),
        )

    def _code_object_identity(self, code: types.CodeType) -> tuple:
        """Structural identity of one ``types.CodeType``, recursing into any
        nested code object in ``co_consts`` instead of ``repr()``-ing it, and
        folding every OTHER const through ``_value_identity`` instead of
        ``repr()`` too.

        A lambda, a generator expression, or -- pre-3.12, before PEP 709
        inlined them -- a plain comprehension compiles to a NESTED code
        object stored in the enclosing method's ``co_consts``. Its ``repr()``
        is ``<code object <genexpr> at 0x...>``: a live memory address, fresh
        every process, which would give a class a different digest in every
        process. A nested code object carries no defaults of
        its own (those belong to the FUNCTION eventually built from it, not to
        the raw code object), so only co_code/co_consts/co_names apply here.

        The same holds for a plain (non-code) const: ``x in {'alpha',
        'beta'}`` compiles a ``frozenset`` straight into ``co_consts``, and
        ``repr()`` of a set/frozenset follows the table's internal
        (hash-order-dependent) iteration, which Python's per-process
        string-hash randomization changes. ``_value_identity``
        folds CONTENT instead, which is order-independent for a set/frozenset.
        """
        return (
            code.co_code,
            tuple(
                self._code_object_identity(k) if isinstance(k, types.CodeType) else self._value_identity(k)
                for k in code_consts_without_docstring(code)
            ),
            tuple(code.co_names),
        )

    def _value_identity(self, v: Any) -> str:
        """Address-free identity for a value that is not itself a code object.

        ``ArgHasher.hash_payload`` folds CONTENT and is the established,
        address-free tool used throughout this file for exactly this. What it
        cannot pickle goes to `_unpicklable_identity`, not to ``repr()``:
        ``repr()`` is a memory ADDRESS for the most ordinary unpicklable
        defaults (``key=lambda r: r``, ``lock=threading.Lock()``), different
        in every process, so the entry would never hit after a restart.

        ``CodeSurface.class_surface_parts`` already refuses a ``repr()`` fallback, on the
        grounds that it "would reintroduce the address leak this member-content
        fold exists to avoid". This makes the two agree.
        """
        try:
            return self._args.hash_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            return self._unpicklable_identity(v)

    def _unpicklable_identity(self, v: Any, _path: frozenset[int] = frozenset()) -> str:
        """Process-stable stand-in for a value ``ArgHasher.hash_payload`` refused.

        Recursive because ``__defaults__`` is hashed as a WHOLE TUPLE: one
        unpicklable element poisons the entire tuple, so every element that
        CAN be folded still is, and only the residue is approximated.

        A lambda, function or class is approximated by its CODE SURFACE, which
        is stable across processes and moves with an edit of the lambda's
        body; an address does neither.

        Everything else keeps its ``repr()`` when that repr is value-based
        (``Config(n=1)``): deterministic across processes, and it moves with
        the value. A repr that is only an address carries neither, so the
        value cannot be keyed: `KeyBuildFailed`, and the call runs uncached,
        as for an unhashable default of the cached function itself.
        """
        # However deep the containers nest: a lambda five lists down is code
        # like any other, and editing it must move the key. *_path* (the
        # containers being walked) ends a container that holds itself.
        if isinstance(v, (list, tuple, set, frozenset, dict)):
            if id(v) in _path:
                return f"<cycle:{type(v).__qualname__}>"
            _path = _path | {id(v)}
        if isinstance(v, (list, tuple, set, frozenset)):
            inner = [self._unpicklable_identity(x, _path) for x in v]
            if isinstance(v, (set, frozenset)):
                # Set iteration order follows the hash table, and string
                # hashing is randomized per process -- sort or reintroduce the
                # very instability this method exists to remove.
                inner.sort()
            return f"{type(v).__qualname__}[{'|'.join(inner)}]"
        if isinstance(v, dict):
            return (
                "dict["
                + "|".join(
                    sorted(
                        f"{self._unpicklable_identity(k, _path)}={self._unpicklable_identity(val, _path)}"
                        for k, val in v.items()
                    )
                )
                + "]"
            )
        try:
            return self._args.hash_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            pass
        surface = self.code_surface_hash(v)
        if surface is not None:
            return f"code:{surface}"
        try:
            text = repr(v)
        except Exception as e:  # noqa: BLE001 - a __repr__ may raise
            logger.debug("[CORE] repr() failed while identifying %s: %s", type(v), e)
            text = ""
        # A repr with an address (``<object object at 0x...>``) is different
        # in every process and does not move when the object's content
        # does: nothing about the value can be keyed. A value-based repr
        # stands for the value, a hex literal in it included.
        if text and not _ADDRESS_REPR.search(text):
            return text
        cls = type(v)
        if isinstance(v, SYNC_TYPES):
            return f"<{cls.__module__}.{cls.__qualname__}>"
        name = f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', '?')}"
        raise KeyBuildFailed(
            "KEY-UNHASHABLE-DEFAULT",
            f"cash cannot key a {name} that code reaching the call holds as a default or constant: "
            f"it cannot be hashed and has no value-based repr, so a change to it could not be "
            f"seen and the call ran uncached.",
            f"get the value out of the signature -- build it in the body or pass it as an argument -- "
            f"or register a hasher with cash.register_hasher({cls.__qualname__}, ...).",
        )

    def _code_surface_own(self, obj: Any) -> str | None:
        """A digest of the user code *obj* itself carries, or ``None``.

        Its OWN surface only -- code merely referenced by that code is folded
        by :meth:`CodeSurface.code_surface_hash`, which combines these per-object digests.
        The split is what keeps the memo below honest: memoizing a digest that
        included a referenced class would serve a stale answer when only that
        OTHER class is redefined (a notebook cell re-run), because *obj* is
        still the same object and still hits its memo entry.

        ``None`` means "cannot determine" and the caller must fall back to
        today's by-reference key. A hashing failure is not "cannot determine":
        it propagates, and the key build that asked runs the call uncached.

        Memoized on the object itself, following ``_user_class_src_cache``:
        redefining a class produces a NEW object and therefore a distinct dict
        key, so the memo cannot serve a stale entry. Keying on ``id()`` would
        be a correctness bug, since CPython recycles addresses.
        """
        # A `functools.partial` is its function plus arguments. The arguments
        # already reach the key (a partial pickles them, by value); its code is
        # the wrapped function's, which pickle names only by reference -- so
        # the wrapped function's code is keyed here, or an edit to its body
        # would keep the key.
        obj = cash_wrapped(unwrap_partials(obj))
        # Dispatch FIRST, memo read second. Every argument to a cached
        # function passes through here, and most are not a
        # class or callable at all -- a list, dict, set, numpy array,
        # DataFrame. Checking the dispatch before touching the memo means
        # those return None from a plain isinstance()/callable() check
        # instead of reaching the memo at all.
        is_type = isinstance(obj, type)
        if not (is_type or callable(obj)):
            return None
        if not is_user_code_object(obj):
            return None
        # By this point *obj* is a class or a callable, and while both are
        # hashable in the overwhelming common case, neither is guaranteed to
        # be (a __call__-implementing instance can set __hash__ = None).
        try:
            cached = self._code_surface_cache.get(obj)
        except TypeError:
            cached = None
        if cached is not None and cached[0] == _defaults_pin(cached[1]):
            return cached[2]
        layers: list[Any] = []
        if is_type:
            parts = self.class_surface_parts(obj)
        else:
            ident = self._code_identity(obj)
            if not ident:
                return None
            parts = [(getattr(obj, "__qualname__", "?"), "", ident)]
            # And the user functions it runs besides its own code: what a
            # decorator wraps, a function a closure holds. A decorator's
            # wrapper code is shared by every function it wraps; the
            # function itself is what tells them apart.
            layers = [obj, *user_layers(obj)]
            for layer in layers[1:]:
                parts.append((getattr(layer, "__qualname__", "?"), "runs", self._code_identity(layer)))
        if not parts:
            return None
        digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
        # A function's defaults are rebound (`f.__defaults__ = ...`) or
        # mutated in place on the same object: the memo holds only while
        # every default is immutable and the containers are unchanged.
        pin = _defaults_pin(layers)
        if pin is not None:
            try:
                self._code_surface_cache[obj] = (pin, layers, digest)
            except TypeError:
                pass  # unhashable object - skip the memo, keep the answer
        return digest

    def code_surface_hash(self, obj: Any) -> str | None:
        """A digest of *obj*'s code AND the user code that code reaches.

        What an argument or global directly carries is not enough: a class
        whose field factory constructs another class changes behaviour when
        THAT class is edited (``A(value=B(value=10))`` becomes
        ``A(value=B(value=1000))``), and nothing in the first class's own
        surface moves.

        Reachability is STATIC: names the code loads from its globals,
        transitively, however deep. Code selected at runtime (a class picked out of
        a dict) still cannot be followed, so this narrows the gap rather than
        closing it.
        """
        own = self._code_surface_own(obj)
        if own is None:
            return None
        reached = self._code_ref_closure(obj)
        if not reached:
            return own
        return hashlib.sha256(":".join([own, *sorted(reached)]).encode("utf-8")).hexdigest()

    def _code_ref_closure(self, obj: Any) -> list[str]:
        """Own-digests of every user-code object reachable from *obj*'s code
        (`CodeRefs.reached`), in the order the walk reaches them."""
        digests: list[str] = []
        for target in self._refs.reached(obj):
            digest = self._code_surface_own(target)
            if digest is not None:
                digests.append(f"{getattr(target, '__qualname__', '?')}:{digest}")
        return digests

    def _dataclass_field_parts(self, base: type, field_map: dict) -> list[tuple]:
        """Fold a dataclass's field metadata, which nothing else reaches.

        ``@dataclass`` moves the per-field declaration off the class attribute
        and into ``__dataclass_fields__``. The attribute that remains is just
        the default value, so a field's TYPE and its ``metadata=`` never
        reached the digest -- and ``__dataclass_fields__`` itself cannot be
        pickled, because ``Field.metadata`` is a ``mappingproxy``, so the
        generic fold cannot take it.

        A field description (``field(metadata={"desc": ...})``) is prompt
        text in every structured-output library there is -- it is not
        decoration, it is the instruction.

        Only the parts nothing else covers are folded. The default value is
        already the class attribute and is hashed there; folding it again would
        change nothing and cost a hash.
        """
        parts: list[tuple] = []
        for fname, f in sorted(field_map.items()):
            try:
                meta = self._args.hash_payload((dict(getattr(f, "metadata", {}) or {}),), {})
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                # An unpicklable metadata VALUE. Skip the metadata rather than
                # repr() it: a repr here would leak an object address and make
                # the digest differ between processes, which is worse than not
                # tracking it.
                meta = None
            # `str(f.type)` and not the object: an annotation may be a string
            # (`from __future__ import annotations`) or a class, and both spell
            # the same thing deterministically this way.
            parts.append((base.__qualname__, f"__dataclass_field__:{fname}", (str(getattr(f, "type", "")), meta)))
        return parts

    def _pydantic_field_parts(self, cls: type) -> list[tuple]:
        """Fold a pydantic model's field declarations.

        The counterpart to :meth:`_dataclass_field_parts`. Pydantic keeps them
        in ``model_fields`` -- a property on the class, so ``vars(cls)`` never
        sees it -- and the only other place they appear is the compiled trio
        skipped above, whose digest is different in every process.

        Without these parts a pydantic spec passed as an argument would have
        no stable digest, and would never be cached across processes.

        `description` is the load-bearing one -- it is the instruction sent to
        the model in every structured-output library there is.
        """
        # Guarded: this runs for EVERY class the surface walk sees, and
        # `model_fields` on a non-pydantic class could be a property that
        # computes something, or raises. Hashing must never be the thing that
        # breaks a call.
        try:
            fields = getattr(cls, "model_fields", None)
        except Exception:  # noqa: BLE001 - a descriptor of someone else's
            return []
        if not isinstance(fields, dict):
            return []
        parts: list[tuple] = []
        for fname, info in sorted(fields.items()):
            try:
                default = self._args.hash_payload((getattr(info, "default", None),), {})
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                default = None
            parts.append(
                (
                    cls.__qualname__,
                    f"__pydantic_field__:{fname}",
                    (
                        str(getattr(info, "annotation", "")),
                        getattr(info, "description", None),
                        getattr(info, "alias", None),
                        default,
                    ),
                )
            )
        return parts

    def class_surface_parts(self, cls: type, _path: tuple[type, ...] = ()) -> list[tuple]:
        """Every user-code member of *cls* and its user base classes.

        Walked in reverse MRO so a subclass override lands after the base it
        replaces, and sorted within each class so dict ordering cannot change
        the digest. *_path* is the chain of classes whose nested members led
        here, which ends a cycle.
        """
        parts: list[tuple] = self._pydantic_field_parts(cls)
        for base in reversed(cls.__mro__):
            # An OPAQUE base contributes nothing, so `cash.opaque(VendorBase)`
            # also stops a `Derived(VendorBase)` digest moving when the vendor
            # edits its own base. Without this the escape hatch worked only
            # when the opaque class was the one passed, which is not how
            # `docs/decorator.md` advertises it. Per-base and exact-match, so
            # opacity still does not inherit: marking a base does not make
            # `Derived` opaque, it only drops that base's own members.
            if base is object or is_opaque(base):
                continue
            if not is_user_code_object(base):
                continue
            for name, member in sorted(vars(base).items(), key=lambda kv: kv[0]):
                parts.extend(self._member_parts(cls, base, name, member, _path))
        parts.extend(self._field_factory_parts(cls))
        return parts

    def _member_parts(self, cls: type, base: type, name: str, member: Any, path: tuple[type, ...]) -> list[tuple]:
        """The parts one class attribute *name* of *base* adds to *cls*'s surface."""
        # __firstlineno__ (class attribute since Python 3.13, absent on
        # 3.10/3.11) records the class's first source line, which shifts
        # when a comment or blank line is added above it -- with no
        # code change at all. Skipping it is what keeps "comments do
        # not invalidate" (see _code_identity) true on 3.13+ too.
        if name in ("__dict__", "__weakref__", "__module__", "__firstlineno__"):
            return []
        # The class docstring is documentation, the same as a method's
        # (masked in `_code_object_identity`), so it is folded as if
        # there were none -- the member itself stays, because for a
        # type whose surface is nothing else (a C type like
        # `_thread.lock`) dropping it left no surface at all and the
        # type was reported as unhashable code. Except on a pydantic
        # model: its docstring is the schema's `description`, which
        # structured-output libraries send to the model as the prompt.
        if name == "__doc__" and not self._pydantic_field_parts(base):
            member = None
        # Pydantic v2 compiles three Rust objects onto every model.
        # They are DERIVED from the field declarations, and their
        # digest differs in every process. `model_fields` below carries the same
        # declarations and is stable, so this loses nothing.
        if name in PYDANTIC_COMPILED:
            return []
        if name == "__dataclass_fields__" and isinstance(member, dict):
            return self._dataclass_field_parts(base, member)
        if isinstance(member, property):
            return [
                (base.__qualname__, f"{name}.{tag}", ident)
                for tag, accessor in (("get", member.fget), ("set", member.fset))
                if (ident := self._code_identity(accessor))
            ]
        target, ident = self._member_code(member)
        if not callable(member):
            return self._data_member_parts(base, name, member, ident)
        if ident:
            return [self._code_member_part(base, name, member, target, ident)]
        return self._codeless_callable_parts(cls, base, name, member, path)

    def _member_code(self, member: Any) -> tuple[Any, tuple]:
        """``(target, ident)``: the function whose code a class attribute
        runs, unwrapped, and its `_code_identity` (``()`` when none).

        Unwraps decoration to reach the function whose __code__
        actually reflects a body edit (mirrors the single-level
        __wrapped__ unwrap in `MethodClassDeps._analyze_method_self_deps`). Without
        this, @functools.wraps and @functools.lru_cache both hash
        the WRAPPER's own generic dispatch code -- fixed regardless
        of what the wrapped body says -- and @functools.
        singledispatchmethod has no __wrapped__ or __code__ at all
        (it exposes the underlying function as .func instead), and
        would reach the data-attribute branch as an
        unchanging descriptor repr.
        """
        outer = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        target = getattr(outer, "__wrapped__", outer)
        if not hasattr(target, "__code__"):
            func_attr = getattr(target, "func", None)
            if func_attr is not None and hasattr(func_attr, "__code__"):
                target = func_attr
        ident = self._code_identity(target)
        if ident and callable(outer):
            # Every other user function it runs: the decorator's own
            # wrapper, and the layers below the first ``__wrapped__``.
            # Under two decorators the method body itself was never
            # reached.
            extra = [
                layer
                for layer in (outer, *user_layers(outer))
                if layer is not target and isinstance(layer, types.FunctionType) and is_user_code_object(layer)
            ]
            if extra:
                ident = (ident, tuple(self._code_identity(layer) for layer in extra))
        return target, ident

    def _member_content(self, member: Any) -> str | None:
        """The digest of a class attribute's own pickled content, or None
        when it cannot be pickled."""
        try:
            return self._args.hash_payload((member,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            return None

    def _code_member_part(self, base: type, name: str, member: Any, target: Any, ident: tuple) -> tuple:
        """The part for a callable class attribute whose code was found.

        When `ident` was reached by UNWRAPPING (``.func`` /
        ``__wrapped__``), it describes the inner function and
        says nothing about the state the wrapper itself
        carries: ``functools.partial(scale, 3)`` and
        ``partial(scale, 4)`` unwrap to the same ``scale``
        and collided. Fold the wrapper's own content too,
        exactly as the non-callable branch does for a
        ``partialmethod``. Skipped when nothing was
        unwrapped, because a plain method's own pickle is
        its module path -- which would make every class's
        digest depend on the module it lives in.
        """
        if target is not member:
            own = self._member_content(member)
            if own is not None:
                return (base.__qualname__, name, (ident, own))
        return (base.__qualname__, name, ident)

    def _codeless_callable_parts(
        self, cls: type, base: type, name: str, member: Any, path: tuple[type, ...]
    ) -> list[tuple]:
        """The parts for a callable class attribute with NO reachable ``__code__``.

        A nested class (``class Outer: inner = Inner``), a
        ``functools.partial``, a callable instance. Its code
        (``Inner.f``) and its content (``partial(scale, 3)``
        against ``partial(scale, 4)``) are both folded.

        The nested walk recurses into ``CodeSurface.class_surface_parts``
        DIRECTLY, not through the memoized ``CodeSurface.code_surface_hash``,
        and a cycle ends on the PATH that led to it, never on a
        set shared across the walk. A shared set would make the
        digest depend on which class happened to be hashed first
        (the memo would hold a cut result for one order and a
        full one for the other) -- reintroducing exactly the
        cross-process instability ``_value_identity`` avoids.
        The path is fixed by *cls* alone, so every
        process gets the same answer regardless of order. No
        depth bound: a nested class is followed however deep.
        """
        nested = None
        inner_cls = member if isinstance(member, type) else type(member)
        if is_user_code_object(inner_cls):
            if inner_cls is cls or inner_cls in path:
                nested = f"cycle:{inner_cls.__qualname__}"
            else:
                sub_parts = self.class_surface_parts(inner_cls, (*path, cls))
                if sub_parts:
                    nested = hashlib.sha256(
                        repr(sub_parts).encode("utf-8"),
                    ).hexdigest()
        content = self._member_content(member)
        if nested is None and content is None:
            return []
        return [(base.__qualname__, name, (nested, content))]

    def _data_member_parts(self, base: type, name: str, member: Any, ident: tuple) -> list[tuple]:
        """The parts for a non-callable class attribute.

        Its OWN content is folded unconditionally -- a descriptor like
        functools.partialmethod carries bound state (.args)
        that lives on the descriptor ITSELF, not on the inner
        function `ident` resolved through .func, and a
        plain object that happens to expose an unrelated `.func`
        attribute must not have its OTHER state go invisible
        just because that lookup succeeded (a partialmethod's
        bound arguments, an unrelated object's own attributes).
        `ident` is folded TOO
        when reachable, so a non-callable descriptor that ALSO
        wraps a real function body -- functools.
        singledispatchmethod, functools.cached_property, both
        confirmed to expose .func without __wrapped__ or
        __code__ of their own -- has that body participate as
        well. Folding only one half silently drops whichever
        state that particular member happens to carry.

        No repr() fallback here (unlike _value_identity):
        falling back to repr() on this specific path would
        reintroduce the address leak this member-content fold
        exists to avoid (a class attribute's repr is often an
        address). If content can't
        be folded and no `ident` was found either, dropping the
        member is strictly safer than a non-deterministic repr.
        """
        content = self._member_content(member)
        if not ident and content is None:
            return []
        return [(base.__qualname__, name, (ident, content))]

    def _field_factory_parts(self, cls: type) -> list[tuple]:
        """The parts for *cls*'s dataclass field factories.

        ``dataclasses`` DELETES the class
        attribute when a field declares ``default_factory``, so the
        ``vars()`` walk cannot see it -- ``getattr_static`` raises
        AttributeError for that name. The factory is nonetheless code that
        decides what every instance holds: editing
        ``field(default_factory=lambda: B(0))`` to ``B(999)`` changes what
        ``A()`` produces.

        Read from ``__dataclass_fields__`` rather than calling
        ``dataclasses.fields()``: the latter raises on a non-dataclass and
        skips pseudo-fields, and this must never raise.
        """
        parts: list[tuple] = []
        fields_map = getattr(cls, "__dataclass_fields__", None)
        if isinstance(fields_map, dict):
            for fname, fld in sorted(fields_map.items()):
                factory = getattr(fld, "default_factory", None)
                # ``MISSING`` is a sentinel INSTANCE, not None; compare by
                # identity against the one dataclasses hands out.
                if factory is None or factory is dataclasses.MISSING:
                    continue
                ident = self._code_identity(factory)
                if ident:
                    parts.append((cls.__qualname__, f"field:{fname}:factory", ident))
        return parts

    def user_class_source_hash(self, cls: type) -> str:
        """Memoized source hash of a USER class.

        A class's source cannot change within a running interpreter: editing the
        file and re-importing produces a NEW class object (a distinct dict key),
        so the hash is computed once per class object and reused on every
        subsequent call. The per-call cost of the instance channel below is then
        a cheap object-graph walk plus dict lookups -- never source I/O.

        Source-first, surface-as-fallback. Both of this method's callers
        (``CodeSurface.instance_class_source_parts``, directly and via
        ``GlobalsFold.fold_read_globals``) gate on ``is_user_class`` -> ``is_user_module``,
        which requires ``__file__`` -- so every class actually reachable here
        already has retrievable source, and ``inspect.getsource`` succeeds. The
        class-aware surface (``CodeSurface.code_surface_hash``) only engages on
        ``SOURCE_RETRIEVAL_ERRORS`` -- a class truly without source, e.g. a
        notebook cell's ``__main__`` has no ``__file__`` -- or when this method
        is reached some other way. Whole-class source is preferred because it
        sees a body edit under ``@functools.wraps``, ``@lru_cache`` or
        ``@singledispatchmethod`` (source is just text), where
        ``CodeSurface.class_surface_parts`` walks the WRAPPER.
        """
        cached = self._user_class_src_cache.get(cls)
        if cached is not None:
            return cached
        h = source_digest(cls)
        if h is None:
            # No source to hash (or it doesn't parse). A class has no
            # __code__, so the callable fallback would key it on its name
            # alone; the class-aware surface sees its members.
            h = self.code_surface_hash(cls) or hash_callable_source(cls)
        self._user_class_src_cache[cls] = h
        return h

    def instance_class_source_parts(
        self,
        value: Any,
        _seen: set | None = None,
        own_pkg: str | None = None,
    ) -> list[tuple[str, str]]:
        """``(qualname, source-hash)`` for the user classes behind an INSTANCE.

        A cached function that reads a pre-built module-level object -- ``pre =
        MyTransformer()`` imported and dropped into a pipeline -- cannot be only
        VALUE-hashed: its ``__dict__`` pickle carries no method source, so an
        edit to ``MyTransformer.transform`` would leave the key unchanged and
        serve a stale result. Fold the source
        of the instance's class -- and of the user-class instances it holds,
        however deep -- so a method-body edit invalidates.

        The walk recurses only into user-class instances: a third-party object
        (a fitted sklearn estimator, a numpy array) is not user-editable and its
        internals must not churn the key, and stopping there also bounds the cost
        on real pipelines. A user class reachable only through a third-party
        container is not folded here; `CodeArgs.carrier_parts`, which a data
        global also goes through, searches library objects for it.
        """
        if _seen is None:
            _seen = set()
        parts: list[tuple[str, str]] = []
        # Depth-first with an explicit stack, however deep the instances nest
        # (a linked list of user objects is as deep as it is long).
        # ``_seen`` ends cycles.
        stack = [value]
        while stack:
            item = stack.pop()
            if id(item) in _seen:
                continue
            _seen.add(id(item))
            cls = type(item)
            if is_user_class(cls, own_pkg):
                try:
                    parts.append((cls.__qualname__, self.user_class_source_hash(cls)))
                except SOURCE_RETRIEVAL_ERRORS:
                    pass
            held = getattr(item, "__dict__", None)
            if isinstance(held, dict):
                found = [
                    inner
                    for attr_val in held.values()
                    for inner in iter_contained(attr_val)
                    if is_user_class(type(inner), own_pkg)
                ]
                stack.extend(reversed(found))
        return parts
