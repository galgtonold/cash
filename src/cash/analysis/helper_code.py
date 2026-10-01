"""Which code a callable runs, and whether that code is the user's.

The helper walk keys every user function a cached function reaches. These
answer, for one callable: is its code the user's (`is_user_code`), what other
functions it runs besides its own code (`callable_layers`: the wrapped
function under a decorator, a partial's function, a singledispatch registry),
whether it is a test double with no code to key (`is_mock`), and the name its
part of the key is stored under (`qualname_of`).
"""

from __future__ import annotations

import functools
import inspect
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..install_paths import in_own_package, top_package
from ..tracking.function_tracker import is_local_module


def _defining_module(obj: Any) -> Any:
    """The module *obj*'s code was written in.

    For a function, its ``__globals__`` say so. ``__module__`` does not
    always: ``functools.wraps`` copies the WRAPPED function's ``__module__``
    onto the wrapper, so a library's wrapper (tenacity's, torch's) claimed to
    be user code and a user's wrapper claimed to be the helper's module.
    """
    if isinstance(obj, types.FunctionType):
        name = obj.__globals__.get("__name__")
        module = sys.modules.get(name) if isinstance(name, str) else None
        if module is not None:
            return module
    return inspect.getmodule(obj)


def own_code_is_user(obj: Any, root_module: str | None) -> bool:
    """`is_user_code`, for callables that may be wrappers."""
    try:
        return is_user_code(obj, root_module)
    except Exception:  # noqa: BLE001 - a probe of arbitrary objects
        return False


#: How many objects `callable_layers` may visit for one callable. Not a depth
#: or a count real code meets: every layer is followed, however deep, and the
#: seen set ends cycles. What can pass it is an object that hands out a NEW
#: wrapper on every read, where the walk would never end; then it raises
#: `UnwalkableLayers` rather than leave the rest out of the key.
_LAYER_LIMIT = 5_000


_NO_LAYER = object()


class UnwalkableLayers(Exception):
    """`callable_layers` could not find every function a callable runs; the
    message says why. The helper walk turns it into ``PurityReport.unwalkable``,
    so the call runs uncached instead of keyed without them."""


def _function_like(value: Any) -> bool:
    return isinstance(value, (types.FunctionType, types.MethodType, functools.partial)) or (
        callable(value)
        and not isinstance(value, (type, types.ModuleType, types.BuiltinFunctionType))
        and hasattr(value, "__wrapped__")
    )


def _layer_candidates(value: Any) -> Iterator[Any]:
    """The objects *value* holds that it may run, in the order it holds them."""
    if isinstance(value, types.MethodType):
        yield value.__func__
        return
    if isinstance(value, functools.partial):
        yield value.func
        return
    wrapped = getattr(value, "__wrapped__", None) if not isinstance(value, type) else None
    if wrapped is not None:
        yield wrapped
    # wrapt's proxies forward `__class__`, so one passes for a plain
    # function below; the user's wrapper function sits here.
    wrapper = getattr(value, "_self_wrapper", None) if not isinstance(value, type) else None
    if wrapper is not None:
        yield wrapper
    if isinstance(value, types.FunctionType):
        for cell in value.__closure__ or ():
            try:
                inner = cell.cell_contents
            except ValueError:
                continue
            if _function_like(inner):
                yield inner
        attrs = getattr(value, "__dict__", None) or {}
    else:
        if callable(value) and not isinstance(value, (type, types.ModuleType)):
            call = getattr(type(value), "__call__", None)
            if isinstance(call, types.FunctionType):
                yield call
        try:
            attrs = dict(vars(value))
        except TypeError:
            attrs = {}
    for key, attr in list(attrs.items()):
        if key == "__wrapped__":
            continue
        if _function_like(attr):
            yield attr
        elif isinstance(attr, (dict, types.MappingProxyType)):
            for item in list(attr.values()):
                if _function_like(item):
                    yield item


def _expand(value: Any, obj: Any) -> Iterator[Any]:
    """`_layer_candidates` of *value*, a layer of *obj*, read in full now."""
    try:
        return iter(list(_layer_candidates(value)))
    except Exception as e:
        raise UnwalkableLayers(
            f"cash could not look inside {type(value).__qualname__}, which {qualname_of(obj)} "
            f"runs ({type(e).__name__}: {e})"
        ) from e


def callable_layers(obj: Any) -> list[Any]:
    """The functions *obj* will run besides its own code, outermost first.

    A decorated helper is two or more functions, and the key has to see all of
    them: with ``functools.wraps`` only the wrapped function was followed, so
    an edit to the wrapper's body was served stale; without it, only the
    wrapper was, so an edit to the wrapped function was. Followed:

    * ``__wrapped__`` (``functools.wraps``, ``update_wrapper``, ``lru_cache``);
    * function-valued closure cells (a wrapper written without ``wraps``, and
      the ``decorator`` package, which keeps the caller in a closure);
    * a bound method's ``__func__``, a ``functools.partial``'s ``func``;
    * a callable instance's class ``__call__`` and its function-valued
      attributes (``np.vectorize.pyfunc``, a class-based decorator's
      ``self.fn``, ``toolz.curry``'s partial, wrapt's ``_self_wrapper``);
    * a function's own ``__dict__`` values and mappings of functions
      (``functools.singledispatch``'s ``registry``).

    Returns FUNCTION objects only, deduplicated, never *obj* itself; whether
    each is user code is the caller's decision. Every layer is followed,
    however deep and however many (a singledispatch registry of 40
    implementations, eight stacked decorators): one left out was not keyed,
    and editing it served the old result. Raises `UnwalkableLayers` when
    the layers do not end, or an object cannot be looked into.
    """
    found: list[Any] = []
    seen: set[int] = {id(obj)}
    # Every object visited stays alive until the walk ends, so an id in
    # ``seen`` cannot be handed to a new object.
    keep: list[Any] = [obj]

    # Depth first, each object's layers in the order it holds them: the walk
    # order names helpers that share a name (``PurityAnalyzer``), so it must
    # not depend on anything but the objects.
    stack = [_expand(obj, obj)]
    while stack:
        value = next(stack[-1], _NO_LAYER)
        if value is _NO_LAYER:
            stack.pop()
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        keep.append(value)
        if is_mock(value):
            # A mock makes a new attribute on every read, so its layers never
            # end; it has no code to key (the walk says so where it meets one).
            continue
        if len(keep) > _LAYER_LIMIT:
            raise UnwalkableLayers(
                f"the functions {qualname_of(obj)} runs do not end (over {_LAYER_LIMIT} wrappers; "
                "an object that makes a new wrapper on every read can cause this)"
            )
        if isinstance(value, types.FunctionType):
            found.append(value)
        stack.append(_expand(value, obj))
    return found


def is_user_code(callee: Any, root_module: str | None) -> bool:
    """Decide whether to recurse into *callee* during purity analysis.

    The boundary rule from the design discussion: user code is
    anything that (a) shares the cached function's top-level package
    OR (b) lives outside stdlib/site-packages.

    Args:
        callee: Resolved callable from the cached function's globals.
        root_module: ``__module__`` of the cached function. Used for
            the same-top-level-package shortcut.

    Returns:
        True when the analyzer should attempt to read source and
        recurse into *callee*. False for library code we trust
        unless explicitly marked stateful.
    """
    module = _defining_module(callee)
    if module is None:
        return False
    if in_own_package(getattr(module, "__name__", None), top_package(root_module)):
        return True
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def is_mock(obj: Any) -> bool:
    """A ``unittest.mock`` object (``pytest-mock`` uses the same classes).

    Checked before anything reads an attribute from a callee: a mock answers
    every attribute truthily, so ``_cash_cached`` or a purity marker would
    read as set. Never imports ``unittest.mock`` itself.
    """
    module = sys.modules.get("unittest.mock")
    if module is None:
        return False
    if isinstance(obj, module.NonCallableMock):
        return True
    # `create_autospec` / `patch(..., autospec=True)` on a function makes a
    # real function that carries its mock. Walked as code, it led into the
    # TEST's side_effect, analysed as production code -- an `__import__` in a
    # fake raised CashImpureFunctionError out of the test.
    return isinstance(obj, types.FunctionType) and isinstance(obj.__dict__.get("mock"), module.NonCallableMock)


def qualname_of(func: Callable[..., Any]) -> str:
    """Name a callable for ``helper_source_hashes``.

    ``__main__`` is resolved the same way ``Cash.get_func_key`` resolves it.
    These keys are folded into the state hash as ``helper:{qual}:{digest}``, so
    leaving this one alone made a direct run and an import disagree on the KEY
    while agreeing on the digest -- the function name matched, the state hash
    did not, and the entry still missed. Deliberately NOT applied to
    ``helper_paths``, whose module string is looked up in ``sys.modules`` at
    runtime and has to stay ``__main__`` to resolve.
    """
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<callable>")
    if isinstance(func, types.FunctionType) and hasattr(func, "__wrapped__"):
        # `functools.wraps` copied the wrapped function's names onto this one,
        # so both halves of a decorated helper answered to the same name and
        # the walk, which visits each name once, followed only one of them.
        # Name the wrapper by where its code was written. Only for wrappers:
        # every other function's name is unchanged, and so are their keys.
        module = func.__globals__.get("__name__") or module
        code = func.__code__
        qualname = getattr(code, "co_qualname", None) or f"{code.co_name}@wrapper"
    if module in MAIN_MODULE_NAMES:
        module = resolve_main_module(func)
    return f"{module}.{qualname}"
