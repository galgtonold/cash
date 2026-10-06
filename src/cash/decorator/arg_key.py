"""Which part of a call's arguments its key is made of: every argument (the
default), every argument but the ignored ones (``ignore=[...]`` and
``cash.Ignore``), or what a ``key=`` function returns for the call."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

from ..effects import EffectKind, classify_call
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from .cached_function import follow_passthrough
from .call_state import KeyBuildFailed

__all__ = [
    "ArgKey",
    "Ignore",
    "arg_key_spec",
    "key_function_impurities",
    "keyed_arguments",
]


class _IgnoreMarker:
    """The metadata `Ignore` puts on an annotation. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "cash.Ignore"


_IGNORE = _IgnoreMarker()

_T = TypeVar("_T")

#: Leave a parameter out of the cache key: ``debug: cash.Ignore[bool] = False``.
#: An ``Annotated`` alias, so a type checker sees ``bool``.
#: ``Annotated[bool, cash.Ignore]`` says the same.
Ignore = Annotated[_T, _IGNORE]


@dataclass(frozen=True)
class ArgKey:
    """How one cached function's arguments become the argument part of its key.

    Exactly one of the two is set: *key_fn*, whose return value stands for
    the arguments, or *ignored*, the parameters left out.
    """

    key_fn: Callable[..., Any] | None = None
    ignored: frozenset[str] = frozenset()

    @property
    def how(self) -> str:
        """How the user asked for it, as explain() says it."""
        if self.key_fn is not None:
            return "key="
        return "ignored parameters (" + ", ".join(sorted(self.ignored)) + ")"


def _signature_owner(func: Any) -> Any:
    """The object whose own signature `call_signature` reads: *func*, or what
    the ``*args, **kwargs`` wrappers on it wrap (`follow_passthrough`). Its
    annotations are the parameters'."""
    try:
        return follow_passthrough(func)[0]
    except (TypeError, ValueError):
        return func


def _globals_of(obj: Any) -> dict[str, Any]:
    """The module namespace *obj*'s annotations are written in."""
    for _ in range(8):
        if isinstance(obj, types.FunctionType):
            return obj.__globals__
        inner = getattr(obj, "__func__", None) or getattr(obj, "func", None)
        if inner is None or inner is obj:
            break
        obj = inner
    return {}


def _marks_ignore(hint: Any) -> bool:
    """Is *hint* ``Ignore[X]`` or ``Annotated[X, Ignore]``, also inside an
    ``Optional[...]`` (Python 3.10 wraps a ``None`` default's hint in one)?"""
    origin = typing.get_origin(hint)
    if origin is Annotated:
        return any(m is _IGNORE or m is Ignore for m in getattr(hint, "__metadata__", ()))
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        return any(_marks_ignore(a) for a in typing.get_args(hint))
    return False


def annotated_ignores(func: Any, sig: inspect.Signature) -> frozenset[str]:
    """The parameters of *func* annotated ``cash.Ignore``.

    The annotations are resolved with ``typing.get_type_hints``, so string
    annotations (``from __future__ import annotations``) count. When the
    hints as a whole cannot be resolved (one names a class only imported under
    ``TYPE_CHECKING``), each parameter's annotation is evaluated on its own.
    One that cannot be evaluated and mentions ``Ignore`` raises: cash cannot
    tell whether it marks the parameter, and guessing would key the function
    differently from what its annotation says.

    Raises:
        TypeError: an annotation that mentions ``Ignore`` cannot be evaluated.
    """
    owner = _signature_owner(func)
    if not any(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()):
        return frozenset()
    try:
        hints: dict[str, Any] | None = typing.get_type_hints(owner, include_extras=True)
    except Exception:  # noqa: BLE001 - any failure: resolve per parameter below
        hints = None
    namespace = _globals_of(owner)
    found = set()
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            continue
        if hints is not None and name in hints:
            hint = hints[name]
        elif isinstance(ann, str):
            try:
                hint = eval(ann, dict(namespace), None)  # what get_type_hints does
            except Exception as exc:  # the annotation names something not importable here
                if re.search(r"\bIgnore\b", ann):
                    raise TypeError(
                        f"@cash.cache on {getattr(func, '__qualname__', func)!r}: the annotation of parameter "
                        f"{name!r}, {ann!r}, cannot be evaluated ({type(exc).__name__}: {exc}), so cash cannot "
                        f"tell whether it leaves the parameter out of the key. Make the names in it importable "
                        f"where the function is defined, or use @cash.cache(ignore=[{name!r}]) instead."
                    ) from exc
                continue
        else:
            hint = ann
        if _marks_ignore(hint):
            found.add(name)
    return frozenset(found)


def _describe(func: Any, sig: inspect.Signature) -> str:
    """``load(path, debug)``: the function as the user wrote its parameters."""
    prefix = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}
    names = ", ".join(prefix.get(p.kind, "") + p.name for p in sig.parameters.values())
    return f"{getattr(func, '__name__', type(func).__name__)}({names})"


def _ignore_names(ignore: Any) -> tuple[str, ...]:
    """``ignore=`` as a tuple of names. A single string is one name."""
    if isinstance(ignore, str):
        return (ignore,)
    if not isinstance(ignore, Iterable) or isinstance(ignore, (bytes, dict)):
        raise TypeError(f"@cash.cache: ignore= takes a list of parameter names, not {type(ignore).__name__}")
    names = tuple(ignore)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"@cash.cache: ignore= takes parameter names as strings, not {type(name).__name__}")
    return names


def arg_key_spec(func: Any, sig: inspect.Signature | None, key: Any, ignore: Any) -> ArgKey | None:
    """The `ArgKey` for *func* decorated with ``key=`` / ``ignore=`` and its
    ``cash.Ignore`` annotations, or None when every argument is keyed.

    Checked when the function is decorated, so a mistake raises there and
    never keys a call differently from what the user wrote.

    Raises:
        TypeError: *key* is not a plain callable, *ignore* is not a list of
            names, *func* has no signature to bind a call to, an annotation
            cannot be read, or ``key=`` is combined with an ignored parameter.
        ValueError: a name in *ignore* is not a parameter of *func*.
    """
    if key is not None:
        if not callable(key):
            raise TypeError(f"@cash.cache: key= takes a function, not {type(key).__name__}")
        if inspect.iscoroutinefunction(key) or inspect.isasyncgenfunction(key) or inspect.isgeneratorfunction(key):
            raise TypeError(
                "@cash.cache: key= takes a plain function that returns the key, not an async or generator "
                "function: it runs before every lookup, outside any event loop."
            )
    listed = _ignore_names(ignore) if ignore is not None else ()
    if sig is None:
        if key is not None or listed:
            raise TypeError(
                f"@cash.cache on {getattr(func, '__qualname__', func)!r}: key= and ignore= need the function's "
                f"signature to bind a call to, and this callable has none cash can read."
            )
        return None
    # A parameter a wrapper fills (`f(LOG, *args, **kwargs)`) is never the
    # caller's argument, so it is out of the key already: naming it is fine.
    try:
        injected = set(follow_passthrough(func)[1].parameters) - set(sig.parameters)
    except (TypeError, ValueError):
        injected = set()
    missing = [name for name in listed if name not in sig.parameters and name not in injected]
    if missing:
        raise ValueError(f"@cash.cache: ignore= names no parameter {missing[0]!r} in {_describe(func, sig)}")
    ignored = frozenset(listed) | annotated_ignores(func, sig)
    if key is not None and ignored:
        raise TypeError(
            f"@cash.cache on {_describe(func, sig)}: key= cannot be combined with ignored parameters "
            f"({', '.join(sorted(ignored))}). key= already decides the whole argument part of the key: "
            f"leave those arguments out of what it returns."
        )
    if key is not None:
        return ArgKey(key_fn=key)
    if ignored:
        return ArgKey(ignored=ignored)
    return None


def keyed_arguments(
    spec: ArgKey,
    sig: inspect.Signature | None,
    func_name: str,
    args: tuple,
    kwargs: dict,
    normalized: tuple[tuple, dict],
) -> tuple[tuple, dict]:
    """The arguments the key is built from, in the form `ArgHasher.hash_payload`
    and `CodeArgs.fold_code_args` take.

    With ``key=``, the key function's return value, as the one argument. With
    ignored parameters, *normalized* (`ArgHasher.normalize_call_args`) without
    them. A call that does not bind to the signature keeps every argument:
    the body raises on it anyway, and nothing is left out of a key unasked.

    Raises:
        KeyBuildFailed: the key function raised (KEY-FUNCTION-RAISED).
    """
    if sig is None:
        return normalized
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        return normalized
    if spec.key_fn is not None:
        try:
            value = spec.key_fn(*bound.args, **bound.kwargs)
        except Exception as exc:  # the user's key function can raise anything
            raise KeyBuildFailed(
                "KEY-FUNCTION-RAISED",
                f"@cash.cache on {func_name}: the key= function raised {type(exc).__name__} ({exc}), so the call "
                f"ran uncached.",
                "make the key function handle every call the function accepts.",
            ) from exc
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise KeyBuildFailed(
                "KEY-FUNCTION-RAISED",
                f"@cash.cache on {func_name}: the key= function returned an awaitable, which cannot be a key, "
                f"so the call ran uncached.",
                "make the key function a plain def that returns the key.",
            )
        return (value,), {}
    ignored = spec.ignored
    canon_args, canon_kwargs = normalized
    var_positional = next((p.name for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_POSITIONAL), None)
    if var_positional in ignored:
        canon_args = ()
    # A ``**kwargs`` entry is spelled ``name:key`` (`normalize_call_args`); a
    # parameter name never holds a colon.
    kept = {k: v for k, v in canon_kwargs.items() if k not in ignored and k.partition(":")[0] not in ignored}
    return canon_args, kept


#: What a key function must not read: anything that is not in its arguments.
_NONDETERMINISTIC = frozenset(
    {
        EffectKind.FILE_READ,
        EffectKind.NETWORK_READ,
        EffectKind.NETWORK,
        EffectKind.DB_READ,
        EffectKind.SUBPROCESS,
        EffectKind.CLOCK,
        EffectKind.ENVIRONMENT,
    }
)


def _lambdas_in(src: str) -> list[ast.Lambda]:
    """The lambdas written in *src*, with *src* as the text their offsets refer to.

    *src* is what ``inspect.getsource`` gave: a whole statement when it
    parses, else a fragment (one line of a ``for`` header, of a call spread
    over lines). A fragment is cut at each ``lambda`` and the longest piece
    from there that parses as an expression is read.
    """
    try:
        return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Lambda)]
    except SyntaxError:
        pass
    found: list[ast.Lambda] = []
    for match in re.finditer(r"\blambda\b", src):
        for end in range(len(src), match.start(), -1):
            piece = src[match.start() : end]
            try:
                tree = ast.parse("(" + piece + ")", mode="eval")
            except SyntaxError:
                continue
            first = next((n for n in ast.walk(tree) if isinstance(n, ast.Lambda)), None)
            if first is not None:
                first.cash_text = ast.get_source_segment("(" + piece + ")", first)  # type: ignore[attr-defined]
                first.cash_line = src.count("\n", 0, match.start()) + 1  # type: ignore[attr-defined]
                found.append(first)
            break
    return found


def _lambda_source(fn: types.FunctionType) -> str | None:
    """The text of the lambda *fn* alone. ``inspect.getsource`` gives the whole
    statement it is written in: for ``@cash.cache(key=lambda ...)``, the
    decorated function too. Lambdas on the line *fn* starts on whose
    parameters are *fn*'s; all of them when several match."""
    try:
        lines, first = inspect.getsourcelines(fn)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    src = textwrap.dedent("".join(lines))
    code = fn.__code__
    names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    line = code.co_firstlineno - first + 1
    found = []
    for node in _lambdas_in(src):
        text = getattr(node, "cash_text", None)
        at = getattr(node, "cash_line", node.lineno)
        params = tuple(a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
        if at == line and params == names:
            found.append(text if text is not None else ast.get_source_segment(src, node) or "")
    return "\n".join(found) if found else None


def _own_text(fn: Any) -> str | None:
    """*fn*'s own source, dedented, or None when it has none."""
    if not isinstance(fn, types.FunctionType):
        return None
    if fn.__name__ == "<lambda>":
        return _lambda_source(fn)
    try:
        lines, _first = inspect.getsourcelines(fn)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return textwrap.dedent("".join(lines))


def _scan_text(src: str, namespace: dict[str, Any]) -> list[str]:
    """What *src* reads that is not an argument: tracked calls and random draws."""
    # Local: the randomness detector pulls in the RNG machinery, which only a
    # decorated key function needs.
    from ..tracking.randomness import RandomnessDetector, describe_random_call

    found: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            effect = classify_call(node, namespace)
            if effect is not None and effect.kind in _NONDETERMINISTIC:
                found.append(f"{effect.name}()")
    try:
        unseeded, _messages, _seeded = RandomnessDetector().analyze_code(src)
    except Exception:  # noqa: BLE001 - the scan must never break decoration
        unseeded = []
    found.extend(describe_random_call(call) for call in unseeded)
    return found


def _named_functions(src: str, namespace: dict[str, Any]) -> list[types.FunctionType]:
    """The plain functions *src* names from *namespace*: the helpers a lambda calls."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[types.FunctionType] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            value = namespace.get(node.id)
            if isinstance(value, types.FunctionType) and value not in out:
                out.append(value)
    return out


def key_function_impurities(key_fn: Callable[..., Any]) -> list[str]:
    """What *key_fn* reads besides its arguments, as the static analysis sees it.

    Its own text is scanned for tracked calls (``open``, the clock, the
    environment, the network) and random draws; the functions a lambda
    names, and a ``def`` itself, also go through the purity analysis the
    decorator gives a cached body, which follows helpers. A file read
    through a reader cash does not name is caught on the first call instead
    (`KeyBuilder.check_key_function_reads`).
    """
    # Local: the analyzer is a heavy import for a module every key build uses.
    from ..analysis.purity_analyzer import get_analyzer
    from ..analysis.purity_report import ISSUE_AMBIENT_READ, ISSUE_IMPURE_CALL, ISSUE_NETWORK_READ

    fn = inspect.unwrap(key_fn) if isinstance(key_fn, types.FunctionType) else key_fn
    namespace = _globals_of(fn)
    text = _own_text(fn)
    found: list[str] = []
    analyzed: list[Any] = []
    if text is not None:
        found.extend(_scan_text(text, namespace))
        if getattr(fn, "__name__", "") == "<lambda>":
            analyzed = _named_functions(text, namespace)
        else:
            analyzed = [fn]
    for helper in analyzed:
        if helper is not fn:
            helper_text = _own_text(helper)
            if helper_text is not None:
                found.extend(_scan_text(helper_text, _globals_of(helper)))
        try:
            report = get_analyzer().analyze(helper)
        except Exception:  # noqa: BLE001 - the analysis must never break decoration
            continue
        for issue in report.issues:
            if issue.kind in (ISSUE_AMBIENT_READ, ISSUE_NETWORK_READ) or (
                issue.kind == ISSUE_IMPURE_CALL and issue.effect_kind in _NONDETERMINISTIC
            ):
                found.append(issue.description.split(" - ")[0])
        for kind, name in sorted(report.environment_reads):
            found.append(f"environment variable {name}" if kind == "env" else "the working directory")
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique
