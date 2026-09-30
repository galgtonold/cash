"""Randomness as an input: the seed epoch in the key, draws replayed on a
hit, and the warnings about unseeded draws."""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import re
import sys
import textwrap
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..analysis.annotations import parse_annotation_line
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..tracking.randomness import (
    CashRandomnessWarning,
    RandomnessDetector,
    capture_rng_state,
    describe_random_call,
    get_seeding_rng_modules,
    restore_rng_state,
    rng_modules_changed,
    seed_epoch_component,
    seeded_rng_modules,
    watch_seeds,
)

if TYPE_CHECKING:
    from .backend_slot import BackendSlot
    from .registry import FunctionRegistry
    from .reporting import Notices

logger = logging.getLogger(__name__)

#: Seeding calls, by the last segment of their dotted name. ``seed`` alone is
#: too common a method name, so it only counts under a ``random`` prefix.
_SEEDING_CALLS = frozenset(
    {
        "default_rng",
        "RandomState",
        "Random",
        "manual_seed",
        "SeedSequence",
        "PCG64",
        "PCG64DXSM",
        "MT19937",
        "Philox",
        "SFC64",
    }
)


def _seed_access_path(node: ast.AST) -> tuple[str, tuple[tuple[str, Any], ...]] | None:
    """``settings.sim.seed`` -> ``("settings", (("attr", "sim"), ("attr", "seed")))``;
    ``opts["seed"]`` -> ``("opts", (("item", "seed"),))``; None for anything
    else (a call, a computed key, an expression)."""
    path: list[tuple[str, Any]] = []
    while True:
        if isinstance(node, ast.Attribute):
            path.append(("attr", node.attr))
            node = node.value
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            path.append(("item", node.slice.value))
            node = node.value
        else:
            break
    if not isinstance(node, ast.Name):
        return None
    return node.id, tuple(reversed(path))


def seed_parameters(src: str) -> dict[str, tuple[str, str, bool, tuple]]:
    """``{seed expression: (call, root name, root is a parameter, path)}``.

    For seeding calls whose seed is a parameter, or an attribute or
    constant-key item reached from a parameter or a module global:
    ``default_rng(seed)``, ``default_rng(settings.seed)``,
    ``default_rng(opts["seed"])``, ``default_rng(CONFIG.seed)``. A root the
    function assigns itself is a local, which cannot be read before the call,
    and is skipped.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return {}
    a = fn.args
    params = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    found: dict[str, tuple[str, str, bool, tuple]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        dotted = ast.unparse(node.func)
        last = dotted.rsplit(".", 1)[-1]
        if not (last in _SEEDING_CALLS or (last == "seed" and "random" in dotted)):
            continue
        seed_arg = (
            node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ("seed", "x", "a")), None)
        )
        if seed_arg is None:
            continue
        access = _seed_access_path(seed_arg)
        if access is None:
            continue
        root, path = access
        is_param = root in params
        if not is_param and root in assigned:
            continue
        expr = ast.unparse(seed_arg)
        found.setdefault(expr, (f"{dotted}({expr})", root, is_param, path))
    return found


#: Keywords that seed a library call (``random_state=``, polars' ``seed=``).
_SEED_KEYWORDS = frozenset({"random_state", "seed", "rng", "generator"})

#: A receiver named as a generator (`rng`, `self.random`, `gen`): its
#: ``.sample()`` is judged by the source scan, not taken for a DataFrame's.
_GENERATOR_NAME = re.compile(r"rng|random|gen|state", re.IGNORECASE)

#: ``obj.sample(...)`` passes ``random_state`` as its fifth positional
#: argument in pandas (``n, frac, replace, weights, random_state``).
_SAMPLE_SEED_POSITION = 4


def _random_state_param(callee: Any) -> tuple[int | None, inspect.Parameter | None] | None:
    """``(position, shuffle)`` of an installed library callable that takes
    ``random_state=None``: where it may be passed positionally (None when only
    by keyword) and its ``shuffle`` parameter, if any. None for anything
    else, the user's own code included."""
    from ..install_paths import is_installed_path

    module = sys.modules.get((getattr(callee, "__module__", None) or "").split(".")[0])
    file = getattr(module, "__file__", None)
    if not file or not is_installed_path(file):
        return None
    try:
        params = list(inspect.signature(callee).parameters.values())
    except (TypeError, ValueError):
        return None
    by_name = {p.name: p for p in params}
    rs = by_name.get("random_state")
    if rs is None or rs.default is not None:
        return None
    position = None
    if rs.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        position = params.index(rs)
    return position, by_name.get("shuffle")


def _shuffles(call: ast.Call, shuffle: inspect.Parameter | None) -> bool:
    """Does *call* leave shuffling on (``KFold(shuffle=True)``, a
    ``train_test_split`` without ``shuffle=False``)? Unknown counts as off: a
    warning must be true."""
    if shuffle is None:
        return True
    for kw in call.keywords:
        if kw.arg == "shuffle":
            return isinstance(kw.value, ast.Constant) and bool(kw.value.value)
    return bool(shuffle.default)


def unseeded_library_calls(func: Callable, src: str) -> list[tuple[str, int]]:
    """``(call, line)`` for each library call in *func* left without a seed.

    The draw is inside compiled library code, where the source scan cannot
    see it: ``train_test_split(X)``, ``KFold(shuffle=True)``,
    ``SGDClassifier()``, ``make_classification()``, ``df.sample(n=3)``. A
    library function or class that takes ``random_state=None`` counts when the
    call passes none (and does not turn ``shuffle`` off); so does
    ``obj.sample(...)`` without ``random_state=`` / ``seed=``. The notebook's
    rule for a fitted estimator, ``random_state is None``, is the same one.
    Lines are relative to *src*.
    """
    from ..analysis.ast_util import resolve_callee
    from ..analysis.purity_analyzer import build_namespace, local_import_map, resolve_local_import

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return []
    namespace = build_namespace(func)
    for local, (module_name, prefix) in local_import_map(fn, func).items():
        obj = resolve_local_import(module_name, prefix, None)
        if obj is not None:
            namespace[local] = obj
    # Names the body binds to a generator (`r = random.Random(0)`): their
    # `.sample()` is a generator's draw, which the source scan judges.
    generators = {
        target.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        for target in node.targets
        if isinstance(target, ast.Name)
        and ast.unparse(node.value.func).rsplit(".", 1)[-1] in (*_SEEDING_CALLS, "check_random_state")
    }
    found: list[tuple[str, int]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or any(kw.arg is None for kw in node.keywords):
            continue  # `**kwargs` may carry the seed
        if any(kw.arg in _SEED_KEYWORDS for kw in node.keywords):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        func_node = node.func
        if isinstance(func_node, ast.Attribute) and func_node.attr == "sample":
            receiver = resolve_callee(func_node.value, namespace)
            if isinstance(receiver, types.ModuleType) or len(node.args) > _SAMPLE_SEED_POSITION:
                continue  # `random.sample` is the source scan's; a seed passed by position
            spelled = ast.unparse(func_node.value)
            last = spelled.rsplit(".", 1)[-1]
            if spelled in generators or _GENERATOR_NAME.search(last):
                continue
            found.append((f"{spelled if len(spelled) <= 24 else ''}.sample()", node.lineno))
            continue
        callee = resolve_callee(func_node, namespace)
        if callee is None or not callable(callee):
            continue
        param = _random_state_param(callee)
        if param is None:
            continue
        position, shuffle = param
        if position is not None and len(node.args) > position - (1 if isinstance(callee, type) else 0):
            continue  # passed by position (a class's signature has no `self`)
        if not _shuffles(node, shuffle):
            continue
        found.append((f"{ast.unparse(func_node)}()", node.lineno))
    return found


_SEED_UNREADABLE = object()


def read_seed(value: Any, path: tuple) -> Any:
    """Follow *path* from *value* WITHOUT running user code, or `_SEED_UNREADABLE`.

    Attributes through ``inspect.getattr_static``: a plain instance or class
    attribute is read, a property or other descriptor is not evaluated. Items
    only from a plain mapping. This runs on every call, so it may not call
    anything the user wrote.
    """
    for kind, key in path:
        if kind == "attr":
            try:
                value = inspect.getattr_static(value, key)
            except AttributeError:
                return _SEED_UNREADABLE
            if hasattr(type(value), "__get__") and not isinstance(
                value, (types.FunctionType, types.BuiltinFunctionType)
            ):
                return _SEED_UNREADABLE  # a property or descriptor
        elif isinstance(value, (dict, types.MappingProxyType)):
            value = value.get(key, _SEED_UNREADABLE)
            if value is _SEED_UNREADABLE:
                return value
        else:
            return _SEED_UNREADABLE
    return value


def rng_marker_key(func_name: str) -> str:
    """Backend key for the "this function draws" verdict."""
    return f"cash:rngdraw:{func_name}"


def capture_rng_pre_state() -> dict | None:
    """Snapshot the global RNG streams, or None if unavailable."""
    try:
        return capture_rng_state()
    except (TypeError, AttributeError):  # pragma: no cover
        return None


def replay_rng_state(metadata: Any) -> None:
    """Put the global RNG where the computed call left it (see
    :meth:`RngWatch.replay_parts`), when it is where that call started."""
    replay = getattr(metadata, "rng_replay", None) or {}
    post, pre = replay.get("rng_post"), replay.get("rng_pre")
    if not post or not pre:
        return
    try:
        # Only the streams the body advanced, and only while each is where
        # that body found it. Every other module is left alone: a process
        # seeds `random` from the OS at import, so comparing all of them
        # would refuse every replay.
        advanced = rng_modules_changed(pre, post)
        if not advanced:
            return
        live = capture_rng_state()
        if any(m not in live for m in advanced):
            return
        if rng_modules_changed({m: pre[m] for m in advanced}, {m: live[m] for m in advanced}):
            return
        restore_rng_state({m: post[m] for m in advanced})
    except Exception:  # a replay must never break a hit
        logger.debug("[CORE] could not replay the RNG state of a hit", exc_info=True)


class RngWatch:
    """Global random number generators: the seed epoch in the key of a function
    that draws, replaying where a hit leaves them, and the unseeded-randomness
    warnings."""

    def __init__(self, registry: FunctionRegistry, backend_slot: BackendSlot, notices: Notices) -> None:
        self._registry = registry
        self._backend_slot = backend_slot
        self._notices = notices
        #: func_name -> the global streams its own body seeds (`note_self_seeding`).
        self._self_seeded: dict[str, set[str]] = {}
        watch_seeds()

    def fold_rng_epoch(self, func_name: str, state_hash: str) -> str:
        """Fold the current seed epoch into the key, for RNG-drawing functions.

        A function that draws from the global stream has an input the key never
        saw. Change ``np.random.seed(12345)`` to ``seed(999)``, re-run, and the
        model trained under the old seed came straight back, silently, with a
        green badge -- on the exact idiom ``cash.help()`` rule 4 recommends.

        Deliberately narrow on three axes:

        * Only functions OBSERVED to draw (``CachedFunction.rng_modules``), so seeding
          the stream does not invalidate functions that never read it.
        * Only the *epoch*, never the raw RNG state -- the state advances on
          every draw, so keying on it would miss forever.
        * Empty when the module is unseeded, so an unseeded sample keeps being
          replayed. That is the freeze contract, and it is what makes caching an
          expensive unseeded draw still worth it.

        The verdict is learned on a miss, so the call that first reveals the
        draw has already been stored under an epoch-free key; the next call
        recomputes once and is stable from then on.
        """
        watch_seeds()
        cf = self._registry.cached.get(func_name)
        modules = cf.rng_modules if cf is not None else None
        if modules is None:
            modules = self._load_rng_draw_marker(func_name)
        if not modules:
            return state_hash
        component = seed_epoch_component(self._stream_inputs(func_name, modules))
        if not component:
            return state_hash
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    def _load_rng_draw_marker(self, func_name: str) -> set[str]:
        """Read the persisted draw verdict, caching the answer for this process.

        The verdict is learned by OBSERVING a call, so it lives in memory -- and
        a kernel restart or a fresh `python run.py` throws it away. That is fatal
        for the case this exists to fix: restart-and-run-all gets exactly one
        call per function, so an in-memory-only verdict is never applied and the
        stale value comes straight back.

        A tiny per-function marker survives the process and can be read BEFORE
        the real key is built, which the entry's own metadata cannot (that would
        need the key it is supposed to inform). One backend read per function per
        process; misses are remembered as empty so it is not retried.
        """
        cf = self._registry.cached.get(func_name)
        if cf is not None and cf.rng_modules is not None:
            return cf.rng_modules
        modules: set[str] = set()
        try:
            stored = self._backend_slot.backend.get(rng_marker_key(func_name))
            # Backends answer with ``(metadata, value)``; unwrap before reading.
            # Treating the pair itself as the payload silently yielded an empty
            # set, so every restart re-learned nothing and the stale value came
            # back -- the whole point of persisting the marker.
            if isinstance(stored, tuple) and len(stored) == 2 and isinstance(stored[0], dict):
                stored = stored[1]
            if isinstance(stored, (set, frozenset, list, tuple)):
                modules = {m for m in stored if isinstance(m, str)}
        except Exception:  # noqa: BLE001 - a marker miss must never break a call
            modules = set()
        if cf is not None:
            cf.rng_modules = modules
        return modules

    def _store_rng_draw_marker(self, func_name: str, modules: set[str]) -> None:
        """Persist the verdict so the next process applies it on its first call."""
        try:
            self._backend_slot.backend.set(rng_marker_key(func_name), set(modules))
        except Exception:  # noqa: BLE001 - best effort; correctness degrades to today's
            logger.debug("could not persist RNG draw marker for %s", func_name)

    def note_draw(self, func_name: str, pre_state: dict | None) -> bool:
        """Record which global RNG modules *func_name* just advanced."""
        if pre_state is None:
            return False
        try:
            changed = rng_modules_changed(pre_state, capture_rng_state())
        except (TypeError, AttributeError):  # pragma: no cover
            return False
        # A module merely imported by the call is newly present rather than
        # advanced; only count streams that already existed.
        drew = {m for m in changed if m in pre_state}
        if not drew:
            return False
        cf = self._registry.cached.get(func_name)
        if cf is None:
            return False
        if cf.rng_modules is None:
            cf.rng_modules = set()
        known = cf.rng_modules
        newly = bool(drew - known)
        if newly:
            known.update(drew)
            self._store_rng_draw_marker(func_name, known)
        if not newly:
            return False
        # Only report "newly seen" -- which suppresses this call's write -- when a
        # drawn module is actually SEEDED. An unseeded draw has no epoch that can
        # change, so its frozen value is correct from the first call; skipping the
        # write there would redraw and break the freeze-from-first-call contract.
        return bool(self._stream_inputs(func_name, drew) & seeded_rng_modules())

    def _stream_inputs(self, func_name: str, modules: set[str]) -> set[str]:
        """The streams of *modules* whose position is an input of *func_name*:
        not one its own body seeds before drawing, since the seed, not where
        the caller left the stream, decides what it draws."""
        own = self._self_seeded.get(func_name)
        return set(modules) - own if own else set(modules)

    def note_self_seeding(self, func: Callable, func_name: str) -> None:
        """Remember which global streams *func*'s own source seeds."""
        try:
            src = textwrap.dedent(inspect.getsource(func))
            own = get_seeding_rng_modules(src)
        except Exception:  # noqa: BLE001 - no source: every drawn stream stays an input
            return
        if own:
            self._self_seeded[func_name] = set(own)

    def replay_parts(self, drew: bool, pre_state: dict | None) -> dict:
        """What a later hit needs to leave the RNG where this call left it.

        A hit never runs the body, so the stream it advanced stays where it was
        and the CALLER's next draw returns what the function drew: with
        ``np.random.seed(0)``, the draw after a hit WAS the cached value. The
        notebook path replays the
        recorded state; this is the same for the decorator.

        Both ends are recorded. Replaying the post-state is only right when the
        stream is where it was when the body ran, so the pre-state is what a hit
        checks first -- a program that drew somewhere else in between is left
        alone rather than rewound.
        """
        if not drew or pre_state is None:
            return {}
        try:
            return {"rng_pre": pre_state, "rng_post": capture_rng_state()}
        except Exception:  # noqa: BLE001 - never break a call over this
            return {}

    def warn_unseeded_randomness(
        self,
        func: Callable,
        func_name: str,
        allow_random: bool,
    ) -> None:
        """Warn once if *func*'s source draws from an unseeded RNG.

        Otherwise ``@cash.cache`` would freeze a non-deterministic result
        forever with nothing on screen to say so. The decorator and the
        notebook path share ONE detector — :class:`~cash.tracking.randomness.RandomnessDetector`, reused
        verbatim — so "what counts as unseeded" cannot drift between them.

        Runs at DECORATION time, once per function. The analysis is a pure
        function of the source, so there is no reason to pay for it per call,
        and ``cache()`` already reads the source anyway (``FunctionRegistry.register`` ->
        ``callable_identity``), which warms ``linecache`` for us.

        A fresh detector is used per function rather than one shared across the
        instance. The detector's seed-tracking is *session*-scoped, which is
        right for a notebook (cells run top-to-bottom in one namespace) but
        wrong here: decoration order is not call order, so letting a
        ``np.random.seed(0)`` inside function A silence function B would be
        unsound. Per-function analysis keeps the verdict a property of the
        source we are actually looking at.

        Silent when:

        * ``allow_random=True``, or the notebook's ``# @cash:allow-random``
          appears in the function's own source (same directive vocabulary,
          parsed by the same ``parse_annotation_line``);
        * the RNG is seeded — the whole point, and the reason a seeded draw
          must not be flagged;
        * the source cannot be read (``exec``/REPL-defined functions). The
          purity analyzer has the identical blind spot and treats it the same
          way: no source, no claim.
        """
        # A seed the caller sets must be seen from now on (`watch_seeds`), and
        # the module that imports numpy has usually just been imported.
        watch_seeds()
        self.note_self_seeding(func, func_name)
        if allow_random:
            return

        try:
            src_lines, first_lineno = inspect.getsourcelines(func)
        except SOURCE_RETRIEVAL_ERRORS:
            # No retrievable source (exec'd, REPL, C function). Staying silent
            # is the conservative choice: we cannot see a draw, so we cannot
            # honestly claim there is one.
            return
        src = textwrap.dedent("".join(src_lines))

        # Honour the notebook's in-source opt-out too. Users coming from
        # ``%cash_on`` reach for the comment, and the source is already in hand.
        for line in src.splitlines():
            ann = parse_annotation_line(line)
            if ann is not None and ann.allow_random:
                return

        # A seeding call fed by a PARAMETER -- `default_rng(seed)` -- counts as
        # seeded to the detector, but whether it is depends on the call:
        # `def simulate(params, seed=None)` draws from OS entropy whenever the
        # caller leaves the seed out, and R Monte Carlo replicates came back
        # identical with nothing said. Note which parameters, and
        # check their bound value per call.
        seed_params = seed_parameters(src)
        if seed_params:
            cf = self._registry.cached.get(func_name)
            if cf is not None:
                cf.seed_params = seed_params

        try:
            unseeded, _messages, _has_seed = RandomnessDetector().analyze_code(src)
        except Exception:  # pragma: no cover - the scan must never break caching
            logger.debug("randomness scan failed for %s", func_name, exc_info=True)
            return

        if not unseeded:
            self._warn_unseeded_library_calls(func, func_name, src, first_lineno)
            return

        call = unseeded[0]
        extra = ""
        if len(unseeded) > 1:
            extra = f" (+{len(unseeded) - 1} more unseeded call(s) in this function)"
        # ``call.lineno`` is relative to the source we handed the detector, which
        # starts at the function's first line. Rebase it onto the file so the
        # number in the message matches what the user's editor shows.
        # ``getsourcelines`` returns 0 for sources it cannot place; keep the
        # relative number rather than reporting a nonsense negative line.
        abs_lineno = call.lineno + first_lineno - 1 if first_lineno else call.lineno

        # ASCII only: this lands in a terminal whose codepage may not be UTF-8.
        # A draw from the global stream may be seeded by the caller, which the
        # source cannot show: say what happens in each case (`watch_seeds`).
        then = "The first"
        if call.carrier is None and call.module in ("random", "numpy.random", "np.random", "numpy"):
            then = (
                "A seed set with random.seed() or np.random.seed() after this "
                "decoration, by the caller too, is part of the key; without one, the first"
            )
        message = (
            f"@cash.cache on {func_name}: Unseeded randomness detected: "
            f"{describe_random_call(call)} at line {abs_lineno}{extra}. "
            f"{then} call's result is cached and replayed on every later "
            f"call - the RNG is never consulted again, so the value is frozen "
            f"and not reproducible across a cleared cache."
        )
        # ``Notices.warn_once`` gives one warning per decorated function for the life
        # of this Cash instance, and it also
        # files the message into ``f.cache_info()['warnings']`` so it stays
        # discoverable if the user missed the stderr emission.
        self._notices.warn_once(
            CashRandomnessWarning,
            func_name,
            "",
            message,
            code="RANDOM-UNSEEDED",
            fix="seed the RNG to make the value reproducible, leave the "
            "function undecorated for a genuinely fresh draw, or pass "
            "@cash.cache(allow_random=True) to keep it frozen on purpose.",
        )

    def _warn_unseeded_library_calls(self, func: Callable, func_name: str, src: str, first_lineno: int) -> None:
        """RANDOM-UNSEEDED for a library call that draws without a seed
        (`unseeded_library_calls`): ``train_test_split(X)`` or
        ``SGDClassifier()`` in a cached body froze the first split or fit
        with nothing said, while the notebook warned about the same fit."""
        try:
            calls = unseeded_library_calls(func, src)
        except Exception:  # the scan must never break caching
            logger.debug("library randomness scan failed for %s", func_name, exc_info=True)
            return
        if not calls:
            return
        call, line = calls[0]
        abs_lineno = line + first_lineno - 1 if first_lineno else line
        extra = f" (+{len(calls) - 1} more)" if len(calls) > 1 else ""
        self._notices.warn_once(
            CashRandomnessWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: Unseeded randomness detected: "
            f"{call} without random_state at line {abs_lineno}{extra}. Whatever "
            f"it draws is unseeded, and the first call's result is cached and "
            f"replayed on every later call - the value is frozen and not "
            f"reproducible across a cleared cache.",
            code="RANDOM-UNSEEDED",
            fix="pass random_state=<int> (polars: seed=<int>) to make the value "
            "reproducible, leave the function undecorated for a genuinely "
            "fresh draw, or pass @cash.cache(allow_random=True) to keep it "
            "frozen on purpose.",
        )

    def warn_if_seed_is_none(self, func: Callable, func_name: str, args: tuple, kwargs: dict) -> None:
        """RANDOM-UNSEEDED for a seed that is None in THIS call.

        The seed may be a parameter or read from one or from a module
        global: ``default_rng(settings.seed)`` with the field None froze one
        draw across processes and said nothing, while the bare
        ``seed=None`` parameter warned.
        """
        bound = None
        g = getattr(func, "__globals__", None) or {}
        for expr, (call, root, is_param, path) in sorted(
            getattr(self._registry.cached.get(func_name), "seed_params", {}).items()
        ):
            if is_param:
                if bound is None:
                    try:
                        bound = inspect.signature(func).bind(*args, **kwargs)
                        bound.apply_defaults()
                    except (TypeError, ValueError):
                        return
                if root not in bound.arguments:
                    continue
                value = read_seed(bound.arguments[root], path)
                origin = f"the parameter '{root}'" if not path else f"'{expr}'"
            else:
                if root not in g:
                    continue
                value = read_seed(g[root], path)
                origin = f"'{expr}'"
            if value is not None:
                continue
            fix = (
                f"pass a seed: {root}=i per replicate keeps each one reproducible and cacheable."
                if is_param and not path
                else f"set {expr} to an integer, or pass the seed as an argument."
            )
            self._notices.warn_once(
                CashRandomnessWarning,
                func_name,
                f"seed-param:{expr}",
                f"@cash.cache on {func_name}: {call} is seeded from "
                f"{origin}, which is None in this call, so the RNG "
                f"draws from OS entropy. The first call's result is cached and "
                f"replayed on every later call with the same arguments -- "
                f"repeated calls return the same 'random' value, and it is "
                f"not reproducible across a cleared cache.",
                code="RANDOM-UNSEEDED",
                fix=f"{fix} Or @cash.cache(allow_random=True) to keep the value frozen on purpose.",
            )
            return

    def warn_unseeded_estimator_result(
        self,
        func_name: str,
        result: Any,
        allow_random: bool,
    ) -> None:
        """Warn when a cached function RETURNS an unseeded fitted estimator.

        ``RngWatch.warn_unseeded_randomness`` reads the source, and
        ``decorator.md`` is right that this hazard is invisible to it:
        randomness inside sklearn's compiled ``.fit()`` is not in any AST. The
        notebook's statement path solves that by asking the LIVE object
        (``get_params()['random_state'] is None``) rather than the source; the
        decorator path had no equivalent, so the recommended way to cache a fit
        was also the silent one.

        Three runs returned the identical model (first tree's `random_state`
        1200527474), with no warning and no badge marker, using the docs' own
        recipe: a report would have called the model "completely stable across
        random seeds".

        Same verdict rule as ``unseeded_estimator_fits``: unseeded iff
        ``get_params()`` HAS ``random_state`` and it is ``None``. A seed of any
        kind, or no such parameter at all (``LinearRegression``), is silent.
        Any failure is silent too -- an advisory must never break a call.
        """
        if allow_random:
            return
        # This runs on EVERY call, hits included, so it must stay cheap once it
        # has had its say. `Notices.warn_once` would dedupe the emission but not the
        # `get_params()` that precedes it, and sklearn's `get_params` walks the
        # signature -- a per-hit cost on exactly the functions people cache to
        # avoid paying for a fit. Check the same key first and leave.
        if self._notices.has_warned((CashRandomnessWarning, func_name, "_estimator_result", "RANDOM-UNSEEDED")):
            return
        get_params = getattr(result, "get_params", None)
        if get_params is None or not callable(get_params):
            return
        try:
            params = get_params()
            if params.get("random_state", "absent") is not None:
                return
        except Exception:  # noqa: BLE001 - advisory only; never break a call
            return

        self._notices.warn_once(
            CashRandomnessWarning,
            func_name,
            "_estimator_result",
            f"@cash.cache on {func_name}: returns a fitted estimator with "
            f"random_state=None. cash caches it, so every later call replays "
            f"that one fit - the model is frozen, not stable. Two genuine fits "
            f"would differ, and comparing runs cannot tell you otherwise.",
            code="RANDOM-UNSEEDED",
            fix="pass random_state=<int> to the estimator for a reproducible "
            "fit, leave the function undecorated for a genuinely fresh "
            "one, or pass @cash.cache(allow_random=True) to keep it frozen "
            "on purpose.",
        )
