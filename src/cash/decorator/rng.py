"""Randomness as an input: the seed epoch in the key, draws replayed on a
hit, and the warnings about unseeded draws."""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap
import types
from collections.abc import Callable
from typing import Any

from ..analysis.annotations import parse_annotation_line
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..tracking.randomness import (
    CashRandomnessWarning,
    RandomnessDetector,
    capture_rng_state,
    describe_random_call,
    restore_rng_state,
    rng_modules_changed,
    seed_epoch_component,
    seed_epochs,
)

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


class RngMixin:
    """The random-number generators a function draws from, as an input."""

    def _fold_rng_epoch(self, func_name: str, state_hash: str) -> str:
        """Fold the current seed epoch into the key, for RNG-drawing functions.

        A function that draws from the global stream has an input the key never
        saw. Change ``np.random.seed(12345)`` to ``seed(999)``, re-run, and the
        model trained under the old seed came straight back, silently, with a
        green badge -- on the exact idiom ``cash.help()`` rule 4 recommends.

        Deliberately narrow on three axes:

        * Only functions OBSERVED to draw (``_rng_drawing_funcs``), so seeding
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
        modules = self._rng_drawing_funcs.get(func_name)
        if modules is None:
            modules = self._load_rng_draw_marker(func_name)
        if not modules:
            return state_hash
        component = seed_epoch_component(modules)
        if not component:
            return state_hash
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    @staticmethod
    def _rng_marker_key(func_name: str) -> str:
        """Backend key for the "this function draws" verdict."""
        return f"cash:rngdraw:{func_name}"

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
        cached = self._rng_drawing_funcs.get(func_name)
        if cached is not None:
            return cached
        modules: set[str] = set()
        try:
            stored = self.backend.get(self._rng_marker_key(func_name))
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
        self._rng_drawing_funcs[func_name] = modules
        return modules

    def _store_rng_draw_marker(self, func_name: str, modules: set[str]) -> None:
        """Persist the verdict so the next process applies it on its first call."""
        try:
            self.backend.set(self._rng_marker_key(func_name), set(modules))
        except Exception:  # noqa: BLE001 - best effort; correctness degrades to today's
            logger.debug("could not persist RNG draw marker for %s", func_name)

    def _note_rng_draw(self, func_name: str, pre_state: dict | None) -> bool:
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
        known = self._rng_drawing_funcs.setdefault(func_name, set())
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
        return bool(drew & set(seed_epochs()))

    @staticmethod
    def _capture_rng_pre_state() -> dict | None:
        """Snapshot the global RNG streams, or None if unavailable."""
        try:
            return capture_rng_state()
        except (TypeError, AttributeError):  # pragma: no cover
            return None

    def _rng_replay_parts(self, drew: bool, pre_state: dict | None) -> dict:
        """What a later hit needs to leave the RNG where this call left it.

        A hit never runs the body, so the stream it advanced stays where it was
        and the CALLER's next draw returns what the function drew: with
        ``np.random.seed(0)``, the draw after a hit WAS the cached value (found
        attacking the decorator before round 26). The notebook path replays the
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

    @staticmethod
    def _replay_rng_state(metadata: Any) -> None:
        """Put the global RNG where the computed call left it (see
        :meth:`_rng_replay_parts`), when it is where that call started."""
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
        except Exception:  # noqa: BLE001 - a replay must never break a hit
            logger.debug("[CORE] could not replay the RNG state of a hit", exc_info=True)

    def _warn_unseeded_randomness(
        self,
        func: Callable,
        func_name: str,
        allow_random: bool,
    ) -> None:
        """Warn once if *func*'s source draws from an unseeded RNG.

        The decorator used to be completely silent here while the notebook path
        warned, so ``@cash.cache`` would freeze a non-deterministic result
        forever with nothing on screen to say so. The two paths now share ONE
        detector — :class:`~cash.tracking.randomness.RandomnessDetector`, reused
        verbatim — so "what counts as unseeded" cannot drift between them.

        Runs at DECORATION time, once per function. The analysis is a pure
        function of the source, so there is no reason to pay for it per call,
        and ``cache()`` already reads the source anyway (``_register_func`` ->
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
        # identical with nothing said (CAS-116). Note which parameters, and
        # check their bound value per call.
        seed_params = seed_parameters(src)
        if seed_params:
            self._seed_params[func_name] = seed_params

        try:
            unseeded, _messages, _has_seed = RandomnessDetector().analyze_code(src)
        except Exception:  # pragma: no cover - detector must never break caching
            logger.debug("randomness scan failed for %s", func_name, exc_info=True)
            return

        if not unseeded:
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
        message = (
            f"@cash.cache on {func_name}: Unseeded randomness detected: "
            f"{describe_random_call(call)} at line {abs_lineno}{extra}. "
            f"The first call's result is cached and replayed on every later "
            f"call - the RNG is never consulted again, so the value is frozen "
            f"and not reproducible across a cleared cache."
        )
        # ``_warn_once`` keys on (category, func_name, "") -> one warning per
        # decorated function for the life of this Cash instance, and it also
        # files the message into ``f.cache_info()['warnings']`` so it stays
        # discoverable if the user missed the stderr emission.
        self._warn_once(
            CashRandomnessWarning,
            func_name,
            "",
            message,
            code="RANDOM-UNSEEDED",
            fix="seed the RNG to make the value reproducible, leave the "
            "function undecorated for a genuinely fresh draw, or pass "
            "@cash.cache(allow_random=True) to keep it frozen on purpose.",
            # 4, not 3: the chain from ``warnings.warn`` is
            # ``warn_diagnostic_message -> _warn_once ->
            # _warn_unseeded_randomness -> cache -> user``, so 3 blamed
            # ``cache`` itself and printed a line inside core.py. Measured
            # against a decoration on a known line; a reader whose warning
            # points into Cash cannot act on it, which is the whole point of
            # this diagnostic. Single caller, so the depth is fixed.,
        )

    def _warn_if_seed_is_none(self, func: Callable, func_name: str, args: tuple, kwargs: dict) -> None:
        """RANDOM-UNSEEDED for a seed that is None in THIS call.

        The seed may be a parameter (CAS-116) or read from one or from a module
        global: ``default_rng(settings.seed)`` with the field None froze one
        draw across processes and said nothing (round 18), while the bare
        ``seed=None`` parameter warned.
        """
        bound = None
        g = getattr(func, "__globals__", None) or {}
        for expr, (call, root, is_param, path) in sorted(self._seed_params.get(func_name, {}).items()):
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
            self._warn_once(
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

    def _warn_unseeded_estimator_result(
        self,
        func_name: str,
        result: Any,
        allow_random: bool,
    ) -> None:
        """Warn when a cached function RETURNS an unseeded fitted estimator.

        ``_warn_unseeded_randomness`` reads the source, and
        ``decorator.md`` is right that this hazard is invisible to it:
        randomness inside sklearn's compiled ``.fit()`` is not in any AST. The
        notebook's statement path solves that by asking the LIVE object
        (``get_params()['random_state'] is None``) rather than the source; the
        decorator path had no equivalent, so the recommended way to cache a fit
        was also the silent one.

        Reported in round 14: three runs returned the identical model (first
        tree's `random_state` 1200527474), no warning, no badge marker, using
        the docs' own recipe. The tester's words for the harm are the reason
        this exists -- "I would have written 'the model is completely stable
        across random seeds' in a report."

        Same verdict rule as ``_unseeded_estimator_fits``: unseeded iff
        ``get_params()`` HAS ``random_state`` and it is ``None``. A seed of any
        kind, or no such parameter at all (``LinearRegression``), is silent.
        Any failure is silent too -- an advisory must never break a call.
        """
        if allow_random:
            return
        # This runs on EVERY call, hits included, so it must stay cheap once it
        # has had its say. `_warn_once` would dedupe the emission but not the
        # `get_params()` that precedes it, and sklearn's `get_params` walks the
        # signature -- a per-hit cost on exactly the functions people cache to
        # avoid paying for a fit. Check the same key first and leave.
        if (CashRandomnessWarning, func_name, "_estimator_result") in self._warning_keys_seen:
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

        self._warn_once(
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
