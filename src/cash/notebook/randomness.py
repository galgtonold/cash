from __future__ import annotations

"""Detection of unseeded random calls that compromise cache reproducibility."""

import ast
import secrets
import hashlib
import inspect
import logging
import types
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..exceptions import CashWarning

__all__ = ["CashRandomnessWarning", "RandomnessCallInfo", "RANDOM_FUNCTIONS", "SEED_FUNCTIONS", "MODULE_ALIASES", "RNG_CARRIER_CONSTRUCTORS", "RandomnessVisitor", "RandomnessDetector", "check_and_warn_randomness", "describe_random_call", "format_stale_randomness_message", "warn_stale_randomness", "format_unseeded_estimator_fit_message", "format_stale_estimator_fit_message", "warn_unseeded_estimator_fit", "warn_stale_estimator_fit", "capture_rng_state", "restore_rng_state", "capture_object_rng_states", "restore_object_rng_states", "get_used_rng_modules", "get_drawing_rng_modules", "get_seeding_rng_modules", "seed_cells_not_yet_run", "rng_modules_changed", "rng_lineage_fingerprint", "rng_virtual_var", "hidden_lineage_reads", "hidden_lineage_writes", "hidden_write_lineage"]

logger = logging.getLogger(__name__)

class CashRandomnessWarning(CashWarning):
    """Warning issued when unseeded randomness is detected in cached code.

    Part of the :class:`~cash.exceptions.CashWarning` family, so the documented
    blanket filter works on it::

        warnings.filterwarnings("ignore", category=cash.CashWarning)

    It sat outside that family while it was unreachable (nothing had yet wired up the
    only code that raises it). ``CashWarning`` is itself a ``UserWarning``, so
    filters written against ``UserWarning`` keep working.
    """

@dataclass
class RandomnessCallInfo:
    """Information about a detected randomness call.

    ``carrier`` names the *variable* a draw was taken from when the call came
    off an RNG object rather than a module global — ``rng.standard_normal()``
    sets ``carrier='rng'``, ``np.random.standard_normal()`` leaves it ``None``.
    The distinction is load-bearing, not cosmetic: a carrier's reproducibility
    is decided by its own constructor, so a carrier call must never be filtered
    through the module-level seed ledger (:meth:`RandomnessDetector.is_seeded`).
    See ``_resolve_carrier_calls``.
    """
    module: str
    function: str
    lineno: int
    col_offset: int
    carrier: str | None = None

# Comprehensive mapping of randomness functions by module
RANDOM_FUNCTIONS = {
    # Python stdlib random
    'random': {
        'random', 'randint', 'randrange', 'choice', 'choices',
        'shuffle', 'sample', 'uniform', 'triangular',
        'betavariate', 'expovariate', 'gammavariate', 'gauss',
        'lognormvariate', 'normalvariate', 'vonmisesvariate',
        'paretovariate', 'weibullvariate', 'getrandbits', 'randbytes'
    },
    # NumPy random
    'numpy.random': {
        'rand', 'randn', 'randint', 'random_integers', 'random_sample',
        'random', 'ranf', 'sample', 'choice', 'shuffle', 'permutation',
        'beta', 'binomial', 'chisquare', 'dirichlet', 'exponential',
        'f', 'gamma', 'geometric', 'gumbel', 'hypergeometric',
        'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial',
        'multivariate_normal', 'negative_binomial', 'noncentral_chisquare',
        'noncentral_f', 'normal', 'pareto', 'poisson', 'power',
        'rayleigh', 'standard_cauchy', 'standard_exponential',
        'standard_gamma', 'standard_normal', 'standard_t', 'triangular',
        'uniform', 'vonmises', 'wald', 'weibull', 'zipf',
        # Generator methods
        'integers', 'bytes'
    },
    # PyTorch
    'torch': {
        'rand', 'randn', 'randint', 'randperm',
        'rand_like', 'randn_like', 'randint_like',
        'bernoulli', 'multinomial', 'normal', 'poisson'
    },
    # TensorFlow
    'tensorflow.random': {
        'uniform', 'normal', 'truncated_normal', 'shuffle',
        'categorical', 'gamma', 'poisson', 'stateless_uniform',
        'stateless_normal', 'stateless_truncated_normal'
    },
    # Shortened aliases (tf.random)
    'tf.random': {
        'uniform', 'normal', 'truncated_normal', 'shuffle',
        'categorical', 'gamma', 'poisson'
    },
}

# Seed functions by module
SEED_FUNCTIONS = {
    'random': {'seed'},
    'numpy.random': {'seed'},
    'numpy': {'random.seed'},  # np.random.seed()
    'torch': {'manual_seed', 'cuda.manual_seed', 'cuda.manual_seed_all'},
    'tensorflow.random': {'set_seed'},
    'tensorflow': {'random.set_seed'},
    'tf.random': {'set_seed'},
    'tf': {'random.set_seed'},
}

# Module aliases commonly used
MODULE_ALIASES = {
    'np': 'numpy',
    'tf': 'tensorflow',
}

# -----------------------------------------------------------------------------
# RNG carriers (objects that hold their own bit-generator state)
# -----------------------------------------------------------------------------
#
# The three kinds below are the single taxonomy of "what is an RNG object" for
# this module.  They are consumed by TWO channels that see different things:
#
# * ``_classify_rng_carrier`` (further down) classifies a **live value**
#   by ``isinstance``.  It runs at store/restore time, when the object exists.
# * The tables here classify a **constructor call in source**.  Detection has to
#   run on the AST because the warning fires *before* the statement executes —
#   and, decisively, because the RNG is often a function local
#   (``def f(): g = np.random.default_rng(); ...``) that never reaches
#   ``user_ns`` at all, so no live-value classifier can ever see it.
#
# The AST channel is also the only one that can answer the question the warning
# actually turns on — *was it seeded?*  ``default_rng()`` and ``default_rng(42)``
# produce indistinguishable ``Generator`` objects; only the source says which
# one you wrote.  So the live classifier deliberately does not drive the
# warning: it would have to guess, and guessing "unseeded" would fire on
# perfectly reproducible generators.  Sharing the kind constants is what keeps
# the two channels from drifting into two different notions of an RNG.

_KIND_NP_GENERATOR = 'numpy.Generator'
_KIND_NP_RANDOMSTATE = 'numpy.RandomState'
_KIND_PY_RANDOM = 'random.Random'

# Constructors that MINT a carrier, keyed by the kind they produce.
RNG_CARRIER_CONSTRUCTORS = {
    'numpy.random.default_rng': _KIND_NP_GENERATOR,
    'numpy.random.Generator': _KIND_NP_GENERATOR,
    'numpy.random.RandomState': _KIND_NP_RANDOMSTATE,
    'random.Random': _KIND_PY_RANDOM,
    'random.SystemRandom': _KIND_PY_RANDOM,
}

# Bare constructor names, for ``from numpy.random import default_rng`` when the
# import happened in an *earlier* statement (each statement is scanned on its
# own, so this visitor never sees that import).  Same pragmatic role as
# MODULE_ALIASES' 'np' -> 'numpy'.
#
# Only ``default_rng`` earns a spot: it is unambiguous in practice.  ``Random``,
# ``Generator`` and ``RandomState`` are NOT here — bare ``Generator(...)`` is far
# more likely to be ``typing.Generator`` or a user's own class, and mistaking one
# for an RNG would warn about code that has no randomness in it at all.
_BARE_CARRIER_CONSTRUCTORS = {
    'default_rng': _KIND_NP_GENERATOR,
}

# numpy bit generators: ``Generator(PCG64(42))`` is seeded, ``Generator(PCG64())``
# is not, so the seed question recurses one level into the bit generator.
_NP_BIT_GENERATORS = frozenset({
    'PCG64', 'PCG64DXSM', 'MT19937', 'Philox', 'SFC64',
})

# Which draw methods each carrier kind offers.  Reusing RANDOM_FUNCTIONS keeps
# one list of "what counts as a draw" per module rather than two.
_CARRIER_DRAW_FUNCTIONS = {
    _KIND_NP_GENERATOR: RANDOM_FUNCTIONS['numpy.random'],
    _KIND_NP_RANDOMSTATE: RANDOM_FUNCTIONS['numpy.random'],
    _KIND_PY_RANDOM: RANDOM_FUNCTIONS['random'],
}

# The module each carrier kind is attributed to in the warning message.
_CARRIER_MODULES = {
    _KIND_NP_GENERATOR: 'numpy.random',
    _KIND_NP_RANDOMSTATE: 'numpy.random',
    _KIND_PY_RANDOM: 'random',
}

# Any name that could be a draw off some carrier.  Used as a cheap pre-filter
# while visiting; the kind-specific set above makes the real decision once the
# carrier's kind is known.
_ANY_CARRIER_DRAW_FUNCTION = frozenset(
    RANDOM_FUNCTIONS['numpy.random'] | RANDOM_FUNCTIONS['random']
)


def _rng_constructor_is_seeded(node: ast.Call) -> bool:
    """Whether an RNG constructor call was handed a reproducible seed.

    ``default_rng()`` / ``default_rng(None)`` draw from OS entropy and are not
    reproducible.  Anything else — a literal, a variable, a bit generator built
    with a seed — is treated as seeded.

    Deliberately biased toward "seeded" for shapes it cannot read (e.g.
    ``default_rng(cfg.seed)``).  A false negative costs a missing warning on an
    unusual spelling; a false positive would fire on reproducible code, which is
    how a warning gets filtered out wholesale and stops protecting anyone.
    """
    args = list(node.args)
    args.extend(kw.value for kw in node.keywords if kw.arg in ('seed', 'bit_generator'))

    if not args:
        return False

    first = args[0]
    if isinstance(first, ast.Constant) and first.value is None:
        return False
    if isinstance(first, ast.Call):
        if isinstance(first.func, ast.Attribute):
            inner_name = first.func.attr
        else:
            inner_name = getattr(first.func, 'id', None)
        if inner_name in _NP_BIT_GENERATORS:
            return _rng_constructor_is_seeded(first)
    return True

class RandomnessVisitor(ast.NodeVisitor):
    """AST visitor that detects randomness and seed calls."""

    def __init__(self):
        self.random_calls: list[RandomnessCallInfo] = []
        self.seed_calls: list[tuple[str, int]] = []  # (module, lineno)
        # Seeds that RE-SEED FROM ENTROPY -- ``seed()`` / ``seed(None)``.
        # The opposite of a reproducibility request: the user is asking for a
        # different stream on every run, so the values below such a call must
        # not be replayed from cache.
        self.entropy_seed_calls: list[tuple[str, int]] = []  # (module, lineno)
        self.imports: dict = {}  # name -> module mapping
        # RNG carriers bound in THIS source: name -> (kind, seeded).
        self.carrier_assigns: dict[str, tuple[str, bool]] = {}
        # Names rebound to a non-carrier here, so a session-level binding for
        # them is stale and must be dropped.
        self.carrier_clears: set[str] = set()
        # ``g = rng`` where ``rng`` is not bound in THIS source: (target, source),
        # resolved by the detector against session state.
        self.carrier_aliases: list[tuple[str, str]] = []
        # Candidate draws off an object: (base, func, lineno, col_offset).
        # Collected unresolved — whether ``base`` is a carrier at all can depend
        # on a previous statement, which this visitor cannot see.  The detector
        # resolves them against session state.
        self.carrier_calls: list[tuple[str, str, int, int]] = []
        # Lexical scope of the ``def``/``async def`` currently being visited:
        # a stack of ``(func_name, {param names})``.  A draw off a name that is a
        # parameter of an enclosing function must NOT resolve against the flat,
        # session-wide carrier ledger (a same-named unseeded global from an
        # earlier cell is a different variable) — see ``_check_carrier_call``.
        self._function_stack: list[tuple[str, set[str]]] = []
        # Draws taken off a parameter, per enclosing function:
        # func_name -> [(param, draw_fn, lineno, col_offset)].  Whether such a
        # draw is unseeded depends on the ARGUMENT passed at a call site, which
        # this visitor cannot see; the detector settles it against the call-arg
        # candidates below.
        self.param_draws: dict[str, list[tuple[str, str, int, int]]] = {}
        # Arguments handing a candidate RNG to a bare-name callee:
        # (callee, param_ref, arg_spec, lineno, col_offset), where ``arg_spec`` is
        # ``('inline', kind, seeded)`` for an inline constructor or
        # ``('alias', name)`` for a bare-name argument the detector resolves
        # against session state.  ``param_ref`` is the parameter NAME for a
        # keyword argument (the source states it) and ``('pos', index)`` for a
        # positional one — naming that slot needs the callee's signature, which
        # is session state, so the detector maps it at resolve time.
        self.call_arg_candidates: list[tuple[str, str | tuple[str, int], tuple, int, int]] = []

    def visit_Import(self, node: ast.Import):
        """Track imports like 'import random', 'import numpy as np'."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track 'from X import Y' statements."""
        if node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                self.imports[name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Give the function body a lexical scope for its parameter names."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """``async def`` scopes its parameters exactly like ``def``."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        params: set[str] = set()
        self._collect_param_names(node.args, params)
        self._function_stack.append((node.name, params))
        # generic_visit still walks decorators, defaults and the body, so every
        # existing channel (carrier draws, module draws, seed calls) keeps
        # firing inside the body; the stack only changes how a bare parameter
        # draw is routed in ``_check_carrier_call``.
        self.generic_visit(node)
        self._function_stack.pop()

    @staticmethod
    def _collect_param_names(args: ast.arguments, into: set[str]) -> None:
        """Register every parameter name (mirrors analysis._add_args_to_scope)."""
        for arg in args.args:
            into.add(arg.arg)
        for arg in args.posonlyargs:
            into.add(arg.arg)
        for arg in args.kwonlyargs:
            into.add(arg.arg)
        if args.vararg:
            into.add(args.vararg.arg)
        if args.kwarg:
            into.add(args.kwarg.arg)

    def _resolve_module(self, name: str) -> str | None:
        """Resolve a name to its module, handling aliases."""
        if name in self.imports:
            return self.imports[name]
        # Check for known aliases
        if name in MODULE_ALIASES:
            return MODULE_ALIASES[name]
        return None

    def _get_call_chain(self, node: ast.expr) -> list[str]:
        """Extract the full call chain from a node, e.g., ['np', 'random', 'seed']."""
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            chain.append(node.id)
        return list(reversed(chain))

    def visit_Assign(self, node: ast.Assign):
        """Track ``rng = np.random.default_rng()`` and friends."""
        self._track_carrier_binding(
            [t.id for t in node.targets if isinstance(t, ast.Name)], node.value,
        )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Track the annotated spelling — ``rng: Generator = default_rng()``."""
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._track_carrier_binding([node.target.id], node.value)
        self.generic_visit(node)

    def _track_carrier_binding(self, names: list[str], value: ast.expr):
        """Bind, re-bind, or clear ``names`` as RNG carriers based on *value*."""
        if not names:
            return

        kind = self._carrier_kind_of(value)
        if kind is not None:
            seeded = _rng_constructor_is_seeded(value)
            for name in names:
                self.carrier_assigns[name] = (kind, seeded)
                self.carrier_clears.discard(name)
            return

        # Aliasing: ``g = rng`` carries the binding across, seededness included.
        if isinstance(value, ast.Name):
            if value.id in self.carrier_assigns:
                for name in names:
                    self.carrier_assigns[name] = self.carrier_assigns[value.id]
                    self.carrier_clears.discard(name)
            else:
                # The source may be a carrier bound in an EARLIER statement,
                # which this visitor cannot see. Emit the alias unresolved and
                # let the detector settle it against session state.
                for name in names:
                    self.carrier_assigns.pop(name, None)
                    self.carrier_aliases.append((name, value.id))
            return

        # Rebound to something that isn't a carrier: forget it, and tell the
        # detector to forget any session-level binding too. Without this a name
        # reused for an ordinary value ("rng = load_config()") would keep
        # warning on every method call that shares a name with a draw.
        for name in names:
            self.carrier_assigns.pop(name, None)
            self.carrier_clears.add(name)

    def _carrier_kind_of(self, value: ast.expr) -> str | None:
        """Return the carrier kind *value* constructs, or None."""
        if not isinstance(value, ast.Call):
            return None
        chain = self._get_call_chain(value.func)
        if not chain:
            return None

        if len(chain) == 1:
            # ``from numpy.random import default_rng; rng = default_rng()``
            imported = self.imports.get(chain[0])
            if imported:
                return RNG_CARRIER_CONSTRUCTORS.get(imported)
            return _BARE_CARRIER_CONSTRUCTORS.get(chain[0])

        resolved_base = self._resolve_module(chain[0]) or chain[0]
        full_name = '.'.join([*resolved_base.split('.'), *chain[1:]])
        return RNG_CARRIER_CONSTRUCTORS.get(full_name)

    def visit_Call(self, node: ast.Call):
        """Check for randomness and seed function calls."""
        chain = self._get_call_chain(node.func)

        if not chain:
            self.generic_visit(node)
            return

        # Build the full dotted name
        full_name = '.'.join(chain)

        # Check various patterns
        self._check_random_call(chain, full_name, node)
        self._check_seed_call(chain, full_name, node)
        self._check_carrier_call(chain, node)
        self._check_call_args(chain, node)

        self.generic_visit(node)

    def _check_call_args(self, chain: list[str], node: ast.Call):
        """Record an argument that hands a candidate RNG to a bare-name callee.

        Scoped to bare-name calls (``price(...)``); ``obj.method(...)`` stays out
        of scope, consistent with the ``self.rng.normal()`` and ``df.sample()``
        gaps this module already carries.

        BOTH argument forms are recorded, differing only in how the parameter
        they bind to is named:

        * a KEYWORD argument states its parameter in the source, so it is
          recorded under that name;
        * a POSITIONAL argument knows only its INDEX here, because naming the
          slot requires the callee's signature — a session fact this visitor
          cannot see.  It is recorded as ``('pos', index)`` and mapped to a name
          at resolve time (:meth:`RandomnessDetector._param_name_for_position`).

        Deferring the index rather than resolving it here is what preserves
        ``_ScanResult``'s purity contract: the scan stays a pure function of the
        source, so it remains memoizable for the life of the session.

        Also left unresolved, for the same reason: whether ``price`` actually
        draws off that parameter, and whether a bare-name argument is a seeded
        carrier.
        """
        if len(chain) != 1:
            return  # a bare-name callee only; ``obj.method(...)`` is out of scope
        callee = chain[0]

        for index, value in enumerate(node.args):
            if isinstance(value, ast.Starred):
                # ``price(*args, rng)``: every slot after an unpacking depends on
                # the runtime length of ``args``, so no later index can be named.
                # Stop rather than mis-attribute one parameter's risk to another.
                break
            self._record_call_arg(callee, ('pos', index), value, node)

        for kw in node.keywords:
            if kw.arg is None:
                continue  # ``**kwargs`` — no parameter name to bind to
            self._record_call_arg(callee, kw.arg, kw.value, node)

    def _record_call_arg(
        self, callee: str, param_ref: str | tuple[str, int],
        value: ast.expr, node: ast.Call,
    ):
        """Record *value* as a candidate RNG argument bound to *param_ref*.

        The classification (inline constructor vs bare-name alias) is shared by
        the keyword and positional paths on purpose: the two forms differ only in
        how the parameter is addressed, never in what counts as an RNG argument.
        Anything else — a literal, an attribute, a call that is not an RNG
        constructor — is not recorded at all, so it can never warn.
        """
        kind = self._carrier_kind_of(value)
        if kind is not None:
            seeded = _rng_constructor_is_seeded(value)
            self.call_arg_candidates.append(
                (callee, param_ref, ('inline', kind, seeded),
                 node.lineno, node.col_offset)
            )
        elif isinstance(value, ast.Name):
            self.call_arg_candidates.append(
                (callee, param_ref, ('alias', value.id),
                 node.lineno, node.col_offset)
            )

    def _check_carrier_call(self, chain: list[str], node: ast.Call):
        """Collect ``<name>.<draw>()`` as a *candidate* carrier draw.

        Deliberately unresolved and deliberately permissive: ``df.sample()``
        lands here too, because ``sample`` is a draw name on both RNG modules.
        That is harmless — the detector only emits a warning once ``base`` is a
        *known* carrier, so an unknown receiver stays silent. It also keeps the
        documented ``df.sample()`` gap exactly where it is rather than turning
        it into a false positive.
        """
        if len(chain) != 2:
            return  # ``np.random.rand()`` (3) is the module path's job;
                    # ``self.rng.normal()`` (3) is a known gap.
        base, func_name = chain
        if func_name not in _ANY_CARRIER_DRAW_FUNCTION:
            return
        if base in MODULE_ALIASES or self._resolve_module(base) is not None:
            return  # an imported module, not an object

        # A draw off a *parameter* of an enclosing function is not a draw off a
        # session global that happens to share the name.  Route it to
        # ``param_draws`` (settled interprocedurally at the call site) and do NOT
        # add it to ``carrier_calls``, so it never resolves against the flat
        # session ledger.  A parameter locally rebound to a concrete carrier
        # (``base in self.carrier_assigns``) keeps the existing flat resolution —
        # that binding is real and this-source, not a stale global.
        if base not in self.carrier_assigns:
            for owner, params in reversed(self._function_stack):
                if base in params:
                    self.param_draws.setdefault(owner, []).append(
                        (base, func_name, node.lineno, node.col_offset)
                    )
                    return

        self.carrier_calls.append((base, func_name, node.lineno, node.col_offset))

    def _check_random_call(self, chain: list[str], full_name: str, node: ast.Call):
        """Check if this call is a randomness function."""
        func_name = chain[-1]

        # 1. Resolve base to full module path
        if len(chain) > 1:
            base = chain[0]
            # Resolve alias (e.g. 'np' -> 'numpy', 'nr' -> 'numpy.random')
            resolved_base = self._resolve_module(base) or base

            # Reconstruct full path with resolved base
            # e.g. np.random.rand -> numpy.random.rand
            # e.g. nr.rand -> numpy.random.rand

            # Start with resolved base
            full_path_parts = resolved_base.split('.')
            # Add remaining parts of the chain (excluding the base we just resolved)
            full_path_parts.extend(chain[1:])

            resolved_full_name = '.'.join(full_path_parts)

            # Check against known randomness functions
            for module, funcs in RANDOM_FUNCTIONS.items():
                # Check for exact match: module.func
                if resolved_full_name == f"{module}.{func_name}" and func_name in funcs:
                    self.random_calls.append(RandomnessCallInfo(
                        module=module,
                        function=func_name,
                        lineno=node.lineno,
                        col_offset=node.col_offset
                    ))
                    return

                # Check for parent module match: module.submodule.func
                if resolved_full_name.startswith(module + '.') and func_name in funcs:
                   self.random_calls.append(RandomnessCallInfo(
                       module=module,
                       function=func_name,
                       lineno=node.lineno,
                       col_offset=node.col_offset
                   ))
                   return

        # 2. Handle direct function imports (chain length 1)
        # e.g. from random import random; random()
        else:
            if func_name in self.imports:
                imported_from = self.imports[func_name]
                for module, funcs in RANDOM_FUNCTIONS.items():
                    if imported_from == f"{module}.{func_name}" or \
                       (imported_from.startswith(module + '.') and func_name in funcs):
                        self.random_calls.append(RandomnessCallInfo(
                            module=module,
                            function=func_name,
                            lineno=node.lineno,
                            col_offset=node.col_offset
                        ))
                        return

    def _check_seed_call(self, chain: list[str], full_name: str, node: ast.Call):
        """Check if this call is a seed function."""
        func_name = chain[-1]

        # 1. Resolve base
        if len(chain) > 1:
            base = chain[0]
            resolved_base = self._resolve_module(base) or base

            full_path_parts = resolved_base.split('.')
            full_path_parts.extend(chain[1:])
            resolved_full_name = '.'.join(full_path_parts)

            for module, seed_funcs in SEED_FUNCTIONS.items():
                # Exact match: module.seed
                if resolved_full_name == f"{module}.{func_name}" and func_name in seed_funcs:
                    self.seed_calls.append((module, node.lineno))
                    if _is_entropy_seed(node):
                        self.entropy_seed_calls.append((module, node.lineno))
                    return

                # Check if it matches a known full path seed like 'numpy.random.seed'
                # The SEED_FUNCTIONS keys are modules, values are function names or 'submod.func'
                for seed_func_signature in seed_funcs:
                     if resolved_full_name == f"{module}.{seed_func_signature}":
                         self.seed_calls.append((module, node.lineno))
                         if _is_entropy_seed(node):
                             self.entropy_seed_calls.append((module, node.lineno))
                         return

        # 2. Handle direct function imports
        else:
             if func_name in self.imports:
                imported_from = self.imports[func_name]
                for module, _seed_funcs in SEED_FUNCTIONS.items():
                    if imported_from == f"{module}.{func_name}":
                        self.seed_calls.append((module, node.lineno))
                        if _is_entropy_seed(node):
                            self.entropy_seed_calls.append((module, node.lineno))
                        return

class _ScanResult(NamedTuple):
    """One statement's purely-source-derived randomness facts.

    Every field is immutable and independent of session state — that is the
    contract that lets ``RandomnessDetector._scan`` memoize this by source
    string and reuse it for the life of the session.
    """
    random_calls: tuple[RandomnessCallInfo, ...]
    seed_calls: tuple[tuple[str, int], ...]
    carrier_assigns: tuple[tuple[str, tuple[str, bool]], ...]
    carrier_clears: frozenset[str]
    carrier_calls: tuple[tuple[str, str, int, int], ...]
    carrier_aliases: tuple[tuple[str, str], ...]
    # func_name -> ((param, draw_fn, lineno, col), ...): draws off a parameter.
    param_draws: tuple[tuple[str, tuple[tuple[str, str, int, int], ...]], ...]
    # (callee, param_ref, arg_spec, lineno, col): RNG arguments at a call site.
    # ``param_ref`` is a parameter name (keyword arg) or ``('pos', index)``
    # (positional arg) — see ``RandomnessVisitor._check_call_args``.
    call_arg_candidates: tuple[tuple[str, str | tuple[str, int], tuple, int, int], ...]


class RandomnessDetector:
    """
    Detects randomness function calls without proper seed setting.

    Two channels feed it, because there are two ways to be random:

    * **Module globals** — ``np.random.rand()``, ``random.random()``.
      Reproducibility is a property of the *module*, tracked in
      ``seeded_modules``.
    * **Carriers** — ``rng = np.random.default_rng(); rng.standard_normal()``.
      Reproducibility is a property of the *object*, decided by its
      constructor and tracked per variable in ``rng_carriers``.

    Both are session-scoped: seeding (or minting a generator) in one cell
    applies to the cells after it.
    """

    def __init__(self, shell: object | None = None):
        """
        Args:
            shell: The live IPython shell, used for the ONE question the source
                cannot answer: which parameter a POSITIONAL argument binds to
                (see :meth:`_param_name_for_position`).  Optional, and resolved
                lazily via ``IPython.get_ipython()`` when not supplied, so a
                detector constructed before/outside a kernel still works — it
                just stays silent about positional arguments, which is the
                module's standing conservative bias anyway.
        """
        self.shell = shell
        # Modules that have been seeded in the current session
        self.seeded_modules: set[str] = set()
        # RNG carriers bound in this session: var name -> (kind, seeded).
        # The object-API analogue of ``seeded_modules``, and session-scoped for
        # the same reason: ``rng = np.random.default_rng()`` and the draw off it
        # are separate statements, each scanned on its own.
        self.rng_carriers: dict[str, tuple[str, bool]] = {}
        # Functions that draw off one of their parameters, discovered as their
        # ``def`` is scanned: func_name -> {param -> (draw_fn, lineno, col)}.
        # Session-scoped like ``rng_carriers`` because the ``def`` and the call
        # that supplies the unseeded argument are separate statements.  This is
        # the registry the call-site channel resolves an argument against — the
        # only interprocedural link in the module, and the reason a same-named
        # global cannot leak into a function body (that resolution never happens
        # for a parameter draw; see ``_check_carrier_call``).
        self.risky_params: dict[str, dict[str, tuple[str, int, int]]] = {}
        # Memo: code string -> the pure AST scan (see ``_ScanResult``).
        # The scan is the expensive half of analyze_code (parse + full visit) and
        # is a pure function of the source, so it is safe to reuse across runs.
        # Only the cheap session-state filters below (``is_seeded``,
        # ``_resolve_carrier_calls``) depend on session state — carrier draws are
        # therefore memoized UNRESOLVED, since whether ``rng`` names a carrier is
        # session state, not a property of the source.
        # Mirrors ``Cash._capture_use_cache`` in core.py: unbounded growth is the
        # real risk in a long notebook session, so the insert is size-gated.
        self._scan_cache: dict[str, _ScanResult] = {}
        # Dedupe ledger: (code, warning message) pairs already warned about in
        # this session.  See ``mark_warned``.
        self._warned: set[tuple[str, str]] = set()

    def reset(self):
        """Reset seeding state (e.g., for new session)."""
        self.seeded_modules.clear()
        self.rng_carriers.clear()
        self.risky_params.clear()
        # A new session re-warns: the user is looking at a fresh set of outputs.
        # ``_scan_cache`` is deliberately kept — it is pure w.r.t. session state.
        self._warned.clear()

    def mark_warned(self, code: str, message: str) -> bool:
        """Record ``(code, message)`` as warned; return True if it was new.

        The dedupe unit is *one warning per statement per session*: a statement's
        source text plus the specific call being flagged.  Editing the statement
        changes ``code`` and re-warns, which is what the user wants — the warning
        is about the code they are looking at.  Re-running an unchanged statement
        stays silent.
        """
        key = (code, message)
        if key in self._warned:
            return False
        if len(self._warned) < 4096:
            self._warned.add(key)
        return True

    def _scan(self, code: str) -> '_ScanResult':
        """Parse + visit *code*, memoized per source string.

        Everything returned is an immutable, purely-source-derived fact, so the
        memo entry stays valid for the life of the session and callers cannot
        corrupt it.
        """
        cached = self._scan_cache.get(code)
        if cached is not None:
            return cached

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result = _ScanResult((), (), (), frozenset(), (), (), (), ())
        else:
            visitor = RandomnessVisitor()
            visitor.visit(tree)
            result = _ScanResult(
                tuple(visitor.random_calls),
                tuple(visitor.seed_calls),
                tuple(visitor.carrier_assigns.items()),
                frozenset(visitor.carrier_clears),
                tuple(visitor.carrier_calls),
                tuple(visitor.carrier_aliases),
                tuple((fn, tuple(draws)) for fn, draws in visitor.param_draws.items()),
                tuple(visitor.call_arg_candidates),
            )

        if len(self._scan_cache) < 4096:
            self._scan_cache[code] = result
        return result

    def bind_carriers(self, assigns, clears, aliases=()) -> None:
        """Fold a statement's carrier bindings into session state."""
        for name in clears:
            self.rng_carriers.pop(name, None)
        for name, entry in assigns:
            self._bind_carrier(name, entry)
        # Resolved last: an alias' source may have been (re)bound just above by
        # this same statement.
        for target, source in aliases:
            entry = self.rng_carriers.get(source)
            if entry is None:
                self.rng_carriers.pop(target, None)
            else:
                self._bind_carrier(target, entry)

    def _bind_carrier(self, name: str, entry: tuple[str, bool]) -> None:
        # Size-gated like the other session ledgers, but a rebind of a name
        # already tracked must always land or a re-seed would be ignored.
        if len(self.rng_carriers) < 4096 or name in self.rng_carriers:
            self.rng_carriers[name] = entry

    def _resolve_carrier_calls(self, carrier_calls) -> list[RandomnessCallInfo]:
        """Turn candidate object draws into confirmed unseeded-randomness calls.

        Silent unless the receiver is a *known* carrier: an unresolvable
        ``obj.sample()`` is far more likely to be a DataFrame than an RNG, and a
        warning that cries wolf gets filtered out wholesale.

        Carrier draws bypass :meth:`is_seeded` on purpose. ``np.random.seed(42)``
        seeds the legacy global singleton; a ``default_rng()`` Generator is
        wholly independent of it. Consulting the module ledger here would let an
        unrelated seed call two cells up silence the one warning that matters.
        """
        found: list[RandomnessCallInfo] = []
        for base, func_name, lineno, col_offset in carrier_calls:
            entry = self.rng_carriers.get(base)
            if entry is None:
                continue
            kind, seeded = entry
            if seeded:
                continue
            if func_name not in _CARRIER_DRAW_FUNCTIONS.get(kind, frozenset()):
                continue
            found.append(RandomnessCallInfo(
                module=_CARRIER_MODULES.get(kind, kind),
                function=func_name,
                lineno=lineno,
                col_offset=col_offset,
                carrier=base,
            ))
        return found

    def register_param_draws(self, param_draws) -> None:
        """Fold a ``def``'s parameter draws into the session ``risky_params``.

        Called as the ``def`` statement is analyzed, before any call to it is
        resolved, so a single statement that both defines and calls a function
        settles against itself.  Replaces the entry for a re-scanned function
        (first draw per parameter wins), size-gated like the other ledgers.
        """
        for func_name, draws in param_draws:
            if len(self.risky_params) >= 4096 and func_name not in self.risky_params:
                continue
            entry: dict[str, tuple[str, int, int]] = {}
            for param, draw_fn, lineno, col_offset in draws:
                entry.setdefault(param, (draw_fn, lineno, col_offset))
            self.risky_params[func_name] = entry

    def _live_callee(self, callee: str) -> object | None:
        """The live object bound to *callee*, or None if it cannot be reached.

        Defensive to the point of paranoia by design: this runs on every call
        site in every statement, and its only job is to feed an ADVISORY warning.
        Any namespace that will not answer (no shell, no ``user_ns``, a mapping
        that raises) is treated as "unknown" — never as an error.
        """
        shell = self.shell
        if shell is None:
            try:
                from IPython import get_ipython
                shell = get_ipython()
            except ImportError:  # IPython is an optional dependency
                return None
        try:
            return shell.user_ns[callee]
        except (AttributeError, TypeError, KeyError):
            return None

    def _param_name_for_position(self, callee: str, index: int) -> str | None:
        """Name the parameter a positional argument at *index* binds to.

        The source alone cannot answer this — ``price(rng, 100)`` says only
        "slot 0" — so the callee's RUNTIME signature is consulted. That is the
        same object the call is about to invoke, which is what makes the mapping
        trustworthy rather than a guess.

        Returns None (-> the caller stays silent) for every shape that is not an
        unambiguous positional slot on a plain function:

        * the callee is not in the namespace, or the namespace is unreachable;
        * it is not a plain function — a class, a builtin, a ``functools.partial``
          or a bound method binds ``self``/arguments in ways this mapping would
          have to guess at;
        * its signature cannot be introspected at all;
        * *index* is not a fixed positional slot: it lands in ``*args``, or past
          the end of the signature, or in keyword-only territory.

        Decorated functions come out right for free: ``inspect.signature``
        follows ``__wrapped__``, so a ``functools.wraps`` wrapper reports the
        wrapped signature, and a wrapper that discards it reports
        ``(*args, **kwargs)`` and lands in the ``*args`` silence above.
        """
        func = self._live_callee(callee)
        if not isinstance(func, types.FunctionType):
            return None

        try:
            parameters = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):  # unintrospectable signature
            return None

        positional: list[str] = []
        for param in parameters:
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                positional.append(param.name)
            else:
                # ``*args`` swallows this index and every later one; keyword-only
                # and ``**kwargs`` end the positional run. Either way there are no
                # further nameable slots.
                break

        if index < len(positional):
            return positional[index]
        return None

    def _param_name(self, callee: str, param_ref: str | tuple[str, int]) -> str | None:
        """Resolve a candidate's ``param_ref`` to a callee parameter name."""
        if isinstance(param_ref, tuple):  # ('pos', index)
            return self._param_name_for_position(callee, param_ref[1])
        return param_ref  # a keyword argument named its parameter in the source

    def _resolve_call_arg_carriers(self, candidates) -> list[RandomnessCallInfo]:
        """Turn ``callee(<unseeded rng>)`` / ``callee(param=<unseeded rng>)`` into
        a confirmed draw.

        The interprocedural half of the detector: an unseeded generator passed
        to a function that draws off that parameter is exactly as unreproducible
        as the inline draw, but the draw itself lives in the callee's body, one
        statement away.  ``risky_params`` (built as the ``def`` was scanned)
        supplies the link; this resolves it at the CALL SITE, where the argument
        is written and where the warning belongs.

        Keyword and positional arguments differ only in how the parameter is
        addressed — ``_param_name`` normalises that away, and the risk lookup
        below is then literally the same code for both.  A positional argument
        whose slot cannot be named unambiguously resolves to None and is dropped
        here, so the gap it leaves is a missing warning, never a wrong one.

        Silent unless the callee was scanned this session AND actually draws off
        that parameter — an out-of-order or imported ``def`` stays quiet, matching
        the module's standing false-positive-avoidance bias.
        """
        found: list[RandomnessCallInfo] = []
        for callee, param_ref, arg_spec, lineno, col_offset in candidates:
            params = self.risky_params.get(callee)
            if not params:
                continue
            param = self._param_name(callee, param_ref)
            if param is None:
                continue  # unnameable slot — stay silent (conservative)
            draw = params.get(param)
            if draw is None:
                continue

            if arg_spec[0] == 'inline':
                _tag, kind, seeded = arg_spec
            else:  # ('alias', name): resolve the argument against session state
                _tag, name = arg_spec
                entry = self.rng_carriers.get(name)
                if entry is None:
                    continue  # not a known carrier — stay silent (conservative)
                kind, seeded = entry
            if seeded:
                continue

            draw_fn, _dl, _dc = draw
            found.append(RandomnessCallInfo(
                module=_CARRIER_MODULES.get(kind, kind),
                function=draw_fn,
                lineno=lineno,
                col_offset=col_offset,
                carrier=param,
            ))
        return found

    def mark_seeded(self, module: str):
        self.seeded_modules.add(module)
        # Also mark parent modules
        if '.' in module:
            parent = module.rsplit('.', 1)[0]
            self.seeded_modules.add(parent)

    def _get_parent_module(self, module: str) -> str | None:
        if '.' in module:
            return module.rsplit('.', 1)[0]
        return None

    def is_seeded(self, module: str) -> bool:
        """Check if a module has been seeded."""
        if module in self.seeded_modules:
            return True
        # Check parent module (e.g., numpy.random seeded by numpy seed)
        parent = self._get_parent_module(module)
        return bool(parent and parent in self.seeded_modules)

    def analyze_code(self, code: str) -> tuple[list[RandomnessCallInfo], list[str], bool]:
        """
        Analyze code for unseeded randomness.

        This method also updates internal seeding state based on detected
        seed calls in the code.

        Args:
            code: Python source code to analyze

        Returns:
            Tuple of (list of unseeded random calls, list of warning messages, has_seed_calls)
            - has_seed_calls is True if the code contains seed function calls
              (these statements should not be cached, as the seed must be executed)
        """
        scan = self._scan(code)

        has_seed_calls = len(scan.seed_calls) > 0

        # Update seeding state from any seed calls in this code
        for module, _lineno in scan.seed_calls:
            self.mark_seeded(module)

        # Bind this statement's carriers BEFORE resolving its own draws, so a
        # single statement that both mints and draws — ``rng = default_rng();
        # x = rng.normal()``, or a whole ``def`` body — resolves against itself.
        self.bind_carriers(scan.carrier_assigns, scan.carrier_clears, scan.carrier_aliases)

        # Register any parameter draws THIS statement's defs contain before
        # resolving its own call sites, so a def-and-call in one statement links.
        self.register_param_draws(scan.param_draws)

        # Then check for unseeded random calls
        unseeded_calls = []
        warnings_list = []

        for call in scan.random_calls:
            if not self.is_seeded(call.module):
                unseeded_calls.append(call)

        unseeded_calls.extend(self._resolve_carrier_calls(scan.carrier_calls))
        unseeded_calls.extend(self._resolve_call_arg_carriers(scan.call_arg_candidates))
        # Source order, so a statement with several draws reads top-to-bottom.
        unseeded_calls.sort(key=lambda c: (c.lineno, c.col_offset))

        for call in unseeded_calls:
            if call.carrier is not None:
                warnings_list.append(
                    f"Unseeded randomness detected: {describe_random_call(call)} "
                    f"at line {call.lineno}. '{call.carrier}' came from an unseeded "
                    f"{call.module} generator, so cached results are frozen, not random. "
                    f"Pass a seed to the generator, or use @cash:allow-random to suppress."
                )
            else:
                warnings_list.append(
                    f"Unseeded randomness detected: {describe_random_call(call)} "
                    f"at line {call.lineno}. Cached results may not be reproducible. "
                    f"Consider calling seed() first or use @cash:allow-random to suppress."
                )

        return unseeded_calls, warnings_list, has_seed_calls


def describe_random_call(call: RandomnessCallInfo) -> str:
    """Render a call the way the user wrote it: ``rng.normal()``/``np.random.rand()``."""
    if call.carrier is not None:
        return f"{call.carrier}.{call.function}()"
    return f"{call.module}.{call.function}()"


def format_stale_randomness_message(call: RandomnessCallInfo) -> str:
    """The message for a cached unseeded value being replayed.

    Deliberately a *different claim* from the compute-time warning, not a
    re-wording of it. On the cold run the value is fresh and correct and the
    warning is advice about the source ("may not be reproducible"). On a restore
    the value on screen is definitively a replay — "may" would understate the
    one thing the user needs to know.

    The distinct text also earns the message its own slot in the dedupe ledger,
    which is keyed on ``(code, message)``. That is what lets the replay be
    announced without disturbing the once-per-statement-per-session
    contract for the compute-time warning.
    """
    # ASCII only: this lands in a kernel's stderr, and a Windows console
    # codepage renders an em-dash as a replacement char mid-sentence.
    return (
        f"Unseeded randomness restored from cache: {describe_random_call(call)} "
        f"at line {call.lineno}. The value you are seeing is a replay of an "
        f"earlier run, not a fresh draw - re-running will not change it. Use "
        f"@cash:no-cache to re-run it every time, seed the RNG for real "
        f"reproducibility, or @cash:allow-random to suppress."
    )


def warn_stale_randomness(
    code: str,
    unseeded_calls: 'Iterable[RandomnessCallInfo]',
    detector: RandomnessDetector,
    suppress_warning: bool = False,
) -> None:
    """Announce that a cached value derived from unseeded randomness was served.

    Called on the cache-*hit* path, after the restore has actually succeeded —
    the claim is about the value handed back, so it must not be made for a
    restore that fell through to execution.

    Args:
        code: The statement source, control-structure markers already stripped,
            so the dedupe key matches the compute-time warning's.
        unseeded_calls: The unseeded calls ``check_and_warn_randomness`` already
            found for this statement. Passed in rather than re-derived: the scan
            has run on the common path, and re-running it here would double the
            detector's seed-tracking side effects.
        detector: Supplies the shared per-statement dedupe ledger.
        suppress_warning: ``# @cash:allow-random`` — the user has said they know.
    """
    if suppress_warning or not unseeded_calls:
        return

    for call in unseeded_calls:
        message = format_stale_randomness_message(call)
        if not detector.mark_warned(code, message):
            continue
        warnings.warn_explicit(
            message, CashRandomnessWarning,
            filename='<cash>', lineno=call.lineno, registry=None,
        )

def check_and_warn_randomness(
    code: str,
    detector: RandomnessDetector,
    suppress_warning: bool = False
) -> tuple[list[RandomnessCallInfo], bool]:
    """
    Check code for unseeded randomness and issue warnings.

    Args:
        code: Python source code to analyze
        detector: RandomnessDetector instance (maintains session state)
        suppress_warning: If True, don't issue warnings (for @cash:allow-random)

    Returns:
        Tuple of (list of unseeded randomness calls, has_seed_calls)
        - has_seed_calls: True if the code contains seed function calls
          (seed statements should not be cached - they must execute to set RNG state)

    Warnings are deduped *once per statement per session* via
    ``detector.mark_warned``, and raised with :func:`warnings.warn_explicit`
    under the ``<cash>`` pseudo-filename Cash compiles user statements with.

    Two things make that the right call rather than a plain ``warnings.warn``:

    * ``registry=None`` bypasses ``__warningregistry__``'s "once per location"
      dedupe.  Every randomness warning is raised from this one source line, so
      the registry would collapse an entire session's worth into a single
      warning.  ``mark_warned`` supplies the per-statement dedupe we actually
      want instead.
    * It still consults the user's filters.  The previous approach —
      ``catch_warnings()`` + ``simplefilter('always')`` — bought registry-bypass
      by *overriding* those filters, which silently defeated the blanket
      ``filterwarnings(..., category=cash.CashWarning)`` recipe that
      :class:`~cash.exceptions.CashWarning` documents.

    ``stacklevel`` has no useful target here: the statement has not executed yet,
    so no user frame is on the stack — every candidate is Cash-internal.
    ``<cash>`` plus the in-message line number attributes it to the statement.
    """
    unseeded_calls, warnings_list, has_seed_calls = detector.analyze_code(code)

    if not suppress_warning and warnings_list:
        for call, warning_msg in zip(unseeded_calls, warnings_list):
            if not detector.mark_warned(code, warning_msg):
                continue
            warnings.warn_explicit(
                warning_msg, CashRandomnessWarning,
                filename='<cash>', lineno=call.lineno, registry=None,
            )

    return unseeded_calls, has_seed_calls


# -----------------------------------------------------------------------------
# Unseeded estimator fits
# -----------------------------------------------------------------------------
#
# made a bare ``estimator.fit(X, y)`` cacheable. But the AST channels
# above are blind to it: the randomness lives inside sklearn's compiled ``.fit()``
# (bootstrap sampling, feature subsampling, weight init), not in any Python call
# this module can see. An estimator built without a ``random_state`` therefore
# gets cached and FROZEN with no warning at all -- the notebook-path analogue of
# the inline unseeded-randomness hazard the detector already flags, and of the
# object-cache hazard covered on the decorator path.
#
# Detection cannot come off the source, so it is done by the caller against the
# LIVE estimator (``get_params()['random_state'] is None``); these helpers only
# format the message and route it through the SAME ``CashRandomnessWarning`` path
# -- same category, same ``<cash>`` pseudo-filename, same per-statement dedupe --
# so users' existing warning filters and the once-per-statement contract carry
# over unchanged. There is no ``lineno`` to attribute (the hazard is not on any
# one source line), so ``0`` is used.

def format_unseeded_estimator_fit_message(receiver: str) -> str:
    """Compute-time message for caching an unseeded estimator fit.

    The estimator-fit twin of the inline "detected" warning: advice about the
    source, framed for the cold run where the fitted model is fresh and correct
    but about to be frozen into the cache.
    """
    return (
        f"Unseeded randomness detected: {receiver}.fit() on an estimator with "
        f"random_state=None. cash caches the fit, but the cached fitted model is a "
        f"frozen replay, not a fresh fit - two genuine fits would differ. Pass "
        f"random_state=<int> to the estimator for reproducibility, or use "
        f"@cash:allow-random to suppress."
    )


def format_stale_estimator_fit_message(receiver: str) -> str:
    """Restore-time message for a replayed unseeded estimator fit.

    Mirrors :func:`format_stale_randomness_message`: a *different claim* from the
    compute-time one, earned only by a successful restore. The fitted model on
    screen definitively IS a replay, so "may differ" would understate it. The
    distinct text also lands in its own ``(code, message)`` dedupe slot.
    """
    # ASCII only (matches format_stale_randomness_message): this reaches a
    # kernel's stderr, where a Windows console codepage mangles an em-dash.
    return (
        f"Unseeded estimator fit restored from cache: {receiver}.fit() with "
        f"random_state=None. The fitted model you are seeing is a replay of an "
        f"earlier fit, not a fresh draw - re-running will not change it. Use "
        f"@cash:no-cache to re-fit every time, pass random_state=<int> for real "
        f"reproducibility, or @cash:allow-random to suppress."
    )


def warn_unseeded_estimator_fit(
    code: str,
    receivers: 'Iterable[str]',
    detector: RandomnessDetector,
    suppress_warning: bool = False,
) -> None:
    """Warn that an unseeded estimator ``.fit()`` is cached as a frozen replay.

    The estimator-fit analogue of :func:`check_and_warn_randomness`. ``receivers``
    are the unseeded estimator variable names, decided by the caller against the
    live namespace (the AST cannot see the randomness inside ``.fit()``).
    Everything else matches the inline path: the ``CashRandomnessWarning``
    category, the ``<cash>`` pseudo-filename, ``registry=None`` to bypass the
    "once per location" collapse, and ``detector.mark_warned`` for the
    once-per-statement-per-session dedupe.

    Args:
        code: Statement source, used as the dedupe key (edit it -> re-warn).
        receivers: Unseeded estimator receiver names to flag.
        detector: Supplies the shared per-statement dedupe ledger.
        suppress_warning: ``# @cash:allow-random`` -- the user has said they know.
    """
    if suppress_warning:
        return
    for receiver in receivers:
        message = format_unseeded_estimator_fit_message(receiver)
        if not detector.mark_warned(code, message):
            continue
        warnings.warn_explicit(
            message, CashRandomnessWarning,
            filename='<cash>', lineno=0, registry=None,
        )


def warn_stale_estimator_fit(
    code: str,
    receivers: 'Iterable[str]',
    detector: RandomnessDetector,
    suppress_warning: bool = False,
) -> None:
    """Announce that a cached unseeded estimator fit was replayed.

    Restore-time twin of :func:`warn_unseeded_estimator_fit`, matching how
    :func:`warn_stale_randomness` follows :func:`check_and_warn_randomness`:
    called ONLY after a restore has actually succeeded, because the claim is
    about the value handed back, not about the source.
    """
    if suppress_warning:
        return
    for receiver in receivers:
        message = format_stale_estimator_fit_message(receiver)
        if not detector.mark_warned(code, message):
            continue
        warnings.warn_explicit(
            message, CashRandomnessWarning,
            filename='<cash>', lineno=0, registry=None,
        )

# =============================================================================
# RNG State Capture and Restore
# =============================================================================

def capture_rng_state() -> dict:
    """
    Capture the current state of all supported RNG modules.

    Returns:
        Dict mapping module name to its RNG state (picklable).
        Only includes modules that are currently imported.
    """
    import sys
    state = {}

    # Standard library random
    if 'random' in sys.modules:
        import random
        try:
            state['random'] = random.getstate()
        except (TypeError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture random state: %s", e)

    # NumPy random
    if 'numpy' in sys.modules or 'numpy.random' in sys.modules:
        try:
            import numpy as np
            state['numpy.random'] = np.random.get_state()
        except (ImportError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture numpy random state: %s", e)

    # PyTorch (if available)
    if 'torch' in sys.modules:
        try:
            import torch
            state['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state['torch.cuda'] = torch.cuda.get_rng_state_all()
        except (ImportError, RuntimeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture torch random state: %s", e)

    # TensorFlow - uses global seed, harder to capture/restore
    # We'll skip TF for now as it doesn't have easy getstate/setstate

    return state

def restore_rng_state(state: dict) -> None:
    """
    Restore RNG state from a previously captured state dict.

    Args:
        state: Dict mapping module name to its RNG state.
    """
    import sys

    if not state:
        return

    # Standard library random
    if 'random' in state and 'random' in sys.modules:
        import random
        try:
            random.setstate(state['random'])
        except (TypeError, ValueError) as e:
            logger.debug("[RANDOMNESS] Failed to restore random state: %s", e)

    # NumPy random
    if 'numpy.random' in state and ('numpy' in sys.modules or 'numpy.random' in sys.modules):
        try:
            import numpy as np
            np.random.set_state(state['numpy.random'])
        except (ImportError, TypeError, ValueError) as e:
            logger.debug("[RANDOMNESS] Failed to restore numpy random state: %s", e)

    # PyTorch
    if 'torch' in state and 'torch' in sys.modules:
        try:
            import torch
            torch.set_rng_state(state['torch'])
            if 'torch.cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['torch.cuda'])
        except (ImportError, RuntimeError) as e:
            logger.debug("[RANDOMNESS] Failed to restore torch random state: %s", e)

# -----------------------------------------------------------------------------
# Per-object RNG carriers
# -----------------------------------------------------------------------------
#
# ``capture_rng_state`` / ``restore_rng_state`` above cover the RNG *module
# globals* (``random``, ``np.random``, ``torch``).  They cannot see a generator
# a user holds in a variable — ``rng = np.random.default_rng(42)`` — because
# that object has no module-level home.
#
# The asymmetry that motivates this: a statement drawing from the
# global channel (``np.random.randint``) replays correctly across a cache hit,
# because the post-state is captured and re-injected.  A statement drawing from
# an object-held generator (``rng.integers``) HITS and restores its output, but
# the live generator is never advanced — so the next draw repeats the values
# the cached statement already consumed.
#
# The fix mirrors the global channel exactly: capture the carrier's post-state
# at store time and inject it back on a hit.  Capture is scoped to the
# statement's declared inputs (not all of ``user_ns``) and gated on an
# ``isinstance`` allowlist, which bounds the cost to the handful of variables a
# statement actually reads.

# The carrier kinds (``_KIND_*``) and the source-level constructor tables live
# at the top of this module — the detector needs them too, and both channels
# must agree on what an RNG is.  See the "RNG carriers" section there.


def _classify_rng_carrier(obj: object) -> str | None:
    """Return the carrier kind for ``obj``, or ``None`` if it isn't one.

    Objects owned by the RNG *module globals* are deliberately excluded: the
    module channel in :func:`capture_rng_state` already replays those, and
    capturing them twice under a variable name would let a stale alias fight
    with the authoritative global state.
    """
    import sys

    if 'numpy' in sys.modules or 'numpy.random' in sys.modules:
        try:
            import numpy as np

            if isinstance(obj, np.random.Generator):
                return _KIND_NP_GENERATOR
            if isinstance(obj, np.random.RandomState):
                # ``np.random.*`` module functions delegate to this singleton;
                # the global channel owns it.
                if obj is not np.random.mtrand._rand:
                    return _KIND_NP_RANDOMSTATE
                return None
        except (ImportError, AttributeError):
            pass

    if 'random' in sys.modules:
        try:
            import random

            if isinstance(obj, random.Random):
                # ``random.*`` module functions delegate to this singleton.
                if obj is not getattr(random, '_inst', None):
                    return _KIND_PY_RANDOM
                return None
        except (ImportError, AttributeError):
            pass

    return None


def capture_object_rng_states(
    names: 'Iterable[str]', user_ns: dict[str, object],
) -> dict[str, dict]:
    """Capture the post-state of any RNG carrier bound to one of ``names``.

    Args:
        names: Variable names to consider — the statement's inputs.  Scoping to
            inputs is what bounds the cost: no full ``user_ns`` walk.
        user_ns: The shell namespace to resolve names against.

    Returns:
        ``{var_name: {'kind': <carrier kind>, 'state': <picklable state>}}``.
        Empty when the statement reads no RNG carriers, which is the common
        case — callers should omit the payload key entirely when empty so
        non-RNG statements keep their existing payload shape.
    """
    states: dict[str, dict] = {}

    for name in names:
        try:
            obj = user_ns.get(name)
        except (TypeError, AttributeError):
            continue
        if obj is None:
            continue

        try:
            kind = _classify_rng_carrier(obj)
            if kind is None:
                continue
            if kind == _KIND_NP_GENERATOR:
                # Foreign / third-party bit generators may raise or hand back
                # something unpicklable here; the except below drops them.
                state = obj.bit_generator.state
            elif kind == _KIND_NP_RANDOMSTATE:
                state = obj.get_state()
            else:
                state = obj.getstate()
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            # e.g. random.SystemRandom.getstate() raises NotImplementedError.
            logger.debug("[RANDOMNESS] Failed to capture RNG state for %r: %s", name, e)
            continue

        states[name] = {'kind': kind, 'state': state}

    return states


def restore_object_rng_states(
    states: dict[str, dict] | None, user_ns: dict[str, object],
) -> None:
    """Inject captured per-object RNG states back onto the live carriers.

    Guarded on presence (the name still resolves) and on type match (the live
    object is still the same kind of carrier).  A name that now holds something
    else is skipped rather than forced.

    Args:
        states: Mapping produced by :func:`capture_object_rng_states`.  Cache
            entries written before this field existed pass ``None`` here and
            restore unchanged.
        user_ns: The shell namespace to resolve names against.
    """
    if not states:
        return

    for name, entry in states.items():
        try:
            kind = entry['kind']
            state = entry['state']
        except (TypeError, KeyError):
            continue

        obj = user_ns.get(name)
        if obj is None:
            continue

        # Type match: only write the state back onto the same carrier kind.
        if _classify_rng_carrier(obj) != kind:
            continue

        try:
            if kind == _KIND_NP_GENERATOR:
                obj.bit_generator.state = state
            elif kind == _KIND_NP_RANDOMSTATE:
                obj.set_state(state)
            else:
                obj.setstate(state)
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            logger.debug("[RANDOMNESS] Failed to restore RNG state for %r: %s", name, e)


def get_used_rng_modules(code: str) -> set[str]:
    """
    Analyze code to determine which RNG modules are used.

    Args:
        code: Python source code

    Returns:
        Set of module names that have RNG calls (e.g., {'random', 'numpy.random'})
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    visitor = RandomnessVisitor()
    visitor.visit(tree)

    modules = set()
    for call in visitor.random_calls:
        modules.add(call.module)
    for module, _ in visitor.seed_calls:
        modules.add(module)
    # A draw off a carrier bound in this same source still uses the module's
    # RNG machinery, so the module belongs in the set.  Carriers bound in an
    # *earlier* statement are not visible here — this helper is deliberately
    # stateless, and the per-object channel (``capture_object_rng_states``) is
    # what replays those.
    for _name, (kind, _seeded) in visitor.carrier_assigns.items():
        modules.add(_CARRIER_MODULES.get(kind, kind))

    return modules


def get_drawing_rng_modules(code: str) -> set[str]:
    """Modules *code* DRAWS from, ignoring modules it merely seeds.

    The narrower companion to :func:`get_used_rng_modules`, which also reports
    the target of a bare ``np.random.seed(0)``. Only a draw's result depends on
    the RNG state, so only a draw's cache key should.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    visitor = RandomnessVisitor()
    visitor.visit(tree)
    return {call.module for call in visitor.random_calls}


def _is_entropy_seed(node: ast.Call) -> bool:
    """True for ``seed()`` / ``seed(None)`` — a reseed from OS entropy.

    ``seed(42)`` asks for reproducibility; ``seed(None)`` asks for the opposite,
    explicitly. A variable argument (``seed(s)``) is treated as deterministic:
    its value rides the statement's input lineage, which is what already makes
    an edited ``SEED = 999`` invalidate.
    """
    if node.keywords:
        for kw in node.keywords:
            if kw.arg in (None, "seed", "a") and isinstance(kw.value, ast.Constant):
                return kw.value.value is None
    if not node.args:
        return True
    first = node.args[0]
    return isinstance(first, ast.Constant) and first.value is None


def get_entropy_reseed_modules(code: str) -> set[str]:
    """Modules *code* re-seeds FROM ENTROPY (``np.random.seed(None)``).

    Such a statement is a declaration that this run should differ from the last.
    Cash's freeze contract covers draws with *no* seed at all; it must not
    extend to a stream the user has explicitly re-randomised, or every value
    computed after the reseed is replayed from a run whose stream no longer
    exists. That is the failure: a re-fit model whose accuracy was
    served from cache, so the printed metric described a model that had already
    been replaced in the kernel.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    visitor = RandomnessVisitor()
    visitor.visit(tree)
    return {module for module, _lineno in visitor.entropy_seed_calls}


def get_seeding_rng_modules(code: str) -> set[str]:
    """Modules *code* SEEDS (``np.random.seed(0)``, ``random.seed(0)``).

    The counterpart to :func:`get_drawing_rng_modules`. A statement that seeds
    opens a new "seed epoch" for its module; every later draw from that module
    is keyed on the epoch, so re-seeding invalidates the draws that follow it.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    visitor = RandomnessVisitor()
    visitor.visit(tree)
    return {module for module, _lineno in visitor.seed_calls}


# -----------------------------------------------------------------------------
# Hidden lineage dependencies (RNG state as a lineage-tracked virtual variable)
# -----------------------------------------------------------------------------
#
# A *hidden dependency* is global mutable state a statement reads or writes
# without binding a user-visible name -- the RNG generator's state is the first
# instance (``os.environ``, ``plt.rcParams`` are the same shape). Modelled as a
# virtual variable that rides the ORDINARY variable-lineage machinery: a
# producer (a ``seed()``) writes its lineage, a consumer (a draw) reads it, and
# every downstream consumer folds that lineage into its own output lineage
# through the same input-lineage code every real variable uses. This is what
# makes a seed change propagate to a cached downstream statement -- the half the
# epoch (a key-only side channel) never did.
#
# The engines (runtime ``StatementProcessor`` and the ``VirtualLineage``
# simulator) integrate at a single seam each via these three helpers; there is
# no RNG-specific branching elsewhere. Adding a second hidden dependency later
# means teaching these three functions about it, nothing more.

def rng_virtual_var(module: str) -> str:
    """Name of the virtual lineage variable modelling *module*'s global RNG state.

    Deliberately not a valid attribute path and never written to ``user_ns`` --
    it exists only as a key in the lineage dict, so it cannot collide with a
    user variable or be mistaken for one.
    """
    return f"__cash_rng__{module}"


# ---------------------------------------------------------------------------
# Seed-epoch registry.
#
# The statement engine owns the ledger (a seed opens a new epoch for its module,
# valued by the seeding statement's cache key). It was reachable only from
# ``StatementProcessor``, so ``@cash.cache`` -- which consumes the same global
# stream -- had no way to see that the seed had changed, and served a model
# trained under the previous one.
#
# The processor PUBLISHES its dict here by reference, so this is a view onto the
# live ledger rather than a second copy that could drift. A fresh processor
# replaces the pointer, which also keeps tests isolated.
# ---------------------------------------------------------------------------
_ACTIVE_SEED_EPOCHS: dict[str, str] | None = None


def publish_seed_epochs(epochs: dict[str, str] | None) -> None:
    """Expose the statement engine's live seed ledger to other front-ends."""
    global _ACTIVE_SEED_EPOCHS
    _ACTIVE_SEED_EPOCHS = epochs


def seed_epochs() -> dict[str, str]:
    """Snapshot of the current seed epoch per RNG module (may be empty)."""
    return dict(_ACTIVE_SEED_EPOCHS or {})


def seed_epoch_component(modules: set[str]) -> str:
    """Cache-key fragment for *modules*, or "" when none of them is seeded.

    Empty when unseeded, so a function that draws from an unseeded stream keeps
    the key it has today: the freeze contract still applies and the value is
    replayed. Only an actual seed -- or a change to one -- moves the key.
    """
    if not modules:
        return ""
    epochs = _ACTIVE_SEED_EPOCHS or {}
    parts = [f"{m}:{epochs[m]}" for m in sorted(modules) if m in epochs]
    return ":rng:" + ":".join(parts) if parts else ""


def observed_rng_reads(tracking_state, code: str) -> set[str]:
    """Virtual RNG variables *code* reads by prior runtime OBSERVATION.

    ``model.fit()`` consumes the global stream while its AST says nothing, so
    the runtime records what it saw and every engine reads that one ledger.

    Deliberately shared by all three seams -- the runtime cache key, the runtime
    output lineage, and the simulation. They must agree byte-for-byte (the
    cache-key unification rule); when only the first one had it, the simulation
    computed different keys and 65 integration tests failed. One function, so
    they cannot drift apart again.
    """
    ledger = getattr(tracking_state, "observed_rng_statement_draws", None)
    if not ledger or not code:
        return set()
    try:
        digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
    except (AttributeError, UnicodeEncodeError):
        return set()
    modules = ledger.get(digest)
    return {rng_virtual_var(m) for m in modules} if modules else set()


def hidden_lineage_reads(code: str) -> set[str]:
    """Virtual variables a statement READS: the RNG state a draw consumes.

    Folded into the statement's cache key AND its output lineage, so a re-seed
    upstream both re-keys the draw and propagates to everything cached
    downstream of it.
    """
    return {rng_virtual_var(m) for m in get_drawing_rng_modules(code)}


def hidden_lineage_writes(code: str) -> set[str]:
    """Virtual variables a statement PRODUCES: a ``seed()`` defines its RNG state."""
    return {rng_virtual_var(m) for m in get_seeding_rng_modules(code)}


def entropy_write_lineage() -> str:
    """Lineage for a stream the user explicitly re-randomised.

    ``seed(None)`` states that this run should differ from the last, so the
    virtual variable must NOT take a value derived from the statement's cache
    key -- that key is byte-identical run to run, which is exactly what let
    downstream consumers keep hitting across a genuinely new stream.

    Fresh per call, so every consumer below the reseed recomputes. The
    simulation cannot predict this value, and that is the correct outcome: it
    sees a changed input and re-runs, which is the safe direction.
    """
    return "entropy:" + secrets.token_hex(16)


def hidden_write_lineage(producing_key: str) -> str:
    """Lineage a hidden variable takes when a statement produces it.

    A ``seed()`` RESETS the generator, so the new lineage depends only on the
    seeding statement's own cache key (which already folds in its source and its
    input lineage -- so ``seed(0)`` -> ``seed(1)`` and ``seed(cfg.seed)`` are
    both caught). Overwrite, not chain: editing an *earlier* seed must not
    invalidate a draw governed by a *later* one.
    """
    return hashlib.sha256(producing_key.encode('utf-8')).hexdigest()


def seed_cells_not_yet_run(
    drawing_modules: set[str],
    notebook_cells: 'list[str]',
    executed_cell_hashes: set[str],
) -> 'list[tuple[str, int]]':
    """Notebook cells that SEED a drawn module but whose source has not run (ADR-017).

    The detection core for the bare-``seed()`` edit-without-rerun defect.
    Editing an ``np.random.seed(N)`` cell and running only a downstream draw
    leaves the draw on the *old* seed's state, silently — because a bare seed
    binds no variable, so nothing links the draw back to the seed cell.

    This finds the mismatch from the one place it is visible: the notebook. A
    cell whose source SEEDS a module that some draw consumes, but whose exact
    source hash is NOT among the cells that have executed this session, is an
    edited-but-not-rerun (or never-run) seed the draw will ignore.

    Pure and cell-granular — it matches the checker's view of the notebook and
    takes the executed-hash set as an argument, so it carries no state and is
    trivially testable. Callers decide what to do with the result (warn now;
    schedule a re-execution once ADR-017 half 2 lands).

    Parameters
    ----------
    drawing_modules : the RNG modules the current cell draws from.
    notebook_cells  : the notebook's cell sources, in order.
    executed_cell_hashes : ``sha256(source).hexdigest()`` for every cell source
        that has actually executed this session.

    Returns ``[(module, cell_index)]``, sorted, one per (unrun seed cell, module).
    """
    if not drawing_modules:
        return []
    out: list[tuple[str, int]] = []
    for idx, src in enumerate(notebook_cells):
        seeded = get_seeding_rng_modules(src) & drawing_modules
        if not seeded:
            continue
        digest = hashlib.sha256(src.encode('utf-8')).hexdigest()
        if digest in executed_cell_hashes:
            continue  # this exact seed cell source has run — not stale
        for module in sorted(seeded):
            out.append((module, idx))
    return out


def rng_lineage_fingerprint(
    variable_lineage: Mapping[str, str], modules: Iterable[str],
) -> tuple:
    """The seeds in force for *modules* — an invalidation key for a saved position.

    A recorded RNG start position stays valid only while the seed that
    determines it is unchanged. Rather than invent a staleness rule for that,
    reuse the one every real variable already uses: the RNG state is modelled as
    a hidden lineage variable per module (``__cash_rng__<module>``), written by a
    seeding statement and read by a draw, so comparing that lineage now against
    the lineage recorded alongside the position IS the standard check.

    An UNSEEDED module contributes ``None`` on both sides and therefore stays
    valid forever. That is deliberate: an unseeded stream has no semantically
    correct position, so cash freezes the one it first saw — the same
    "non-determinism frozen, not blocked" rule the rest of the system follows.
    """
    return tuple(sorted(
        (m, variable_lineage.get(rng_virtual_var(m))) for m in modules
    ))


def rng_modules_changed(before: dict, after: dict) -> set[str]:
    """Modules whose captured RNG state differs between *before* and *after* (ADR-018).

    Used to observe a draw that static analysis cannot see because it happens
    inside a called function. Compares the per-module states from two
    :func:`capture_rng_state` snapshots. A module present in only one snapshot
    (e.g. numpy imported mid-cell) counts as changed. Arrays are compared by
    bytes, never ``repr``, so display options cannot mask a difference.
    """
    changed: set[str] = set()
    for module in set(before) | set(after):
        b, a = before.get(module), after.get(module)
        if b is None or a is None:
            changed.add(module)
        elif not _rng_states_equal(b, a):
            changed.add(module)
    return changed


def _rng_states_equal(before: object, after: object) -> bool:
    """True when two captured per-module RNG states are identical.

    Compares the states DIRECTLY rather than digesting both sides. This is on the
    hot path -- the observer runs on every statement -- and digesting dominated
    it: the stdlib state is MT19937's 624 state words plus a position, carried as
    a 625-element tuple of Python ints, and feeding those ints one at a time into
    sha256 cost ~155us per comparison versus ~3.5us for numpy's ndarray (one
    ``tobytes``). Direct comparison does the same job ~100x faster (measured
    484us -> 4.6us for a two-module snapshot pair).

    Arrays are still compared by BYTES, never ``repr``, so display truncation can
    never mask a difference -- and torch tensors now go by bytes too (via
    ``.numpy()``), where the digest fallback would have used a TRUNCATED repr and
    could have called two different states equal.
    """
    if before is after:
        return True
    b_tobytes, a_tobytes = getattr(before, 'tobytes', None), getattr(after, 'tobytes', None)
    if callable(b_tobytes) and callable(a_tobytes):
        return b_tobytes() == a_tobytes()  # numpy ndarray
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)):
        if len(before) != len(after):
            return False
        try:
            # Fast path: an all-scalar sequence (the stdlib's 625 ints) compares
            # at C speed in one call. Raises when an element is an ndarray or
            # tensor -- whose ``==`` yields an array, making the tuple compare's
            # truthiness check ambiguous -- so those fall through to elementwise.
            return bool(before == after)
        except (ValueError, RuntimeError):
            return all(_rng_states_equal(x, y) for x, y in zip(before, after))
    if isinstance(before, (int, float, complex, str, bytes, bool, type(None))):
        return type(before) is type(after) and before == after
    b_np, a_np = getattr(before, 'numpy', None), getattr(after, 'numpy', None)
    if callable(b_np) and callable(a_np):
        try:
            return b_np().tobytes() == a_np().tobytes()  # torch tensor
        except (TypeError, ValueError, RuntimeError):
            pass
    # Anything exotic: fall back to the byte digest rather than risk a
    # container's ambiguous ``==`` (an ndarray/tensor compare returns an array).
    try:
        return _digest_rng_state(before) == _digest_rng_state(after)
    except (TypeError, ValueError, RuntimeError):
        return False


def _digest_rng_state(value: object) -> str:
    """Byte-digest of one module's RNG state (arrays by bytes, not repr)."""
    h = hashlib.sha256()

    def feed(item: object) -> None:
        tobytes = getattr(item, 'tobytes', None)
        if callable(tobytes):
            h.update(tobytes())
        elif isinstance(item, (tuple, list)):
            for sub in item:
                feed(sub)
        else:
            h.update(repr(item).encode('utf-8'))

    feed(value)
    return h.hexdigest()
