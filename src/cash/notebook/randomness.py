from __future__ import annotations

"""Detection of unseeded random calls that compromise cache reproducibility."""

import ast
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["CashRandomnessWarning", "RandomnessCallInfo", "RANDOM_FUNCTIONS", "SEED_FUNCTIONS", "MODULE_ALIASES", "RandomnessVisitor", "RandomnessDetector", "check_and_warn_randomness", "capture_rng_state", "restore_rng_state", "capture_object_rng_states", "restore_object_rng_states", "get_used_rng_modules"]

logger = logging.getLogger(__name__)

class CashRandomnessWarning(UserWarning):
    """Warning issued when unseeded randomness is detected in cached code."""

@dataclass
class RandomnessCallInfo:
    """Information about a detected randomness call."""
    module: str
    function: str
    lineno: int
    col_offset: int

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

class RandomnessVisitor(ast.NodeVisitor):
    """AST visitor that detects randomness and seed calls."""

    def __init__(self):
        self.random_calls: list[RandomnessCallInfo] = []
        self.seed_calls: list[tuple[str, int]] = []  # (module, lineno)
        self.imports: dict = {}  # name -> module mapping

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

        self.generic_visit(node)

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
                    return

                # Check if it matches a known full path seed like 'numpy.random.seed'
                # The SEED_FUNCTIONS keys are modules, values are function names or 'submod.func'
                for seed_func_signature in seed_funcs:
                     if resolved_full_name == f"{module}.{seed_func_signature}":
                         self.seed_calls.append((module, node.lineno))
                         return

        # 2. Handle direct function imports
        else:
             if func_name in self.imports:
                imported_from = self.imports[func_name]
                for module, _seed_funcs in SEED_FUNCTIONS.items():
                    if imported_from == f"{module}.{func_name}":
                        self.seed_calls.append((module, node.lineno))
                        return

class RandomnessDetector:
    """
    Detects randomness function calls without proper seed setting.

    Tracks seeding state across statements within a session to allow
    seeding in one cell to apply to subsequent cells.
    """

    def __init__(self):
        # Modules that have been seeded in the current session
        self.seeded_modules: set[str] = set()

    def reset(self):
        """Reset seeding state (e.g., for new session)."""
        self.seeded_modules.clear()

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
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [], [], False

        visitor = RandomnessVisitor()
        visitor.visit(tree)

        has_seed_calls = len(visitor.seed_calls) > 0

        # Update seeding state from any seed calls in this code
        for module, _lineno in visitor.seed_calls:
            self.mark_seeded(module)

        # Then check for unseeded random calls
        unseeded_calls = []
        warnings_list = []

        for call in visitor.random_calls:
            if not self.is_seeded(call.module):
                unseeded_calls.append(call)
                warnings_list.append(
                    f"Unseeded randomness detected: {call.module}.{call.function}() "
                    f"at line {call.lineno}. Cached results may not be reproducible. "
                    f"Consider calling seed() first or use @cash:allow-random to suppress."
                )

        return unseeded_calls, warnings_list, has_seed_calls

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
    """
    unseeded_calls, warnings_list, has_seed_calls = detector.analyze_code(code)

    if not suppress_warning and warnings_list:
        # Use 'always' filter so warnings show every time, not just once
        with warnings.catch_warnings():
            warnings.simplefilter('always', CashRandomnessWarning)
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, CashRandomnessWarning, stacklevel=4)

    return unseeded_calls, has_seed_calls

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
# The asymmetry that motivates this (CAS-90): a statement drawing from the
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

_KIND_NP_GENERATOR = 'numpy.Generator'
_KIND_NP_RANDOMSTATE = 'numpy.RandomState'
_KIND_PY_RANDOM = 'random.Random'


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

    return modules

