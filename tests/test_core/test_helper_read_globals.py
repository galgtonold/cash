"""A global read by a HELPER must be re-checked in the helper's own module.

`_fold_read_globals` runs twice: once for the decorated function, and once per
module-bounded helper on that helper's behalf. Both write their pre-call hash
into the same flat `{name: ...}` scratch dict. `_learn_mutating_captures` then
re-hashed every entry out of the DECORATED function's ``__globals__`` -- where a
helper's global does not exist. It hashed ``None``, saw a different digest, and
concluded the call had mutated the global.

Two consequences, and the second is the expensive one:

1. A warning telling the user their function modifies a global it only reads.
2. The name was added to `_mutating_globals`, so it stopped being folded into
   the key. The key therefore DIFFERED between the first process and the
   second, and a warm run recomputed. Measured on a three-document pipeline:
   the second run made 3 provider calls where it should have made 0.

The registry lives in `helper_registry.py` because same-module globals go down
the other branch and never showed the defect.
"""
from __future__ import annotations

import warnings

from cash import Cash
from cash.exceptions import CashImpurityWarning

from . import helper_registry


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _warnings_from(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args)
        return [str(w.message) for w in rec
                if issubclass(type(w.message), CashImpurityWarning)]


def test_a_registry_read_by_a_helper_is_not_reported_as_mutated(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def ask(q: str) -> str:
        return helper_registry.complete(q, model="fast")

    messages = _warnings_from(ask, "hello")
    mutation = [m for m in messages if "modifies the module global" in m]
    assert not mutation, mutation


def test_a_scalar_global_read_by_a_helper_is_not_reported_as_mutated(tmp_path):
    """Same defect, without a dict of callables -- so a fix that only special-
    cased dispatch tables would not pass this."""
    c = _cash(tmp_path)

    @c.cache
    def check(n: int) -> bool:
        return helper_registry.over_threshold(n)

    messages = _warnings_from(check, 9)
    mutation = [m for m in messages if "modifies the module global" in m]
    assert not mutation, mutation


def test_the_key_is_stable_across_calls_so_a_warm_call_hits(tmp_path):
    """The expensive half. A demoted global changes the key, so the entry
    written by the first call is never found again."""
    c = _cash(tmp_path)

    @c.cache
    def ask(q: str) -> str:
        return helper_registry.complete(q, model="fast")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ask("hello")
        ask("hello")

    info = ask.cache_info()
    assert info["hits"] == 1, info
    assert info["misses"] == 1, info


def test_a_helper_global_that_REALLY_changes_still_invalidates(tmp_path):
    """Control arm: the re-check must still work, not merely go quiet.

    Editing the registry between calls has to produce a different key -- if the
    fix had simply skipped names missing from ``func.__globals__``, this would
    keep serving the first answer.
    """
    c = _cash(tmp_path)

    @c.cache
    def ask(q: str) -> str:
        return helper_registry.complete(q, model="fast")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = ask("hello")
        original = helper_registry.MODELS["fast"]
        try:
            helper_registry.MODELS["fast"] = lambda prompt: f"REPLACED:{prompt}"
            second = ask("hello")
        finally:
            helper_registry.MODELS["fast"] = original

    assert first == "fast:hello"
    assert second == "REPLACED:hello", "swapping the registry entry must invalidate"
