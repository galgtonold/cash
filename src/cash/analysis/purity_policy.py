"""The decorator's effect policy, and the method names it reports on any
receiver."""

from __future__ import annotations

from ..effects import METHOD_VERBS, MUTATOR_METHODS, Action, EffectKind

#: What a ``@cash.cache`` function's first call does about each kind of effect
#: its body has. What a kind IS lives in :mod:`cash.effects`, shared with the
#: notebook, whose own table is ``cash.analysis.file_effects.NOTEBOOK_POLICY``.
#: A decorated function is always cached -- refusing would cost the user the
#: compute and prevent nothing, since the body has run -- so the choice here is
#: only what to say. A test keeps this covering every kind.
DECORATOR_POLICY: dict[EffectKind, Action] = {
    EffectKind.FILE_WRITE: Action.WARN,
    EffectKind.FILE_READ: Action.CACHE_AS_INPUT,
    # What the server returns is an input the key cannot see: advise `ttl=`,
    # which silences it (KEY-NETWORK-READ). A database is a server too.
    EffectKind.NETWORK_READ: Action.SUGGEST_TTL,
    EffectKind.NETWORK_WRITE: Action.WARN,
    EffectKind.NETWORK: Action.WARN,
    EffectKind.DB_READ: Action.SUGGEST_TTL,
    EffectKind.DB_WRITE: Action.WARN,
    EffectKind.SUBPROCESS: Action.WARN,
    # These two are reported as ambient reads (KEY-AMBIENT-READ), not as
    # side effects: a hidden input is frozen, nothing is skipped.
    EffectKind.CLOCK: Action.WARN,
    # A read whose name is written out is folded into the key by value
    # (`EnvironmentFold.fold_environment`); one whose name is only known at run time
    # still warns, as an ambient read.
    EffectKind.ENVIRONMENT: Action.CACHE_AS_INPUT,
    # A hit drops what the first call printed. A log line (`is_log_line`) is
    # exempt: a hit skipping it is what caching means.
    EffectKind.CONSOLE: Action.WARN,
    EffectKind.DISPLAY: Action.WARN,
    EffectKind.INTERACTIVE: Action.WARN,
}


#: Kinds reported as ambient reads rather than as side effects.
AMBIENT_KINDS = frozenset({EffectKind.CLOCK, EffectKind.ENVIRONMENT})


#: Method names the decorator reports on any receiver: a mutator, or a verb
#: whose kind it warns about. The discarded-call rule skips these (the call is
#: already reported), and "does this change a global?" reads them.
REPORTED_METHODS: frozenset[str] = MUTATOR_METHODS | frozenset(
    name for name, kind in METHOD_VERBS.items() if DECORATOR_POLICY[kind] is Action.WARN
)
