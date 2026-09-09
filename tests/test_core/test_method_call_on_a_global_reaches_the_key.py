"""Reading a global through a method call still reads it.

Round-16 gate finding (WRONG ANSWER, 5/5 minimal, 3/3 on the full pipeline). A
module-level lookup table read as ``ALIASES.get(v, v)`` -- the ordinary way to
use a mapping -- never reached the cache key, so editing the table published
stale labels with nothing to see. The tester's matrix is what made it
undeniable, because every neighbouring spelling was correct::

    ALIASES                     bare read          tracked
    ALIASES[v]                  subscript          tracked
    v in ALIASES                membership         tracked
    d = ALIASES; d.get(v)       rebind, then call  tracked
    ALIASES.get(v, v)           method call        *** STALE ***

The rule was "a method call on a name may mutate it, and we cannot prove
otherwise", so the name was excluded from folding entirely. That is the same
over-broad refusal CAS-270 already fixed one level up for bare arguments --
`sum(G)`, `len(G)`, `helper(G)` -- where the answer was to fold the name and
confirm at runtime instead of refusing it.

Now the method-call rule only fires for methods that actually write (``append``,
``update``, ``sort`` … -- the table the purity analyzer uses for side effects).
Everything else is provisional: folded, then demoted by
``_learn_mutating_captures`` if calling the function is ever observed to move
the value. The accumulator protection this rule exists for is unchanged, and
the last two tests here pin it.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash, InMemoryBackend

from . import helper_registry


@pytest.fixture
def cash_instance():
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    yield c
    c.backend.clear()


ALIASES = {"emea": "eu"}
COUNTS = {"a": 1}
WORDS = ["alpha"]


def test_a_method_call_on_a_global_dict_reaches_the_key(cash_instance):
    """The reported shape."""
    runs: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def label(v):
        runs.append(v)
        return ALIASES.get(v, v)

    assert label("emea") == "eu"
    globals()["ALIASES"] = {"emea": "EUROPE"}
    try:
        assert label("emea") == "EUROPE", "the edited lookup table was ignored"
        assert len(runs) == 2
    finally:
        globals()["ALIASES"] = {"emea": "eu"}


@pytest.mark.parametrize("reader,expected_first,expected_second", [
    ("get", "eu", "EUROPE"),
    ("keys", "['emea']", "['emea', 'apac']"),
])
def test_other_read_only_methods_track_too(cash_instance, reader, expected_first,
                                           expected_second):
    """``.get`` is not special: any non-writing method is a read."""
    runs: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def read(v):
        runs.append(v)
        if reader == "get":
            return ALIASES.get(v, v)
        return str(list(ALIASES.keys()))

    assert read("emea") == expected_first
    globals()["ALIASES"] = ({"emea": "EUROPE"} if reader == "get"
                            else {"emea": "eu", "apac": "ap"})
    try:
        assert read("emea") == expected_second
        assert len(runs) == 2
    finally:
        globals()["ALIASES"] = {"emea": "eu"}


def test_the_neighbouring_spellings_still_track(cash_instance):
    """The control that the fix did not trade one spelling for another."""
    runs: list[int] = []

    @cash_instance.cache(assume_safe=True)
    def spellings(n):
        runs.append(n)
        return f"{ALIASES['emea']}|{'emea' in ALIASES}|{len(ALIASES)}"

    assert spellings(1) == "eu|True|1"
    globals()["ALIASES"] = {"emea": "EUROPE"}
    try:
        assert spellings(1) == "EUROPE|True|1"
        assert len(runs) == 2
    finally:
        globals()["ALIASES"] = {"emea": "eu"}


def test_an_unchanged_table_still_hits(cash_instance):
    """Without this, every assertion above passes on "never cache anything"."""
    runs: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def label(v):
        runs.append(v)
        return ALIASES.get(v, v)

    assert label("emea") == "eu"
    assert label("emea") == "eu"
    assert len(runs) == 1, f"the second call recomputed: {runs}"


# --- the accumulator protection this rule exists for ------------------------

def test_a_written_global_is_still_refused(cash_instance):
    """``append`` writes, so the name stays out of the key.

    Folding an accumulator the call itself moves would key every entry on the
    previous call's output and miss for ever -- the trap the method-call rule
    was built for. Narrowing it to writing methods must not open that.
    """
    runs: list[int] = []

    @cash_instance.cache(assume_safe=True)
    def grow(n):
        runs.append(n)
        WORDS.append(f"w{n}")
        return len(WORDS)

    first = grow(1)
    second = grow(1)
    assert second == first, "the entry keyed on its own output"
    assert len(runs) == 1


def test_a_mutating_method_the_table_does_not_know_is_learned_at_runtime(cash_instance):
    """The provisional half: folded, then demoted when observed to move.

    ``setdefault`` is not in the write-method table, so it is folded on the
    first call. The runtime observer sees the value change as a result of the
    call and stops folding it -- which is what keeps the perpetual-miss trap
    closed for methods nobody enumerated.
    """
    runs: list[int] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @cash_instance.cache(assume_safe=True)
        def touch(n):
            runs.append(n)
            COUNTS.setdefault(f"k{n}", n)
            return len(COUNTS)

        touch(1)
        touch(1)
        touch(1)

    assert len(runs) <= 2, (
        f"a self-mutating global was folded for ever: {len(runs)} executions"
    )


def test_a_module_attribute_read_by_method_call(cash_instance):
    """The other spelling of the same read: ``mod.TABLE.keys()``.

    A library keeps its lookup in another module and the consumer reaches it as
    an attribute. This channel already worked -- pinned so narrowing the
    method-call rule cannot cost it.
    """
    runs: list[str] = []
    original = dict(helper_registry.MODELS)

    @cash_instance.cache(assume_safe=True)
    def summarise(prompt):
        runs.append(prompt)
        return ",".join(sorted(helper_registry.MODELS.keys())) + f"|{prompt}"

    assert summarise("p") == "fast,smart|p"
    helper_registry.MODELS["extra"] = helper_registry._fast
    try:
        assert summarise("p") == "extra,fast,smart|p", (
            "the edited registry in another module was ignored"
        )
        assert len(runs) == 2
    finally:
        helper_registry.MODELS.clear()
        helper_registry.MODELS.update(original)


# --------------------------------------------------------------------------- #
# Dunder-named constants                                                       #
# --------------------------------------------------------------------------- #

__version__ = "0.1.0"


def test_a_version_dunder_reaches_the_key(cash_instance):
    """A second round-16 finding, same file because it is the same channel.

    Every dunder global used to be skipped. The tester bumped ``__version__``,
    watched it invalidate nothing, and kept publishing a report stamped with
    the old version through three further edits that each correctly invalidated
    other stages. ``RELEASE`` in the same module was tracked; only the dunder
    spelling was not, and no documented knob fixed it -- ``depends_on=[getter]``
    snapshots the getter's source, and a local rebind does not help either.
    """
    runs: list[int] = []

    @cash_instance.cache(assume_safe=True)
    def stamp(x):
        runs.append(x)
        return f"{__version__}|{x}"

    assert stamp(1) == "0.1.0|1"
    globals()["__version__"] = "0.2.0"
    try:
        assert stamp(1) == "0.2.0|1", "bumping __version__ invalidated nothing"
        assert len(runs) == 2
    finally:
        globals()["__version__"] = "0.1.0"


def test_the_machinery_dunders_are_still_skipped(cash_instance):
    """The control, and the reason the blanket rule existed.

    ``__file__`` and ``__name__`` differ per checkout and per invocation, so
    folding them would make a key un-shareable between two machines and between
    ``python job.py`` and ``python -m job``. They must stay out.
    """
    from cash.core import Cash as CashClass

    def reads_machinery(x):
        return f"{__file__}|{__name__}|{__doc__}|{x}"

    names = cash_instance._read_global_data_names(reads_machinery)

    assert "__file__" not in names
    assert "__name__" not in names
    assert "__doc__" not in names
    assert CashClass._MACHINERY_DUNDERS.isdisjoint(names)


def test_a_user_dunder_is_a_candidate(cash_instance):
    """The other half of the same rule, asserted where it is decided."""
    def reads_version(x):
        return f"{__version__}|{x}"

    assert "__version__" in cash_instance._read_global_data_names(reads_version)
