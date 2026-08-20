import pytest

import cash
from cash import Cash as CashCls


def test_mark_opaque_records_the_type():
    c = CashCls()
    class Marker: pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker) is True


def test_an_unmarked_type_is_not_opaque():
    c = CashCls()
    class Other: pass
    assert c._is_opaque(Other) is False


def test_the_opaque_decorator_marks_and_returns_the_class():
    @cash.opaque
    class Decorated: pass
    assert Decorated.__name__ == "Decorated"      # returns the class, not a wrapper
    assert cash.Cash()._is_opaque(Decorated) is True


def test_the_opaque_decorator_does_not_replace_the_class_object():
    """`__name__` matching a wrapper's proxied name would not actually rule
    out a wrapper -- identity is the real proof. A decorator that silently
    swapped in a replacement would break every `isinstance` check downstream
    (pickle-by-reference in particular, which is the whole reason this
    feature exists), so pin object identity directly rather than a proxy
    for it.
    """
    class Plain: pass
    original = Plain
    result = cash.opaque(Plain)
    assert result is original
    assert id(result) == id(original)
    # Still a fully working class through the returned object: construction
    # and isinstance both resolve through the SAME class, not a stand-in.
    instance = result()
    assert isinstance(instance, Plain)


def test_a_subclass_of_an_opaque_class_does_not_inherit_opacity():
    """Deliberate design choice, not an oversight: ``@cash.opaque`` marks
    ONE class. A subclass may carry its own freshly-written methods the
    user actively edits, and inheriting opacity from an ancestor would
    silently exempt that new code from ever invalidating the cache -- for a
    decision made about a different class. A subclass that wants the same
    treatment decorates itself.

    The first assertion is the control: it must stay True, or the second
    assertion (False) would pass for the wrong reason -- e.g. a broken
    ``_is_opaque`` that simply always returns False.
    """
    @cash.opaque
    class Base: pass

    class Derived(Base): pass

    c = cash.Cash()
    assert c._is_opaque(Base) is True       # control: the decorated class itself
    assert c._is_opaque(Derived) is False   # the actual claim: no inheritance


def test_mark_opaque_does_not_cover_a_subclass():
    """Same contract as the decorator spelling, pinned independently: a
    plain set has no notion of "and its descendants," so registering a base
    through ``mark_opaque`` must not silently cover a subclass nobody
    registered. Keeps the two spellings -- "the same thing" per the
    docstring -- actually equivalent, rather than diverging on inheritance
    because one happens to be implemented as a dunder attribute and the
    other as a set.
    """
    c = CashCls()
    class Marker: pass
    class SubMarker(Marker): pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker) is True     # control: the registered type itself
    assert c._is_opaque(SubMarker) is False  # the actual claim: no inheritance


def test_is_opaque_never_raises_on_an_unhashable_class():
    """``target in Cash._OPAQUE_TYPES`` needs `target` to be hashable. A
    metaclass that defines ``__eq__`` without ``__hash__`` makes the CLASS
    ITSELF unhashable -- Python's data-model default, not just its
    instances -- which is a real, if unusual, class shape (some ORM/model
    metaclasses do this), not a contrived one.

    The `pytest.raises` block is the in-test control that proves the
    premise by measurement: if a future Python version stopped making such
    a class unhashable, this test would fail loudly on its own setup rather
    than silently stop testing anything.
    """
    class WeirdMeta(type):
        def __eq__(cls, other):
            return NotImplemented

    class Foo(metaclass=WeirdMeta): pass

    with pytest.raises(TypeError):
        {Foo}  # confirms Foo really is unhashable before trusting the rest

    assert cash.Cash()._is_opaque(Foo) is False


def test_an_instance_of_a_registered_type_is_opaque():
    """Every test above passes ``_is_opaque`` a class directly, which only
    ever exercises the ``isinstance(obj, type)`` branch of ``target = obj if
    isinstance(obj, type) else type(obj)``. This is the one that exercises
    the OTHER branch: ``_is_opaque``'s own docstring promises "a class, or
    an instance of one," and Task 4 needs that -- a callable object with a
    marked class is yielded to ``_is_opaque`` as the INSTANCE, not
    ``type(instance)`` (see ``_iter_code_carriers``'s ``callable(value)``
    branch in the Task 4 plan).
    """
    c = CashCls()
    class Marker: pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker()) is True


def test_an_instance_of_a_decorated_class_is_opaque():
    """Same gap, for the ``@cash.opaque`` spelling."""
    @cash.opaque
    class Decorated: pass
    assert cash.Cash()._is_opaque(Decorated()) is True


# ---------------------------------------------------------------------------
# Task 4: user code reached THROUGH THE ARGUMENTS folds into the cache key.
# ---------------------------------------------------------------------------

import functools
import itertools
import sys
import types
import warnings

from cash.exceptions import CashImpurityWarning

_nb_counter = itertools.count()

_V1 = "class S:\n    def r(self): return 'V1'\n"
_V2 = "class S:\n    def r(self): return 'V2'\n"


def _nb_module():
    """One registered, fileless module standing in for a notebook's ``__main__``.

    Every version of an edited class or function goes into THE SAME module,
    which is what a real kernel does when you re-run an edited cell.

    This is load-bearing, not cosmetic. Pickle serializes a class BY REFERENCE
    -- module plus qualname -- so two versions defined in two DIFFERENT modules
    already differ in ``args_hash``, and every invalidation assertion below
    would then hold with the code fold deleted outright. Measured on unmodified
    core.py with a fresh module per version: all four invalidation tests passed
    and three of the "should already pass" tests failed, i.e. exactly backwards.
    Same module, same qualname => byte-identical pickles => the code fold is
    the ONLY thing that can move the key.
    """
    mod = types.ModuleType(f"_cash_test_nb_codeargs_{next(_nb_counter)}")
    sys.modules[mod.__name__] = mod
    return mod


def _code_advisories(records):
    """Only THIS feature's advisory, out of everything ``catch_warnings``
    recorded. The counter in ``_counting`` also draws a ``CashImpurityWarning``
    from the purity analyzer (``calls.append()`` is a write method), so
    filtering on the category alone would count an unrelated warning as this
    one -- measured: it did."""
    return [
        str(w.message) for w in records
        if issubclass(w.category, CashImpurityWarning)
        and "its code could not be hashed" in str(w.message)
    ]


def _define(mod, body, name="S"):
    """Exec *body* into *mod*, as a re-run notebook cell does, and return the
    (re)defined object. Call ``takes(old)`` BEFORE redefining: pickle refuses a
    class its module no longer resolves to ("it's not the same object")."""
    exec(body, mod.__dict__)
    return getattr(mod, name)


@pytest.fixture()
def c(tmp_path):
    return CashCls(cache_dir=str(tmp_path))


@pytest.fixture()
def opaque_registry():
    """Isolate the process-global opaque registry.

    ``mark_opaque`` writes to ``Cash._OPAQUE_TYPES``, which every Cash instance
    in the process shares and nothing ever clears. The suite runs under
    ``--dist worksteal``, which splits ONE file across workers, so a leak here
    is a cross-test dependency that only appears under parallelism.
    """
    saved = set(CashCls._OPAQUE_TYPES)
    yield
    CashCls._OPAQUE_TYPES.clear()
    CashCls._OPAQUE_TYPES.update(saved)


@pytest.fixture()
def warned_unhashable():
    """Isolate the process-global advisory dedup set, for the same reason."""
    saved = set(CashCls._WARNED_UNHASHABLE)
    CashCls._WARNED_UNHASHABLE.clear()
    yield
    CashCls._WARNED_UNHASHABLE.clear()
    CashCls._WARNED_UNHASHABLE.update(saved)


def _counting(c, name="takes"):
    """A cached function plus a call counter, so ``len(calls)`` is the oracle
    for "did this recompute?".

    ``calls`` IS mutated by the body -- the counter has to be. That is safe
    here only because the counter is not part of the key, and every test below
    that asserts a MISS also asserts a HIT on a repeat call, which is exactly
    the assertion that would fail if the counter leaked into the key.

    *name* renames the wrapper before decoration so a single test can hold two
    independent cached functions; ``func_name`` is module+qualname, so two
    plain ``_counting(c)`` calls would otherwise share one cache entry.
    """
    calls = []

    def takes(x):
        calls.append(1)
        return len(calls)

    takes.__name__ = name
    takes.__qualname__ = f"_counting.<locals>.{name}"
    return c.cache(takes), calls


def test_a_class_argument_invalidates_when_its_body_changes(c):
    """The motivating bug. The two HIT assertions are in-test controls: an
    implementation that simply never caches (unhashable argument, key error)
    would otherwise satisfy the MISS assertion for the wrong reason."""
    takes, calls = _counting(c)
    nb = _nb_module()
    v1 = _define(nb, _V1)
    takes(v1)
    assert len(calls) == 1
    takes(v1)
    assert len(calls) == 1, "control: a repeat call on the same class must HIT"
    v2 = _define(nb, _V2)
    takes(v2)
    assert len(calls) == 2, "editing the passed class did not invalidate"
    takes(v2)
    assert len(calls) == 2, "control: the post-edit key must be warm too"


def test_the_control_an_unchanged_class_still_hits(c):
    """Without this, an implementation that simply never caches passes the
    test above."""
    takes, calls = _counting(c)
    nb = _nb_module()
    takes(_define(nb, _V1))
    takes(_define(nb, _V1))
    assert len(calls) == 1, "an identical class re-definition should still hit"


def test_a_function_argument_invalidates_when_its_body_changes(c):
    takes, calls = _counting(c)
    nb = _nb_module()
    f1 = _define(nb, "def f(): return 'V1'\n", name="f")
    takes(f1)
    takes(f1)
    assert len(calls) == 1, "control: the same function twice must HIT"
    takes(_define(nb, "def f(): return 'V2'\n", name="f"))
    assert len(calls) == 2


def test_a_class_nested_in_a_container_is_reached(c):
    """The motivating shape: the class arrives inside an options dict."""
    takes, calls = _counting(c)
    nb = _nb_module()
    v1 = _define(nb, _V1)
    takes({"temperature": 0.7, "responseType": v1})
    takes({"temperature": 0.7, "responseType": v1})
    assert len(calls) == 1, "control: the same options dict twice must HIT"
    v2 = _define(nb, _V2)
    takes({"temperature": 0.7, "responseType": v2})
    assert len(calls) == 2


def test_an_instance_invalidates_when_its_class_body_changes(c):
    takes, calls = _counting(c)
    nb = _nb_module()
    v1 = _define(nb, _V1)
    takes(v1())
    takes(v1())
    assert len(calls) == 1, "control: two equal instances of one class must HIT"
    takes(_define(nb, _V2)())
    assert len(calls) == 2


def test_a_callable_instance_folds_its_class_code(c):
    """A deviation from the task brief, pinned so it cannot regress silently.

    The brief's walk yielded any ``callable(value)`` and returned. A callable
    INSTANCE has no ``__code__`` of its own -- its code lives on its class --
    so ``_code_surface_hash`` returned None for it and it contributed nothing,
    while the SAME class without ``__call__`` folded fine through the
    ``__dict__`` branch. Measured: adding ``__call__`` to a class removed that
    class from the key and additionally tripped the unhashable advisory.
    """
    takes, calls = _counting(c)
    nb = _nb_module()
    body = "class S:\n    def __call__(self): return {v!r}\n"
    v1 = _define(nb, body.format(v="V1"))
    takes(v1())
    takes(v1())
    assert len(calls) == 1, "control: two equal callable instances must HIT"
    takes(_define(nb, body.format(v="V2"))())
    assert len(calls) == 2, "editing __call__ did not invalidate"


def test_an_opaque_class_does_not_invalidate(c, opaque_registry):
    """The second arm is the control: the identical edit on an UNMARKED pair
    must invalidate, or this test would also pass against an implementation
    that never folds a class argument at all."""
    opaque_takes, opaque_calls = _counting(c, name="opaque_takes")
    nb = _nb_module()
    v1 = _define(nb, _V1)
    CashCls.mark_opaque(v1)
    opaque_takes(v1)
    v2 = _define(nb, _V2)
    CashCls.mark_opaque(v2)
    opaque_takes(v2)
    assert len(opaque_calls) == 1, "an opaque class must not participate in the key"

    plain_takes, plain_calls = _counting(c, name="plain_takes")
    nb2 = _nb_module()
    plain_takes(_define(nb2, _V1))
    plain_takes(_define(nb2, _V2))
    assert len(plain_calls) == 2, "control: the same edit, unmarked, must invalidate"


def test_a_third_party_class_does_not_participate(c):
    """Pins the gate in the other direction, so a future widening that folds
    library code shows up as a failure rather than as a slow cache."""
    import json.encoder
    takes, calls = _counting(c)
    takes(json.encoder.JSONEncoder)
    takes(json.encoder.JSONEncoder)
    assert len(calls) == 1


def test_the_third_party_gate_is_measurable_not_merely_stable(c):
    """The end-to-end test above passes the SAME class object twice, so it
    holds even if library code IS folded -- one class folds to one digest
    either way. It can therefore only catch a widening that makes library code
    UNHASHABLE, never one that keys on it. Measure the gate itself, with a user
    class as the control that must move the hash."""
    import json.encoder
    base = "0" * 64
    assert c._fold_code_args((json.encoder.JSONEncoder,), {}, base) == base
    nb = _nb_module()
    assert c._fold_code_args((_define(nb, _V1),), {}, base) != base


def test_a_comment_only_edit_does_not_invalidate(c):
    """Pins the bytecode primitive's advantage. Comments are absent from
    bytecode, so a reformat must not cost a recompute -- and a future switch
    back to source hashing becomes a visible regression rather than a quiet
    slowdown. The second arm is the control: a REAL edit in the same shape
    must still invalidate."""
    takes, calls = _counting(c)
    nb = _nb_module()
    takes(_define(nb, _V1))
    takes(_define(nb, "class S:\n    # a newly added comment\n    def r(self): return 'V1'\n"))
    assert len(calls) == 1
    takes(_define(nb, _V2))
    assert len(calls) == 2, "control: a real body edit must still invalidate"


def test_a_changed_default_on_a_passed_callable_invalidates(c):
    """``__defaults__`` is NOT in ``co_code``, so this only passes because
    ``_code_identity`` folds it in explicitly."""
    takes, calls = _counting(c)
    nb = _nb_module()
    f1 = _define(nb, "def f(k=7): return k\n", name="f")
    takes(f1)
    takes(f1)
    assert len(calls) == 1, "control: the same function twice must HIT"
    takes(_define(nb, "def f(k=9): return k\n", name="f"))
    assert len(calls) == 2


def test_a_changed_closure_cell_does_NOT_invalidate(c):
    """A decision on the record, not an oversight.

    ``__closure__`` cells hold arbitrary live objects, and folding them would
    drag the whole argument-hashing problem into the code channel. If this test
    ever starts failing because someone added closure support deliberately,
    delete it -- but do not let it start passing by accident.

    The ``__qualname__`` rebind is what makes the test able to measure anything
    at all: a factory-made closure's qualname is ``make.<locals>.f``, which
    pickle refuses outright, and an unpicklable argument means the call never
    caches -- so BOTH halves would "miss" and the test would read as a pass for
    closure-sensitivity when it is really a pass for not-caching. Rebinding to
    the module-level name the object is actually stored under makes it pickle
    by reference, identically for both versions. The final arm is the control:
    an edit to the same carrier's BODY must still invalidate.
    """
    takes, calls = _counting(c)
    nb = _nb_module()
    _define(nb, "def make(v):\n    def f(): return v\n    return f\n", name="make")

    def _publish(v):
        g = nb.make(v)
        g.__name__ = g.__qualname__ = "g"
        nb.g = g
        return g

    takes(_publish("V1"))
    takes(_publish("V2"))
    assert len(calls) == 1

    _define(nb, "def make(v):\n    def f(): return (v, 'edited')\n    return f\n", name="make")
    takes(_publish("V2"))
    assert len(calls) == 2, "control: editing the closure's BODY must invalidate"


def test_explain_agrees_with_a_real_call_for_a_code_argument(c):
    """The two key-building paths must not disagree.

    ``_resolve_cache_key`` and ``_explain_call`` each build a key from their
    own ``_state_hasher.compute`` chain. If only one folds code arguments,
    ``explain()`` reports ``no_entry`` for a call that in fact hits -- the
    silent-divergence failure this task is most likely to produce. The first
    assertion is the cold control, so a broken ``explain`` that always answers
    False cannot pass.
    """
    takes, _calls = _counting(c)
    nb = _nb_module()
    v1 = _define(nb, _V1)
    assert takes.explain(v1).would_hit is False      # control: cold
    takes(v1)
    assert takes.explain(v1).would_hit is True       # the claim: same key both ways
    assert takes.explain(_define(nb, _V2)).would_hit is False


def test_a_partial_argument_warns_at_most_once_across_distinct_partials(c, warned_unhashable):
    """``repr(functools.partial(...))`` embeds a memory address, so keying the
    once-per-type dedup on ``repr()`` -- as the task brief's fallback did --
    warns for EVERY partial ever constructed and grows a class-global set
    without bound. ``assert == 1`` is two-sided: 0 would mean the advisory
    never fires (dead code), 3 would mean the dedup does not dedup.
    """
    takes, _calls = _counting(c)
    nb = _nb_module()
    _define(nb, "def scale(k, x): return k * x\n", name="scale")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        takes(functools.partial(nb.scale, 2))
        takes(functools.partial(nb.scale, 3))
        takes(functools.partial(nb.scale, 4))
    advisories = _code_advisories(rec)
    assert len(advisories) == 1, advisories
    assert "0x" not in advisories[0], "the advisory leaked an address"


def test_ordinary_arguments_still_hit_warm_and_stay_silent(c):
    """The walk now runs over EVERY argument of every cached call. Plain data
    must neither raise, nor warn, nor move the key between two equal calls."""
    takes, calls = _counting(c)
    payload = {"a": [1, 2, 3], "b": {"x", "y"}, "c": (4.5, None, b"z"), 7: "k"}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        takes(payload)
        takes(payload)
    assert len(calls) == 1
    assert _code_advisories(rec) == []


async def test_the_async_wrapper_folds_code_arguments_too(c):
    """The async wrapper is a second production entry point into the ONE
    ``_resolve_cache_key`` the fold is wired into. Pinned separately because
    "they share a helper" is an argument, not a measurement, and the async
    wrapper reaches that helper down its own path.
    """
    calls = []

    @c.cache
    async def atakes(x):
        calls.append(1)
        return len(calls)

    nb = _nb_module()
    v1 = _define(nb, _V1)
    await atakes(v1)
    await atakes(v1)
    assert len(calls) == 1, "control: a repeat async call must HIT"
    await atakes(_define(nb, _V2))
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# The walk must not lose a user class to what it INHERITS FROM, or to how it
# is DECLARED. Every shape below was verified to pickle byte-identically
# across its edit, so `args_hash` cannot move and the code fold is the only
# thing that can.
# ---------------------------------------------------------------------------


def _assert_edit_invalidates(c, template, build, name, func_name):
    """Define *template* at V1, call twice, redefine at V2, call again.

    The repeat call is an in-test control: "never caches" and "argument
    unhashable" both satisfy the MISS assertion on their own, and only the HIT
    assertion rules them out. *build* turns the (re)defined object into the
    argument, and is called afresh each time so the two V1 calls pass two
    DISTINCT but equal objects -- a warm hit on object identity would prove
    nothing.
    """
    takes, calls = _counting(c, name=func_name)
    nb = _nb_module()
    _define(nb, template.format(v="V1"), name=name)
    takes(build(nb))
    takes(build(nb))
    assert len(calls) == 1, "control: an unedited argument must HIT"
    _define(nb, template.format(v="V2"), name=name)
    takes(build(nb))
    assert len(calls) == 2, "editing the argument's class did not invalidate"


def test_a_str_subclass_argument_folds_its_class(c):
    """`isinstance(v, (str, int, ...))` is true for SUBCLASSES, so a fast path
    written with it skips a user class deriving from `str` before it ever
    reaches the fold -- a missed invalidation on exactly the bug class this
    feature exists for. The exact-type test is what makes this pass."""
    _assert_edit_invalidates(
        c, "class StrSub(str):\n    def r(self): return {v!r}\n",
        lambda mod: mod.StrSub("fixed"), name="StrSub", func_name="strsub_takes",
    )


def test_an_int_enum_member_folds_its_enum(c):
    """The same hole with no hand-written subclassing: an `IntEnum` member's
    type IS the user's own enum class, and it is also an `int`. Its pickle does
    not carry the method body at all, so `args_hash` cannot see this edit."""
    _assert_edit_invalidates(
        c,
        "import enum\n"
        "class Col(enum.IntEnum):\n    A = 1\n    def label(self): return {v!r}\n",
        lambda mod: mod.Col.A, name="Col", func_name="enum_takes",
    )


def test_a_slots_instance_folds_its_class(c):
    """`__slots__` gives instances no `__dict__`, which says nothing about
    whether the user edits the class. Gating the instance branch on
    `hasattr(value, "__dict__")` dropped every one of them."""
    def build(mod):
        obj = mod.Slotted()
        obj.a = 1
        return obj

    _assert_edit_invalidates(
        c, "class Slotted:\n    __slots__ = ('a',)\n    def r(self): return {v!r}\n",
        build, name="Slotted", func_name="slots_takes",
    )


def test_a_dict_subclass_instance_folds_its_class(c):
    """A `dict` subclass is BOTH a container to walk and a user object whose
    class carries code. Walking only the contents made the class invisible."""
    _assert_edit_invalidates(
        c, "class Opts(dict):\n    def r(self): return {v!r}\n",
        lambda mod: mod.Opts(a=1), name="Opts", func_name="dictsub_takes",
    )


def test_a_list_subclass_instance_folds_its_class(c):
    """Same hole on the sequence branch, pinned independently: the dict and
    sequence branches are separate code paths."""
    _assert_edit_invalidates(
        c, "class Seq(list):\n    def r(self): return {v!r}\n",
        lambda mod: mod.Seq([1, 2]), name="Seq", func_name="listsub_takes",
    )


def test_a_self_referential_container_argument_terminates(c):
    """The `_seen` set is no longer written for every element, only for the
    containers that can actually cycle and the carriers that are yielded. Pin
    the property that motivated it, so a future narrowing cannot quietly
    reintroduce unbounded recursion. The class assertion is the control: the
    walk must still REACH the payload, not merely survive it."""
    takes, calls = _counting(c, name="cycle_takes")
    nb = _nb_module()
    v1 = _define(nb, _V1)
    payload = [v1]
    payload.append(payload)
    assert [x.__qualname__ for x in c._iter_code_carriers(payload)] == ["S"]
    base = "0" * 64
    assert c._fold_code_args((payload,), {}, base) != base
    assert len(calls) == 0  # nothing above should have called the function


def test_the_advisory_ignores_a_stdlib_callable_object(c, warned_unhashable):
    """`_is_user_code_object` answers "could not confirm -> user code", which is
    the right direction for deciding whether to HASH and the wrong one for
    deciding whether to WARN: a `functools.partial` and a `weakref.ref` have no
    `__qualname__` of their own, so neither can ever be confirmed and both were
    reported as un-hashable user code.

    The second arm is the control, and it is what stops this test from passing
    by simply disabling the advisory: a partial over a USER function must still
    warn, because that function's body genuinely is absent from the key.
    """
    import json
    import weakref

    takes, _calls = _counting(c, name="advisory_takes")
    nb = _nb_module()
    _define(nb, "def user_scale(k, x): return k * x\n", name="user_scale")

    class Holder:
        pass

    holder = Holder()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        takes(functools.partial(json.dumps))
        takes(weakref.ref(holder))
    assert _code_advisories(rec) == [], "a stdlib callable object must not warn"

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        takes(functools.partial(nb.user_scale, 3))
    assert len(_code_advisories(rec)) == 1, "control: a partial over USER code must warn"
