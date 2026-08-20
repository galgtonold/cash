"""Code reached from other code must fold into the cache key.

Two gaps, both measured as wrong ANSWERS rather than wasted recomputes:

1. A dataclass ``field(default_factory=...)`` callable was invisible.
   ``dataclasses`` deletes the class attribute when a factory is used, so
   ``getattr_static`` raises and the class-surface walk never saw it.
2. Code reached only THROUGH folded code was invisible. ``A``'s factory
   constructs ``B``, so editing ``B`` changes what ``A()`` produces -- but
   ``B`` appeared nowhere in ``A``'s own surface.

The controls matter as much as the assertions here: following references
too eagerly (annotations, non-user code) trades a stale-answer bug for an
invalidate-always one, which is just as unusable.
"""

import textwrap
import warnings

import pytest

from cash import Cash


@pytest.fixture()
def cash():
    return Cash(register_magic=False)


def build(src, name="A"):
    """Compile a module body under a user-code filename; return one symbol."""
    ns = {}
    exec(compile(textwrap.dedent(src), "usermod_probe.py", "exec"), ns)
    return ns[name]


BASE = """
    from dataclasses import dataclass, field

    class B:
        def __init__(self, v):
            self.value = v {b_expr}

    @dataclass
    class A:
        value: B = field(default_factory=lambda: B({factory_arg}))
        variable: str = "{variable}"
    """


def surface(cash, **over):
    params = dict(b_expr="", factory_arg="0", variable="t1")
    params.update(over)
    return cash._code_surface_hash(build(BASE.format(**params)))


# --- the null control --------------------------------------------------

def test_identical_source_gives_identical_surface(cash):
    """Without this, every assertion below could pass for the wrong reason."""
    assert surface(cash) == surface(cash)


# --- gap 1: the field default_factory itself ---------------------------

def test_default_factory_body_change_is_seen(cash):
    """`A()` genuinely produces a different value; the digest must move."""
    a1 = build(BASE.format(b_expr="", factory_arg="0", variable="t1"))
    a2 = build(BASE.format(b_expr="", factory_arg="999", variable="t1"))
    assert a1().value.value != a2().value.value, "premise: behaviour differs"
    assert cash._code_surface_hash(a1) != cash._code_surface_hash(a2)


def test_plain_field_default_change_is_seen(cash):
    """Already worked; pinned so the fix does not regress it."""
    assert surface(cash, variable="t1") != surface(cash, variable="t2")


# --- gap 2: code reached THROUGH folded code ---------------------------

def test_class_reached_via_default_factory_is_seen(cash):
    """The reported case: editing B changes what A() yields."""
    a1 = build(BASE.format(b_expr="", factory_arg="10", variable="t1"))
    a2 = build(BASE.format(b_expr="* 100", factory_arg="10", variable="t1"))
    assert a1().value.value != a2().value.value, "premise: behaviour differs"
    assert cash._code_surface_hash(a1) != cash._code_surface_hash(a2)


def test_class_referenced_from_a_method_body_is_seen(cash):
    """The general shape, not just dataclass fields."""
    src = """
        class Helper:
            def scale(self, v):
                return v {expr}

        class Owner:
            def run(self, v):
                return Helper().scale(v)
        """
    o1 = build(src.format(expr="* 2"), name="Owner")
    o2 = build(src.format(expr="* 3"), name="Owner")
    assert cash._code_surface_hash(o1) != cash._code_surface_hash(o2)


# --- controls against over-invalidation --------------------------------

def test_annotation_only_reference_is_not_followed(cash):
    """A type hint has no runtime effect, so editing it must NOT invalidate.

    ``__annotations__`` holds the NAME, so a reference-follower that reads
    annotations would invalidate on a change that cannot alter any result.
    """
    src = """
        class Unused:
            def method(self):
                return {v}

        class Owner:
            def run(self, x: "Unused") -> "Unused":
                return x
        """
    o1 = build(src.format(v="1"), name="Owner")
    o2 = build(src.format(v="2"), name="Owner")
    assert cash._code_surface_hash(o1) == cash._code_surface_hash(o2)


def test_stdlib_reference_is_not_followed(cash):
    """Only USER code is followed; otherwise every class drags in the stdlib."""
    src = """
        import json

        class Owner:
            def run(self, x):
                return json.dumps(x) {expr}
        """
    o1 = build(src.format(expr=""), name="Owner")
    assert cash._code_surface_hash(o1) is not None


# --- traversal must terminate ------------------------------------------

def test_mutual_reference_terminates(cash):
    """A <-> B referencing each other must not loop forever."""
    src = """
        class A:
            def make(self):
                return B() {expr}

        class B:
            def make(self):
                return A()
        """
    a1 = build(src.format(expr=""))
    a2 = build(src.format(expr="or 1"))
    assert cash._code_surface_hash(a1) is not None
    assert cash._code_surface_hash(a1) != cash._code_surface_hash(a2)


def test_self_reference_terminates(cash):
    src = """
        class A:
            def clone(self):
                return A()
        """
    assert cash._code_surface_hash(build(src)) is not None


# --- end to end, through a real cached call ----------------------------
#
# The unit tests above exercise ``_code_surface_hash``, which is the
# ARGUMENT path. A class used in the function BODY is folded by a
# different mechanism -- the purity analyzer's helper walk -- and an
# earlier attempt at this fix passed every test above while leaving the
# reported bug completely unchanged. These tests walk the reported path.

import importlib
import importlib.util
import linecache
import shutil
import sys

MODULE = '''
from dataclasses import dataclass, field

class B:
    def __init__(self, v):
        self.value = v {b_expr}
    def __repr__(self):
        # Value-based on purpose: the default repr prints a memory ADDRESS,
        # which differs between a restored object and a fresh one no matter
        # what, so it cannot tell a correct cache from a stale one.
        return "B(value=%r)" % (self.value,)

@dataclass
class A:
    value: B = field(default_factory=lambda: B(10))

@cash_instance.cache
def compute(n):
    return A()
'''


def _load_module(work, cash_instance, b_expr):
    (work / "usermod.py").write_text(MODULE.format(b_expr=b_expr))
    shutil.rmtree(work / "__pycache__", ignore_errors=True)
    sys.modules.pop("usermod", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    spec = importlib.util.spec_from_file_location("usermod", str(work / "usermod.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["usermod"] = mod
    mod.__dict__["cash_instance"] = cash_instance
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def work(tmp_path, monkeypatch):
    d = tmp_path / "work"
    d.mkdir()
    monkeypatch.syspath_prepend(str(d))
    yield d
    sys.modules.pop("usermod", None)


def _run(work, cash_instance, b_expr):
    mod = _load_module(work, cash_instance, b_expr)
    before = mod.compute.cache_info()["hits"]
    result = mod.compute(1)
    hit = mod.compute.cache_info()["hits"] > before
    return hit, repr(result), repr(mod.A())


def test_end_to_end_unchanged_still_hits(tmp_path, work):
    """Null control: the arms below prove nothing without it."""
    cash_instance = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    hit, _, _ = _run(work, cash_instance, "")
    assert hit is False, "first call must compute"
    hit, _, _ = _run(work, cash_instance, "")
    assert hit is True, "an unchanged program must still reuse its cache"


def test_end_to_end_editing_reached_class_recomputes(tmp_path, work):
    """The reported bug: B is reached only through A's field factory.

    Asserts against the UNCACHED truth rather than a hardcoded value, so
    the test states the property -- the cache must not disagree with a
    fresh computation -- rather than a snapshot of today's output.
    """
    cash_instance = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    _run(work, cash_instance, "")
    hit, cached, truth = _run(work, cash_instance, "* 100")
    assert cached == truth, f"cache served {cached}, a fresh call gives {truth}"
    assert hit is False


def test_end_to_end_does_not_warn_about_the_reached_class(tmp_path, work):
    """Following B must not report its ``self.value = v`` as impurity.

    A class is followed for the cache key, not audited. Reporting every
    ordinary __init__ assignment as a scope mutation would bury the real
    findings under noise.
    """
    cash_instance = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        _run(work, cash_instance, "")
    impurity = [w for w in log if "scope_mutation" in str(w.message)
                or "side effects" in str(w.message)]
    assert not impurity, f"unexpected impurity warning: {[str(w.message) for w in impurity]}"
