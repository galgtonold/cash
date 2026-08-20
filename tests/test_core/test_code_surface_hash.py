import sys
import types
from cash.core import Cash


def test_a_fileless_module_counts_as_user_code():
    """A notebook cell's __main__ has no __file__. The existing
    _is_user_module rejects it, which would make this whole feature a no-op
    in the environment it exists for."""
    nb = types.ModuleType("nbmod")          # no __file__, like a notebook __main__
    assert Cash._is_user_code_module(nb) is True


def test_builtins_are_not_user_code_despite_being_fileless():
    """The control that stops 'fileless means user code' from sweeping in
    every builtin type."""
    assert Cash._is_user_code_module(sys.modules["builtins"]) is False


def test_stdlib_and_site_packages_are_still_excluded():
    import json
    assert Cash._is_user_code_module(sys.modules["json"]) is False
import hashlib
import itertools
import subprocess

from cash import Cash as CashCls

_notebook_module_counter = itertools.count()


def _exec_class(body: str, name: str = "S"):
    """Execute *body* the way a notebook cell actually does: into a
    REGISTERED, fileless module's ``__dict__`` -- exactly like a real
    kernel's ``user_ns``, which IS ``sys.modules['__main__']``, has no
    ``__file__``, and already has ``__name__ == '__main__'`` set before any
    cell ever runs. ``inspect.getsource`` cannot read source from the
    result, exactly as it can't for something defined in a real kernel --
    ``TypeError`` for a class (``inspect.getfile`` calls it "a built-in
    class" once its module has no ``__file__``), ``OSError`` for a plain
    function (resolved through ``co_filename`` instead, which fails at the
    file-read rather than the type-check). Both are in
    ``SOURCE_RETRIEVAL_ERRORS``, so which one fires never matters to the
    code under test -- only that source retrieval fails at all.

    A bare ``exec(body, {})`` does NOT reproduce this. The empty dict has no
    ``__name__``, so CPython's implicit ``__module__ = __name__`` lookup at
    definition time falls through to ``builtins``' own dunder for a class
    (inheriting the literal string ``'builtins'``, despite having nothing to
    do with it) and lands on ``None`` for a function (no such fallback
    exists for functions). Neither is registered in ``sys.modules`` either
    way, so ``_is_user_code_object``'s ``sys.modules.get(obj.__module__)``
    step fails regardless of what ``__module__`` says. That is a real,
    Python-version-independent CPython quirk (confirmed identical on
    3.10/3.11/3.14) -- but it is a bare-dict-``exec`` artifact, not the
    notebook path these tests are named for; a real kernel's globals were
    never a bare ``{}``. A previous version of this helper used one, so
    these tests were passing for the wrong reason.

    Works identically for a class body or a ``def`` body -- nothing below is
    class-specific, despite the name.
    """
    mod = types.ModuleType(f"_cash_test_notebook_{next(_notebook_module_counter)}")
    sys.modules[mod.__name__] = mod
    exec(body, mod.__dict__)
    return getattr(mod, name)


def _code_surface_hash_in_subprocess(class_body: str) -> str:
    """Run *class_body* (defining a class named ``S``) in a FRESH interpreter
    process and return ``c._code_surface_hash(S)`` as hex text.

    Must be a genuinely separate process, not a loop in this one: a memory
    address leaked into the digest is invisible within a single process --
    nothing forces two objects allocated moments apart on ONE interpreter's
    heap to collide the way two SEPARATE interpreter invocations reliably
    expose (ASLR plus a freshly initialized allocator).
    """
    script = "from cash import Cash\n" + class_body + "\nc = Cash()\nprint(c._code_surface_hash(S))\n"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


def test_two_different_source_less_classes_do_not_collide():
    c = CashCls()
    a = _exec_class("class S:\n    def r(self): return 'AAA'\n")
    b = _exec_class("class S:\n    def r(self): return 'BBB'\n")
    assert c._code_surface_hash(a) != c._code_surface_hash(b)


def test_editing_a_method_body_changes_the_hash():
    c = CashCls()
    v1 = _exec_class("class S:\n    def r(self): return 'V1'\n")
    v2 = _exec_class("class S:\n    def r(self): return 'V2'\n")
    assert c._code_surface_hash(v1) != c._code_surface_hash(v2)


def test_an_identical_redefinition_hashes_the_same():
    """The control. A hash that changes every time is not a fix -- it just
    never caches."""
    c = CashCls()
    body = "class S:\n    def r(self): return 'V1'\n"
    assert c._code_surface_hash(_exec_class(body)) == c._code_surface_hash(_exec_class(body))


def test_a_class_level_data_attribute_participates():
    c = CashCls()
    a = _exec_class("class S:\n    TOP_K = 7\n    def r(self): return 1\n")
    b = _exec_class("class S:\n    TOP_K = 9\n    def r(self): return 1\n")
    assert c._code_surface_hash(a) != c._code_surface_hash(b)


def test_a_plain_function_is_hashed_by_its_code():
    c = CashCls()
    f1 = _exec_class("def f(): return 'V1'", name="f")
    f2 = _exec_class("def f(): return 'V2'", name="f")
    assert c._code_surface_hash(f1) != c._code_surface_hash(f2)


def test_a_base_class_body_participates():
    c = CashCls()
    a = _exec_class("class B:\n    def helper(self): return 'V1'\nclass S(B):\n    pass\n")
    b = _exec_class("class B:\n    def helper(self): return 'V2'\nclass S(B):\n    pass\n")
    assert c._code_surface_hash(a) != c._code_surface_hash(b)


def test_a_third_party_class_returns_none():
    """Library code stays by-reference: folding thousands of methods into every
    key would churn on every upgrade for no correctness gain."""
    import json.encoder
    c = CashCls()
    assert c._code_surface_hash(json.encoder.JSONEncoder) is None


def test_it_never_raises_on_an_exotic_object():
    c = CashCls()
    assert c._code_surface_hash(object()) is None
    assert c._code_surface_hash(len) is None          # builtin, no __code__
    # object() and len are both HASHABLE, so neither exercises the memo-read
    # path below -- they do not discriminate the Blocker-4 bug. A list, a
    # dict, and a set are unhashable, which is the common case for a cache
    # ARGUMENT (Task 4's consumption path): dict.get() raises TypeError on
    # them, and that lookup used to sit outside the try/except meant to catch
    # exactly this.
    assert c._code_surface_hash([1, 2, 3]) is None
    assert c._code_surface_hash({"a": 1}) is None
    assert c._code_surface_hash({1, 2, 3}) is None


def test_source_less_classes_no_longer_collide_in_the_instance_channel():
    """Regression: _user_class_source_hash returned sha256("type") for every
    class whose source could not be read, so two unrelated classes were
    indistinguishable in the instance channel."""
    c = CashCls()
    a = _exec_class("class S:\n    def r(self): return 'AAA'\n")
    b = _exec_class("class S:\n    def r(self): return 'BBB'\n")
    assert c._user_class_source_hash(a) != c._user_class_source_hash(b)


def test_a_comprehension_in_a_method_hashes_identically_across_processes():
    """Blocker 1 regression: a nested code object (a lambda, a generator
    expression, or -- pre-3.12, before PEP 709 inlined them -- even a plain
    comprehension) went into the digest via repr(), which embeds a live
    memory address (``<code object <genexpr> at 0x...>``). That is a
    permanent cross-process cache miss for any user class with one of these
    in a method -- invisible to an in-process test, since nothing forces two
    objects built moments apart on ONE interpreter's heap to collide the way
    two SEPARATE processes reliably do."""
    body = (
        "class S:\n"
        "    def transform(self, xs):\n"
        "        return sum(x * 2 for x in xs)\n"
    )
    h1 = _code_surface_hash_in_subprocess(body)
    h2 = _code_surface_hash_in_subprocess(body)
    assert h1 != "None"  # positive control: it must actually have hashed something
    assert h1 == h2


def test_a_sentinel_data_attribute_hashes_identically_across_processes():
    """Blocker 2 regression: a non-callable class attribute went into the
    digest via repr(), which embeds a live memory address for anything like a
    ``MISSING = object()`` sentinel -- the same disease as Blocker 1, one
    layer up (a class-level constant instead of a nested code object)."""
    body = (
        "class S:\n"
        "    MISSING = object()\n"
        "    def r(self): return 1\n"
    )
    h1 = _code_surface_hash_in_subprocess(body)
    h2 = _code_surface_hash_in_subprocess(body)
    assert h1 != "None"
    assert h1 == h2


def test_a_functools_wraps_decorated_method_body_edit_invalidates():
    """Blocker 3 regression: _class_surface_parts walked the WRAPPER function
    functools.wraps leaves behind, not the wrapped one -- so it hashed the
    wrapper's own fixed `return func(*args, **kwargs)` dispatch code, which
    does not change no matter what the wrapped body says. Needs source
    retrieval to fail to even reach _class_surface_parts, hence _exec_class
    rather than a real file-backed class."""
    body_template = (
        "import functools\n"
        "def logged(func):\n"
        "    @functools.wraps(func)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return func(*args, **kwargs)\n"
        "    return wrapper\n"
        "class S:\n"
        "    @logged\n"
        "    def m(self):\n"
        "        return {value!r}\n"
    )
    c = CashCls()
    v1 = _exec_class(body_template.format(value="V1"))
    v2 = _exec_class(body_template.format(value="V2"))
    assert c._code_surface_hash(v1) != c._code_surface_hash(v2)


def test_a_partialmethod_bound_argument_change_invalidates():
    """Fold-both regression: the __wrapped__/.func unwrap that fixed
    functools.singledispatchmethod (previous round) let a successful `ident`
    fully REPLACE the content fold for any non-callable member -- so a
    functools.partialmethod's bound arguments, which live on the descriptor
    itself rather than on the function .func resolves to, went invisible.
    Both classes are exec'd into ONE module, because _exec_class builds a new
    module per call and the content fold pickles the descriptor by reference --
    embedding that module's name. Two separate modules therefore produce
    different digests whether or not the bug exists, which would make this
    assertion pass for the wrong reason. The equal-argument control below is
    what proves the bound value is carrying it."""
    def _two_in_one_module(arg_a, arg_b):
        """Both classes in ONE registered fileless module, same qualname, sharing
        ONE ``base`` function object.

        ``base`` is defined once and never re-executed. Re-running a body that
        also defines it rebinds the module's ``base`` to a new object, which
        leaves the first class's descriptor holding an orphan -- pickle then
        raises ``it's not the same object as <module>.base`` for one class and
        not the other, and that difference, not the bound value, decides the
        assertion.
        """
        mod = types.ModuleType(f"_cash_pm_probe_{next(_notebook_module_counter)}")
        sys.modules[mod.__name__] = mod
        exec("import functools\ndef base(self, y):\n    return y\n", mod.__dict__)
        cls_body = "class S:\n    pm = functools.partialmethod(base, {arg})\n"
        exec(cls_body.format(arg=arg_a), mod.__dict__)
        first = mod.S
        exec(cls_body.format(arg=arg_b), mod.__dict__)
        return first, mod.S

    c = CashCls()
    same_a, same_b = _two_in_one_module(3, 3)
    assert c._code_surface_hash(same_a) == c._code_surface_hash(same_b), (
        "CONTROL: identical bound arguments in the same module must collide. "
        "If this fails, something other than the bound value is carrying the "
        "assertion below and the test proves nothing."
    )

    diff_a, diff_b = _two_in_one_module(3, 4)
    assert c._code_surface_hash(diff_a) != c._code_surface_hash(diff_b)


def test_a_frozenset_literal_in_a_method_hashes_identically_across_processes():
    """Blocker-1-class regression: `x in {...}` compiles a frozenset straight
    into co_consts, and repr() of a set/frozenset follows hash-table
    iteration order -- randomized per process by Python's default string-hash
    randomization. Six elements, not two: a 2-element set has only 2 possible
    orderings, so two independent processes coincidentally agree about half
    the time (measured) even with the bug present, which would make the
    mutation check below unreliable. Six elements diverged in 10 of 10
    sampled fresh processes during development of this test."""
    body = (
        "class S:\n"
        "    def m(self, x):\n"
        "        return x in {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'}\n"
    )
    h1 = _code_surface_hash_in_subprocess(body)
    h2 = _code_surface_hash_in_subprocess(body)
    assert h1 != "None"
    assert h1 == h2


def test_a_default_sentinel_hashes_identically_across_processes():
    """Blocker-1-class regression: the common `def m(self, x=_MISSING)`
    sentinel-default idiom reprs `__defaults__` with the sentinel's own
    memory address (`<object object at 0x...>`), the same disease as
    Blocker 1's nested code objects, one layer up (a function default
    instead of a comprehension)."""
    body = (
        "class S:\n"
        "    def m(self, x=object()):\n"
        "        return x\n"
    )
    h1 = _code_surface_hash_in_subprocess(body)
    h2 = _code_surface_hash_in_subprocess(body)
    assert h1 != "None"
    assert h1 == h2
