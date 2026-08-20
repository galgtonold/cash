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

from cash import Cash as CashCls


def _exec_class(body: str, name: str = "S"):
    """A class with no retrievable source -- the notebook case, reproducible
    in-process. inspect.getsource raises OSError on these, exactly as it does
    for a class defined in a real kernel."""
    ns: dict = {}
    exec(body, ns)
    return ns[name]


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
    ns1: dict = {}; exec("def f(): return 'V1'", ns1)
    ns2: dict = {}; exec("def f(): return 'V2'", ns2)
    assert c._code_surface_hash(ns1["f"]) != c._code_surface_hash(ns2["f"])


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


def test_source_less_classes_no_longer_collide_in_the_instance_channel():
    """Regression: _user_class_source_hash returned sha256("type") for every
    class whose source could not be read, so two unrelated classes were
    indistinguishable in the instance channel."""
    c = CashCls()
    a = _exec_class("class S:\n    def r(self): return 'AAA'\n")
    b = _exec_class("class S:\n    def r(self): return 'BBB'\n")
    assert c._user_class_source_hash(a) != c._user_class_source_hash(b)
