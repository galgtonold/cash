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
