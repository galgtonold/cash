"""A class a library builds at run time with no code of its own is library
code.

pint builds ``type("Quantity", (registry.Quantity,), {"_REGISTRY": ureg})``
per registry, naming the module ``pint`` where that name finds another
class. Not found by its name, it fell into the fallback for exec'd and
notebook code: its data, the registry, was folded as the user's class data
on every call, and the weak caches inside made the key unbuildable. Its
code is its bases'; a function of its own, or a base of the user's, keeps
the user-code verdict.
"""

from __future__ import annotations

import json

from cash.decorator.user_code import is_user_code_object


class _UserBase:
    def run(self):
        return 1


def _built(bases, namespace):
    return type("Thing", bases, {"__module__": "json", **namespace})


def test_a_data_only_class_named_after_a_library_module_is_not_user_code():
    assert not hasattr(json, "Thing")
    assert is_user_code_object(_built((json.JSONDecoder,), {"_REGISTRY": object()})) is False


def test_control_a_function_of_its_own_is_user_code():
    cls = _built((json.JSONDecoder,), {"_REGISTRY": object(), "run": lambda self: 2})
    assert is_user_code_object(cls) is True


def test_control_a_base_of_the_user_s_is_user_code():
    assert is_user_code_object(_built((_UserBase,), {"_REGISTRY": object()})) is True


def test_control_a_class_named_after_a_fileless_module_is_user_code():
    cls = type("Thing", (json.JSONDecoder,), {"__module__": "builtins", "_REGISTRY": object()})
    assert is_user_code_object(cls) is True
