"""A global whose attributes alone a function reads is keyed by those
chains (``console.log``), not whole. Python 3.12 loads a name an import
bound with a call's NULL ahead of it when an attribute of it is called
(``console.log(x)``), where 3.13 pushes the NULL after the chain; the
reader took that NULL for a call of the name itself, so on 3.12 such a
global was keyed whole. An object the key cannot hash (a console holding
a lock) was then walked for the code it holds, the thread-local it keeps
included, and the state memo, which watches such a global only for its
binding, served a segment a fresh fold no longer built.
"""

from __future__ import annotations

import pytest

from cash.decorator.global_reads import global_attr_chains, module_attr_chains

pytestmark = [
    pytest.mark.core,
]


def _code(source: str, name: str = "f"):
    ns: dict = {}
    exec(compile(source, "<test>", "exec"), ns)
    return ns[name].__code__


@pytest.mark.parametrize(
    ("source", "chains"),
    [
        ("from os import path\ndef f(x):\n    return path.join(x, 'a')\n", (("join",),)),
        ("from os import path\ndef f(x):\n    return path.join(x) + path.sep\n", (("join",), ("sep",))),
        ("path = None\ndef f(x):\n    return path.sep + path.join(x, 'a')\n", None),
        ("from os import path\ndef f(x):\n    return path(x)\n", None),
        ("from os import path\ndef f(x):\n    return path(x.join)\n", None),
    ],
    ids=["imported_called", "imported_called_and_read", "a_method_called", "name_called", "name_called_on_attr"],
)
def test_an_attribute_called_off_a_global_is_a_chain(source, chains):
    assert global_attr_chains((_code(source),), "path", True) == chains


@pytest.mark.parametrize(
    ("source", "chains"),
    [
        ("import os\ndef f(x):\n    return os.getcwd() + os.path.sep\n", (("sep",),)),
        ("import os\ndef f(x):\n    return os(x)\n", None),
    ],
    ids=["another_attribute_called", "module_called"],
)
def test_another_attribute_called_off_the_module_leaves_the_chain(source, chains):
    assert module_attr_chains((_code(source),), ("os", "path")) == chains
