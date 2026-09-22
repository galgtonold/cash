"""A docstring is documentation, not code: editing one keeps the cache.

``test_source_normalization.py`` covers the source channel, where the fix
lives in ``normalize_source_for_hash``. This file covers the channels that
never see source text:

* ``bytecode_identity`` -- the fallback when source cannot be read. A
  function's docstring is its first constant, so it moved with every edit.
* The class surface -- what a class passed as an argument, or reached from a
  cached function, is keyed on. Method docstrings arrive through the same
  constant, and the class's own through its ``__doc__`` attribute.

One exception is pinned at the end: a pydantic model's docstring is its
schema's ``description``, which structured-output libraries send to the
model as part of the prompt, so there it still counts.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from cash import Cash
from cash.source_norm import bytecode_identity, code_consts_without_docstring


def _define(source: str, name: str):
    """Compile *source* with no file behind it, so only bytecode is left."""
    namespace: dict = {"__name__": "docmod"}
    exec(compile(textwrap.dedent(source), "<no source>", "exec"), namespace)
    return namespace[name]


ONE = '''
    def f(n):
        """Double n."""
        return n * 2
'''
REWORDED = ONE.replace("Double n.", "Return twice n, at greater length.")
UNDOCUMENTED = '''
    def f(n):
        return n * 2
'''
CHANGED = ONE.replace("n * 2", "n * 3")


class TestBytecodeIdentity:

    def test_a_reworded_docstring_keeps_it(self):
        assert bytecode_identity(_define(ONE, "f")) == bytecode_identity(_define(REWORDED, "f"))

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason=(
        "3.14 stores a docstring only when there is one and shifts every "
        "other constant's index to fit, which co_code shows"))
    def test_adding_one_keeps_it(self):
        assert bytecode_identity(_define(ONE, "f")) == bytecode_identity(_define(UNDOCUMENTED, "f"))

    def test_a_real_edit_still_moves_it(self):
        assert bytecode_identity(_define(ONE, "f")) != bytecode_identity(_define(CHANGED, "f"))

    def test_a_returned_string_is_not_a_docstring(self):
        a = _define('def f():\n    return "one"\n', "f")
        b = _define('def f():\n    return "two"\n', "f")
        assert bytecode_identity(a) != bytecode_identity(b)

    def test_a_lambda_returning_a_string_keeps_its_string(self):
        """A lambda has no docstring, so its first constant is never masked."""
        a = _define('f = lambda: "one"\n', "f")
        b = _define('f = lambda: "two"\n', "f")
        assert bytecode_identity(a) != bytecode_identity(b)

    def test_a_nested_function_s_docstring_is_masked_too(self):
        outer = '''
            def outer():
                def inner():
                    """One."""
                    return 1
                return inner
        '''
        assert (bytecode_identity(_define(outer, "outer"))
                == bytecode_identity(_define(outer.replace("One.", "Two."), "outer")))

    def test_the_mask_only_touches_the_docstring(self):
        code = _define(ONE, "f").__code__
        masked = code_consts_without_docstring(code)
        assert masked[0] is None
        assert masked[1:] == code.co_consts[1:]


CLASS = '''
    class Box:
        """A box."""

        def size(self):
            """How big it is."""
            return 3
'''


@pytest.fixture()
def cash_instance(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


class TestClassSurface:

    def test_rewording_class_and_method_docstrings_keeps_it(self, cash_instance):
        a = _define(CLASS, "Box")
        b = _define(CLASS.replace("A box.", "A crate.").replace("How big it is.", "Its size."), "Box")
        assert cash_instance._code_surface_hash(a) == cash_instance._code_surface_hash(b)

    def test_a_method_edit_still_moves_it(self, cash_instance):
        a = _define(CLASS, "Box")
        b = _define(CLASS.replace("return 3", "return 4"), "Box")
        assert cash_instance._code_surface_hash(a) != cash_instance._code_surface_hash(b)

    def test_a_pydantic_model_s_docstring_still_counts(self, cash_instance):
        """It is the schema's description: the prompt, not commentary."""
        pytest.importorskip("pydantic")
        model = '''
            from pydantic import BaseModel

            class Invoice(BaseModel):
                """Extract the invoice total."""
                total: float
        '''
        a = _define(model, "Invoice")
        b = _define(model.replace("invoice total", "invoice subtotal"), "Invoice")
        assert a.model_json_schema()["description"] != b.model_json_schema()["description"]
        assert cash_instance._code_surface_hash(a) != cash_instance._code_surface_hash(b)
