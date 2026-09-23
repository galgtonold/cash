"""Rewording a docstring in a cell does not re-run what calls the function.

Comments were already free, because a statement is keyed on its
``ast.unparse`` text and that drops them. A docstring survived it: the
``def`` statement's text moved, so did the lineage of the name it binds, and
every statement calling the function missed. Now both halves of the key --
the statement's own text and the function source it calls -- are digested
without docstrings (``cache_key.statement_source_hash``).

A bare string statement in a cell is NOT a docstring: it is the cell's
displayed value, and editing it must still count.
"""

from unittest.mock import MagicMock

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cache_key import statement_source_hash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics

# Force caching regardless of the 10 ms min-execution-time floor.
_PERSIST = CacheAnnotation(persist=True)


class MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    processor = magics._statement_processor
    yield magics, shell, backend, processor
    backend.clear()
    shell.user_ns.clear()


DOUBLE = 'def double(n):\n    """Double n."""\n    return n * 2'
REWORDED = 'def double(n):\n    """Return twice n.\n\n    Longer now."""\n    return n * 2'
UNDOCUMENTED = "def double(n):\n    return n * 2"
TRIPLE = 'def double(n):\n    """Double n."""\n    return n * 3'

CLASS = 'class Box:\n    """A box."""\n\n    def size(self):\n        """How big."""\n        return 3'
CLASS_REWORDED = 'class Box:\n    """A crate."""\n\n    def size(self):\n        """Its size."""\n        return 3'


def _run_call(processor, definition, call="y = double(21)"):
    processor.process_statement(definition)
    return processor.process_statement(call, annotation=_PERSIST)


class TestTheCallerIsServedFromCache:
    def test_after_rewording_the_docstring(self, magics_fixture):
        _, shell, _, processor = magics_fixture
        assert _run_call(processor, DOUBLE)["status"] == CacheStatus.COMPUTED
        metrics = _run_call(processor, REWORDED)
        assert metrics["status"] == CacheStatus.RESTORED, (
            "only the docstring of `double` changed, and its caller re-ran"
        )
        assert shell.user_ns["y"] == 42

    def test_after_removing_the_docstring(self, magics_fixture):
        _, _, _, processor = magics_fixture
        _run_call(processor, DOUBLE)
        assert _run_call(processor, UNDOCUMENTED)["status"] == CacheStatus.RESTORED

    def test_after_rewording_a_class_and_its_methods(self, magics_fixture):
        _, shell, _, processor = magics_fixture
        call = "z = Box().size()"
        assert _run_call(processor, CLASS, call)["status"] == CacheStatus.COMPUTED
        assert _run_call(processor, CLASS_REWORDED, call)["status"] == CacheStatus.RESTORED
        assert shell.user_ns["z"] == 3


def test_a_real_edit_still_re_runs_the_caller(magics_fixture):
    """The control. Without it the tests above could pass on a cache that
    never invalidates at all."""
    _, shell, _, processor = magics_fixture
    _run_call(processor, DOUBLE)
    assert _run_call(processor, TRIPLE)["status"] == CacheStatus.COMPUTED
    assert shell.user_ns["y"] == 63


class TestTheStatementDigest:
    def test_ignores_a_def_s_docstring(self):
        assert statement_source_hash(DOUBLE) == statement_source_hash(REWORDED)
        assert statement_source_hash(DOUBLE) == statement_source_hash(UNDOCUMENTED)

    def test_still_sees_the_body(self):
        assert statement_source_hash(DOUBLE) != statement_source_hash(TRIPLE)

    def test_a_bare_string_statement_is_a_value_not_a_docstring(self):
        """The last line of a cell is displayed; changing it changes output."""
        assert statement_source_hash("'shown'") != statement_source_hash("'other'")

    def test_leaves_a_statement_without_one_byte_identical(self):
        """Keys that never had a docstring in them must not move."""
        import hashlib

        for code in ("x = 1", "y = f('a')", UNDOCUMENTED):
            assert statement_source_hash(code) == hashlib.sha256(code.encode()).hexdigest()
