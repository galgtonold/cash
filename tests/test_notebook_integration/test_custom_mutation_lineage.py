"""A standalone method-call mutation bumps the receiver's lineage (fixed).

When a statement mutates an object via a top-level bare-``Expr`` METHOD call
(custom ``box.add(x)`` -- ``add`` is a known set method -- or builtin
``box.items.append(x)``) the receiver carries no Store target, so the AST flow
analysis never surfaces it as an *output* and its lineage was historically left
frozen. A cached downstream consumer (``result = box.total()``) then served a
STALE value after the mutation was edited. (In normal use the consumer is
usually sub-cost-floor and recomputes, so the bug only bit when the consumer was
genuinely cached -- ``%cash_persist`` here forces that.)

THE FIX (``cacheability.standalone_method_mutation_receivers``): the receiver of
a top-level bare-``Expr`` mutating method call is routed into the statement's
*output* set in BOTH the runtime (``StatementProcessor.process_statement``) and
the upstream simulation (``VirtualLineage._update_virtual_lineage``). Both engines
then compute the receiver's lineage with the SAME source-based formula
``f(source_hash, sorted(input_lineages), ...)`` -- so they stay in sync, unlike
the reverted runtime-only content-hash bump that desynced cross-cell restore. The
statement is additionally skip-cached, so the mutated receiver is never
round-tripped (caching arbitrary mutated objects corrupts stateful / loop /
multi-mutation receivers such as a ``dataclass`` with ``field(default_factory=
list)`` or a loop-written ``StringIO``); it re-executes each run and re-applies
the mutation.

Scope is narrow on purpose -- only methods known to mutate (``MUTATING_METHODS``)
or pandas ``inplace=True``. Custom mutators whose names are not known-mutating
(observer ``bus.on``, stack ``s.push``) are NOT yet covered; see
test_interaction_observer_pattern / test_interaction_iterator_protocol for that
remaining family (currently sub-cost-floor, so they pass for a different reason).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(90)]

_BOX = (
    "class Box:\n"
    "    def __init__(self):\n        self.items = []\n"
    "    def add(self, x):\n        self.items.append(x)\n"
    "    def total(self):\n        return sum(self.items)"
)


class TestCustomMutationLineage:

    def test_custom_method_mutation_invalidates_cached_consumer(self, nb_runner):
        nb_runner.create_notebook([
            _BOX,
            "box = Box()\nbox.add(10)\nresult = box.total()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "box = Box()\nbox.add(10)\nbox.add(20)\nresult = box.total()\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2), \
            f"stale cached consumer: {nb_runner.get_output(2)!r}"

    def test_builtin_method_mutation_invalidates_cached_consumer(self, nb_runner):
        nb_runner.create_notebook([
            _BOX,
            "box = Box()\nbox.items.append(10)\nresult = box.total()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "box = Box()\nbox.items.append(10)\nbox.items.append(20)\nresult = box.total()\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2), \
            f"stale cached consumer: {nb_runner.get_output(2)!r}"


_STACK = (
    "class Stack:\n"
    "    def __init__(self):\n        self._items = []\n"
    "    def push(self, x):\n        self._items.append(x)\n"
    "    def total(self):\n        return sum(self._items)"
)

_BUS_CLS = (
    "class Bus:\n"
    "    def __init__(self):\n        self.handlers = []\n"
    "    def on(self, fn):\n        self.handlers.append(fn)\n"
    "    def emit(self, x):\n        return [h(x) for h in self.handlers]"
)


class TestBroadCustomMutators:
    """The broad-precise extension: a custom-named mutator (``stack.push``,
    ``bus.on``) -- not in MUTATING_METHODS -- should bump the receiver's lineage,
    detected by runtime content-observation. Picklable receivers are observed
    precisely; unpicklable ones (lambda / builtin handlers) fall to the
    conservative assume-mutate path.
    """

    def test_picklable_custom_mutator_invalidates_consumer(self, nb_runner):
        """stack.push -- custom name, picklable receiver -> observed precisely."""
        nb_runner.create_notebook([
            _STACK,
            "s = Stack()\ns.push(1)\nresult = s.total()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "result=1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "s = Stack()\ns.push(1)\ns.push(2)\nresult = s.total()\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2), \
            f"stale cached consumer: {nb_runner.get_output(2)!r}"

    def test_unpicklable_custom_mutator_invalidates_consumer(self, nb_runner):
        """bus.on(callable) -- custom name, unpicklable receiver -> assume-mutate."""
        nb_runner.create_notebook([
            _BUS_CLS,
            "bus = Bus()\nbus.on(str.upper)\nresult = bus.emit('hi')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert "['HI']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "bus = Bus()\nbus.on(str.upper)\nbus.on(lambda s: s[::-1])\nresult = bus.emit('hi')\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "HI" in out and "ih" in out, f"stale cached consumer: {out!r}"
