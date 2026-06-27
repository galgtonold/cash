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

    @pytest.mark.xfail(reason="method-call mutation does not bump the receiver's lineage; "
                              "cached consumer serves stale value. Both attempted fixes "
                              "(cache receiver / runtime content-hash bump) hit the "
                              "runtime-vs-simulation lineage desync -- see module docstring.",
                       strict=False)
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

    @pytest.mark.xfail(reason="builtin method-call mutation (list.append on an attribute) "
                              "does not bump the receiver's lineage; cached consumer stale "
                              "(same runtime-vs-simulation desync -- see module docstring)",
                       strict=False)
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
