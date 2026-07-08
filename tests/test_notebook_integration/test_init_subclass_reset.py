"""A subclass def that triggers a base ``__init_subclass__`` registry side effect
must reset on isolated re-run (CAS-103).

``class Sub(Base): pass`` runs ``Base.__init_subclass__(cls, ...)`` during CLASS
CREATION — before any statement in the cell body executes — so the mutation is
invisible to the object-protocol executable-node walk. When that hook appends the
subclass to a module list (``registry.append(cls.__name__)``), an isolated re-run
of the subclass cell re-registers and the registry doubles. The fix analyses the
base hook's body like a constructor and routes the free-var mutation to the
self-protecting CAS-68A reset channel.

Each subclass def registers ONCE, so the correct registry is identical on the
first run and every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


_BASE = (
    "class PluginBase:\n"
    "    def __init_subclass__(cls, **kw):\n"
    "        super().__init_subclass__(**kw)\n"
    "        registry.append(cls.__name__)"
)


def test_init_subclass_registry_rerun_idempotent(nb_runner):
    """Main repro: isolated re-run of the subclass cell then the print cell must
    not double-register."""
    nb_runner.create_notebook([
        "registry = []",
        _BASE,
        "class AlphaPlugin(PluginBase):\n    pass",
        "print('plugins=', registry)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "plugins= ['AlphaPlugin']" in nb_runner.get_output(4), (
        f"run_all: {nb_runner.get_output(4)!r}"
    )

    # Isolated re-run of the subclass cell, then the consumer.
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "plugins= ['AlphaPlugin']" in out, (
        f"__init_subclass__ registry accumulated on isolated re-run: {out!r}"
    )


def test_run_all_registers_exactly_once(nb_runner):
    """CONTROL (a): a plain run_all must yield exactly one entry — the reset must
    not over-fire and DROP the legitimate registration."""
    nb_runner.create_notebook([
        "registry = []",
        _BASE,
        "class AlphaPlugin(PluginBase):\n    pass",
        "print('plugins=', registry)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "plugins= ['AlphaPlugin']" in nb_runner.get_output(4), (
        f"run_all over-reset and dropped the entry: {nb_runner.get_output(4)!r}"
    )


def test_second_distinct_subclass_still_registers(nb_runner):
    """CONTROL (b): a SECOND distinct subclass in another cell still registers —
    run_all must yield both names."""
    nb_runner.create_notebook([
        "registry = []",
        _BASE,
        "class AlphaPlugin(PluginBase):\n    pass",
        "class BetaPlugin(PluginBase):\n    pass",
        "print('plugins=', registry)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "AlphaPlugin" in out and "BetaPlugin" in out, (
        f"a distinct second subclass failed to register: {out!r}"
    )


def test_cross_cell_class_var_accumulator_not_corrupted(nb_runner):
    """CONTROL (c): a legitimate cross-cell class-VAR accumulator (CAS-75 shape:
    ``Base.count`` bumped by an upstream cell AND this cell) must keep the plain
    run_all value — the class-def reset is suppressed by the cross-cell guard, so
    the mutation is not clobbered.

    Modeled on tests/test_notebook_integration/test_class_state_rerun.py
    ::test_init_increments_class_counter.
    """
    nb_runner.create_notebook([
        "class Widget:\n    count = 0\n    def __init__(self):\n        Widget.count += 1",
        "w0 = Widget()",
        "w = Widget()\nprint('count', Widget.count)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # Plain-kernel value after two constructions across two cells.
    assert "count 2" in nb_runner.get_output(3), (
        f"cross-cell class-var accumulator corrupted: {nb_runner.get_output(3)!r}"
    )
