"""Test isolation for the overhead-benchmark suite.

The benchmark drivers (``benchmarks._overhead_driver.run_notebook`` and
``benchmarks.eval_cost_model``) call ``InteractiveShell.instance()`` to get
an in-process shell. That registers a *process-global* singleton, so
``IPython.get_ipython()`` keeps returning a live shell for the rest of the
worker's lifetime — even after these tests finish.

Downstream tests that assume no active IPython session then break when xdist
schedules them onto the same worker (``--dist loadscope`` keeps a module's
tests together but still runs many modules per worker). Concretely:

* ``cash.reset_session()`` re-creates the global singleton when a shell is
  active, so ``test_reset_session`` sees ``_global_cash`` is not ``None``.
* ``Cash()`` auto-registers magics when a shell is active, so
  ``test_ergonomics::test_no_magic_registered_outside_ipython_by_default``
  sees ``register_magic`` get called.

Clearing the singleton after each test in this package undoes the leak at its
source without touching the drivers' intended within-run singleton reuse.
"""
import pytest
from IPython.core.interactiveshell import InteractiveShell


@pytest.fixture(autouse=True)
def _clear_ipython_singleton():
    yield
    InteractiveShell.clear_instance()
