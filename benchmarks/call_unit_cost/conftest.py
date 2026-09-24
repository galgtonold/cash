"""Measurements behind the call-unit cost floor in ``cash/notebook/call_unit.py``.

The constants there are fitted from these scripts, so re-run them before
changing one. Each drives real kernels through the notebook-integration
``nb_runner`` fixture and prints its numbers as JSON; none of them asserts.
pytest only collects them when named explicitly, one file at a time:

    pytest benchmarks/call_unit_cost/measure_store_cost.py -n 0 -s --timeout=0

- ``measure_store_cost.py``: what one call-unit store and hit cost.
- ``measure_break_even.py``: the body size where caching a call starts to pay.
- ``measure_sub_floor_band.py``: warm-rerun cost of bodies below the floor.
- ``measure_steady_state.py``: the same band once the verdict is learned.
- ``measure_high_iteration_count.py``: whether overhead scales with n.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.test_notebook_integration.conftest import *  # noqa: F403 - re-exports the integration fixtures
