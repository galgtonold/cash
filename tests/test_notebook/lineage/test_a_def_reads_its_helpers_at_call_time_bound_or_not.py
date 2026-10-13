"""What a ``def`` leaves out of its function's identity does not hang on
whether the helpers its body calls are bound yet.

After a restart the simulation keys ``def load(): _ran('load')`` before
``def _ran`` has run again. The run before had ``_ran`` bound, and a
callable named with a leading ``_`` was taken for a module there, so the two
engines gave ``load`` different identities and the cell's final value was
rebuilt instead of restored.
"""

import types

from cash.notebook.lineage_formula import call_time_reads

CODE = "def load():\n    _ran('load')\n    return np.full(3, time.time())"


def _ran(name):
    return name


def test_a_helper_named_with_an_underscore_is_read_at_call_time_bound_or_not():
    modules = {"np": types.ModuleType("numpy"), "time": types.ModuleType("time")}
    in_the_kernel = call_time_reads(CODE, {"_ran": _ran, **modules})
    simulated = call_time_reads(CODE, {}, {"np", "time"})
    assert in_the_kernel == simulated == frozenset({"_ran"})
