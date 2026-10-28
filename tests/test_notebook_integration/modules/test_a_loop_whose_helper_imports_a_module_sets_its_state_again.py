"""A loop that calls a helper setting state on a module runs again, as a
statement that sets it does.

``def bump(): import cnt; cnt.k += 1``, with ``cnt`` a module made as the
notebook ran (``types.ModuleType('cnt')``): having no file, it did not count
as the user's, and a loop that rebinds its own outputs (``t = t + 1``) was
served from the cache, leaving ``cnt.k`` as the cell above reset it.
After a restart, before the run saw the helper write, a module from a file
too: the helper binds ``cnt`` as its own local, not one of the notebook's.
"""

import pytest

LOOP = "for i in range(300):\n    time.sleep(0.0008)\n    t = t + 1\n    bump()"
MADE = "import time, types, sys\nm = types.ModuleType('cnt')\nm.k = 0\nsys.modules['cnt'] = m\n"
FILE = (
    "import os, sys, time, importlib\nopen('cntfile.py', 'w').write('k = 0\\n')\n"
    "sys.path.insert(0, os.getcwd())\nimportlib.invalidate_caches()\nimport cntfile\ncntfile.k = 0\n"
)


@pytest.mark.parametrize(("setup", "name"), [(MADE, "cnt"), (FILE, "cntfile")], ids=["made_as_it_ran", "from_a_file"])
def test_every_run_sets_the_state_again(nb_runner, setup, name):
    nb_runner.create_notebook(
        [
            setup + f"def bump():\n    import {name}\n    {name}.k += 1",
            "t = 0\n" + LOOP,
            f"import {name}\nprint({name}.k)",
        ]
    )
    nb_runner.start_kernel()
    for restart in (False, False, False, True, False):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3), f"the loop was served: {nb_runner.get_output(3)!r}"
