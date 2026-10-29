"""A loop that calls a helper setting state on a module runs again, as a
statement that sets it does.

``def bump(): import cnt; cnt.k += 1``, with ``cnt`` a module made as the
notebook ran (``types.ModuleType('cnt')``): having no file, it did not count
as the user's, and a loop that rebinds its own outputs (``t = t + 1``) was
served from the cache, leaving ``cnt.k`` as the cell above reset it.
After a restart, before the run saw the helper write, a module from a file
too: the helper binds ``cnt`` as its own local, not one of the notebook's.
The same through a function the helper defines, through ``setattr``, and
for a helper that is a module's own function.
"""

import linecache
import sys
import types

import pytest

from cash.notebook.callee_reach import module_state_writes

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


HELPERS = {
    # Imported in a function defined in the helper.
    "nested_def": (MADE, "def bump():\n    def inner():\n        import cnt\n        cnt.k += 1\n    inner()", "cnt"),
    "setattr": (MADE, "def bump():\n    import cnt\n    setattr(cnt, 'k', cnt.k + 1)", "cnt"),
    # The helper is a module's own function, importing the module it sets.
    "helper_in_a_file": (
        FILE.replace("cntfile", "cnthf")
        + "open('helphf.py', 'w').write('def bump():\\n    import cnthf\\n    cnthf.k += 1\\n')\n"
        + "importlib.invalidate_caches()\nimport helphf\nimportlib.reload(helphf)\nfrom helphf import bump",
        "",
        "cnthf",
    ),
}


@pytest.mark.parametrize("helper", list(HELPERS))
def test_a_helper_reaching_the_module_another_way_sets_it_again(nb_runner, helper):
    setup, bump, name = HELPERS[helper]
    nb_runner.create_notebook([setup + bump, "t = 0\n" + LOOP, f"import {name}\nprint({name}.k)"])
    nb_runner.start_kernel()
    for restart in (False, False, False, True, False):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3), f"the loop was served: {nb_runner.get_output(3)!r}"


@pytest.mark.parametrize("helper", ["nested_def", "setattr"])
def test_the_helper_is_read_as_setting_the_module(helper, monkeypatch):
    module = types.ModuleType("cnt")
    module.k = 0
    monkeypatch.setitem(sys.modules, "cnt", module)
    namespace: dict = {"__name__": "__main__"}
    source = HELPERS[helper][1] + "\n"
    filename = f"<cell-{helper}>"
    monkeypatch.setitem(linecache.cache, filename, (len(source), None, source.splitlines(True), filename))
    exec(compile(source, filename, "exec"), namespace)
    assert module_state_writes("bump()", namespace) == {"cnt"}


#: ``(bump, what reads the state)``: what the helper took from the module by
#: ``from cnt import ...`` changed in place, and ``setattr`` under a name of
#: its own.
TAKEN = {
    "from_import_dict": ("def bump():\n    from cnt import d\n    d['k'] += 1", "cnt.d['k']"),
    "from_import_list": ("def bump():\n    from cnt import l\n    l.append(1)", "len(cnt.l)"),
    "setattr_alias": ("def bump():\n    import cnt\n    sa = setattr\n    sa(cnt, 'k', cnt.k + 1)", "cnt.k"),
}


@pytest.mark.parametrize("helper", list(TAKEN))
def test_a_helper_changing_what_it_took_from_the_module_sets_it_again(nb_runner, helper):
    bump, read = TAKEN[helper]
    setup = MADE + "m.d = {'k': 0}\nm.l = []\n"
    nb_runner.create_notebook([setup + bump, "t = 0\n" + LOOP, f"import cnt\nprint({read})"])
    nb_runner.start_kernel()
    for restart in (False, False, False, True, False):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3), f"the loop was served: {nb_runner.get_output(3)!r}"


@pytest.mark.parametrize("helper", list(TAKEN))
def test_what_it_took_is_read_as_the_modules(helper, monkeypatch):
    module = types.ModuleType("cnt")
    module.k, module.d, module.l = 0, {"k": 0}, []
    monkeypatch.setitem(sys.modules, "cnt", module)
    namespace: dict = {"__name__": "__main__"}
    source = TAKEN[helper][0] + "\n"
    filename = f"<cell-{helper}>"
    monkeypatch.setitem(linecache.cache, filename, (len(source), None, source.splitlines(True), filename))
    exec(compile(source, filename, "exec"), namespace)
    assert module_state_writes("bump()", namespace) == {"cnt"}
