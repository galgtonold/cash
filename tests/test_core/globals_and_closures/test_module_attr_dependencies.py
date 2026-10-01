"""A cached function must see the module attributes it reads.

``import conf; conf.RATE`` went permanently stale: the only global the body
references is ``conf``, a module, and modules were filtered out before their
attributes were ever considered. The equivalent ``from conf import RATE``
invalidated correctly, so the same dependency was tracked or not depending on
which import spelling you happened to use -- and the failure was silent, with
``explain()`` reporting a confident ``[HIT]``.

Found by an adversarial sweep against the 0.1.0 wheel and reproduced
independently 3/3.

The first two tests run real scripts in fresh processes on purpose. The bug is
that a *second run* reuses the persisted entry after the constant changed,
which an in-process test cannot express; and writing the function inside a test
body would make ``conf`` a closure variable (``LOAD_DEREF``), which compiles
differently from the module-level ``import`` that real code uses.
"""

from __future__ import annotations

import types

import cash
from cash.install_paths import is_user_module
from tests._scripts import run_python

MAIN = """\
import warnings; warnings.simplefilter('ignore')
import time
import cash
from cash.install_paths import is_user_module
import conf
RAN = [0]

@cash.cache
def compute(x):
    RAN[0] += 1
    {body}

print(compute(10), RAN[0])
"""


def _write_main(tmp_path, body):
    (tmp_path / "main.py").write_text(MAIN.format(body=body), encoding="utf-8")


def _run(tmp_path):
    # The edits keep conf.py's size (``RATE = 2.0`` -> ``3.0``); run_python
    # writes no .pyc, so the next process compiles the edited source.
    cp = run_python("main.py", cwd=tmp_path)
    return cp.stdout.strip()


def test_module_attribute_read_invalidates_when_it_changes(tmp_path):
    _write_main(tmp_path, "return x * conf.RATE")
    conf = tmp_path / "conf.py"

    conf.write_text("RATE = 2.0\n", encoding="utf-8")
    assert _run(tmp_path) == "20.0 1"

    conf.write_text("RATE = 3.0\n", encoding="utf-8")
    assert _run(tmp_path) == "30.0 1", "stale value served after the constant changed"

    conf.write_text("RATE = 7.0\n", encoding="utf-8")
    assert _run(tmp_path) == "70.0 1"


def test_helper_reached_through_the_module_sees_its_own_constants(tmp_path):
    """``conf.get_rate()`` whose SOURCE never changes but whose constant does.

    The helper-source channel hashes the callee's source, which is unchanged
    here, so nothing invalidated. One level of recursion into the helper's own
    module globals closes it.
    """
    _write_main(tmp_path, "return x * conf.get_rate()")
    conf = tmp_path / "conf.py"

    conf.write_text("RATE = 2.0\ndef get_rate():\n    return RATE\n", encoding="utf-8")
    assert _run(tmp_path) == "20.0 1"

    conf.write_text("RATE = 5.0\ndef get_rate():\n    return RATE\n", encoding="utf-8")
    assert _run(tmp_path) == "50.0 1", "helper's constant changed but nothing invalidated"


def test_installed_module_attrs_do_not_churn_the_key():
    """Reading a stdlib / site-packages attribute must not add key churn.

    This is the over-invalidation guard. Folding every module attribute would
    rope in things like ``os.environ``, making every call miss. Third-party and
    stdlib contents are fixed for a given environment, so ``is_user_module``
    excludes them.
    """
    import math

    calls = []

    @cash.cache
    def compute(x):
        calls.append(x)
        return x * math.pi

    assert compute(2) == compute(2)
    assert len(calls) == 1, "a stdlib attribute read caused a spurious cache miss"


def test_function_reading_no_module_attrs_is_unaffected():
    """Regression guard: ordinary functions must keep hitting."""
    calls = []

    @cash.cache
    def plain(x):
        calls.append(x)
        return x + 1

    assert plain(1) == 2
    assert plain(1) == 2
    assert len(calls) == 1


def test_is_user_module_classification():
    """The stdlib / site-packages boundary itself."""
    import math
    import os as os_mod

    assert is_user_module(math) is False, "stdlib must be excluded"
    assert is_user_module(os_mod) is False, "stdlib must be excluded"
    assert is_user_module(cash) is False, "cash's own modules must be excluded"

    fake = types.ModuleType("looks_like_user_code")
    fake.__file__ = "/home/someone/project/conf.py"
    assert is_user_module(fake) is True

    builtin = types.ModuleType("no_file_at_all")
    assert is_user_module(builtin) is False
