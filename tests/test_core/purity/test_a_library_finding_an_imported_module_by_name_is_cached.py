"""A library's own lookup of an imported module by a run-time name caches
on Python 3.12+; the same lookup in the user's code still runs uncached.

Array-API dispatch (scikit-learn's and SciPy's vendored
``array_api_compat``) tests an input's type against a module only when it
is already imported: ``sys.modules[modname]`` in an ``lru_cache``-d helper.
Reported like ``eval``, it ran every call of a cached library function
that reaches it (``randomized_svd``) uncached. In installed code it is a
lookup by name like the library's ``getattr`` and ``import_module``: the
library is keyed by its version, and the call is traced as theirs are.
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

LIB = """
import sys
from functools import lru_cache


@lru_cache(100)
def _issubclass_fast(cls, modname, clsname):
    try:
        mod = sys.modules[modname]
    except KeyError:
        return False
    return issubclass(cls, getattr(mod, clsname))


def is_fraction(x):
    return _issubclass_fast(type(x), "fractions", "Fraction")


def weight(xs):
    return sum(10 if is_fraction(x) else 1 for x in xs)
"""

JOB = """
import cash
from {module} import weight

cached = cash.cache(weight)
print(cached([1, 2, 3]))
"""


def _project(tmp_path, module_home: str):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "proj"\n', encoding="utf-8")
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    if module_home == "library":
        (site / "arraylib").mkdir()
        (site / "arraylib" / "__init__.py").write_text(textwrap.dedent(LIB), encoding="utf-8")
        info = site / "arraylib-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: arraylib\nVersion: 1.0\n", encoding="utf-8")
        (info / "top_level.txt").write_text("arraylib\n", encoding="utf-8")
        module = "arraylib"
    else:
        (proj / "dispatch.py").write_text(textwrap.dedent(LIB), encoding="utf-8")
        module = "dispatch"
    (proj / "job.py").write_text(JOB.format(module=module), encoding="utf-8")
    return proj, site


def _hits(proj, site) -> tuple[str, str]:
    env = {"PYTHONPATH": os.pathsep.join([CASH_SRC, str(site)]), "CASH_SUMMARY": "1"}
    p = run_python("job.py", cwd=proj, env=env)
    return p.stdout.strip(), p.stderr


@pytest.mark.skipif(sys.version_info < (3, 12), reason="a lookup by name is cached on a trace, which needs 3.12+")
def test_a_library_s_lookup_is_served_in_the_next_process(tmp_path):
    proj, site = _project(tmp_path, "library")
    out, _ = _hits(proj, site)
    assert out == "3"
    out, err = _hits(proj, site)
    assert out == "3"
    assert "1 hit" in err, err


def test_the_user_s_own_lookup_still_runs_uncached(tmp_path):
    proj, site = _project(tmp_path, "user")
    _hits(proj, site)
    out, err = _hits(proj, site)
    assert out == "3"
    assert "1 hit" not in err, err
    assert "run-time name" in err, err


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="checks the run without a trace")
def test_without_a_trace_a_library_s_lookup_runs_uncached(tmp_path):
    proj, site = _project(tmp_path, "library")
    _hits(proj, site)
    out, err = _hits(proj, site)
    assert out == "3"
    assert "1 hit" not in err, err
