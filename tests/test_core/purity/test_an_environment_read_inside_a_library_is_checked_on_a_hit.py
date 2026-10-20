"""An environment variable read inside an installed package is checked on a hit.

The key folds the environment reads written out in the user's code. One
inside an installed package (a team package's ``os.environ["TENANT"]``) was
not seen, and the first tenant's result was served to every other tenant.
A computed call now notes the ``os.environ`` reads installed code made while
it ran, and the entry keeps a digest of each value: a lookup reads them
again and serves the entry only when they all match.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

LIB = """
import os

def tenant_rows(n):
    return f"{os.environ.get('TENANT', 'none')}:{n}"
"""

JOB = """
import sys
import cash
import tenantlib

@cash.cache
def report(n):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return tenantlib.tenant_rows(n)

print(report(3))
print(report(3))
"""


@pytest.fixture
def proj(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "proj"\n', encoding="utf-8")
    (proj / "job.py").write_text(JOB, encoding="utf-8")
    site = tmp_path / "venv" / "lib" / "site-packages"
    (site / "tenantlib").mkdir(parents=True)
    (site / "tenantlib" / "__init__.py").write_text(textwrap.dedent(LIB), encoding="utf-8")
    info = site / "tenantlib-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: tenantlib\nVersion: 1.0\n", encoding="utf-8")
    (info / "top_level.txt").write_text("tenantlib\n", encoding="utf-8")
    return proj, site


def _run(proj, site, tenant):
    env = {"PYTHONPATH": os.pathsep.join([CASH_SRC, str(site)]), "TENANT": tenant}
    p = run_python("job.py", cwd=proj, env=env)
    assert p.returncode == 0, p.stderr
    return p.stdout.split(), p.stderr.count("[RUN]")


def test_another_value_recomputes(proj):
    proj, site = proj
    assert _run(proj, site, "acme") == (["acme:3", "acme:3"], 1), "the first call is stored"
    assert _run(proj, site, "acme") == (["acme:3", "acme:3"], 0), "the same value hits"
    assert _run(proj, site, "globex") == (["globex:3", "globex:3"], 1), "another tenant was served acme's result"
    assert _run(proj, site, "acme") == (["acme:3", "acme:3"], 1), "acme's entry was replaced by globex's"
