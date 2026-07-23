"""Reading a class constant through the class NAME must be a tracked dependency.

``self.RATE`` was tracked, but ``ClassName.RATE`` and ``type(self).RATE`` -- the
same value reached a different way -- were not, so editing a class-level config
constant served a stale result. Class config constants (``Config.BATCH_SIZE``,
``Model.THRESH``) read by the class name are common, and this hit a plain FREE
function too, not only methods.

Cross-process (the stale serve only shows after a second process rebuilds the
key); ``time.sleep(0.3)`` clears the persistence floor; ``.cash`` is never
cleared, or the test is vacuous.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def _run(tmp_path):
    cp = subprocess.run(
        [sys.executable, "main.py"], cwd=str(tmp_path),
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, f"failed:\n{cp.stdout}\n{cp.stderr}"
    return cp.stdout.strip().splitlines()[-1].split("R ", 1)[1].strip()


FREE_FN = """\
import warnings; warnings.simplefilter("ignore")
import time, cash
class Cfg:
    LIMIT = {k}
@cash.cache
def compute(x):
    time.sleep(0.3)
    return x + Cfg.LIMIT
print("R", compute(100))
"""

METHOD_TYPE_SELF = """\
import warnings; warnings.simplefilter("ignore")
import time, cash
class Model:
    THRESH = {k}
    def __init__(self, x):
        self.x = x
    @cash.cache
    def compute(self, y):
        time.sleep(0.3)
        return self.x + y + type(self).THRESH
    def __hash__(self):
        return hash(self.x)
    def __eq__(self, o):
        return isinstance(o, Model) and self.x == o.x
print("R", Model(100).compute(0))
"""


@pytest.mark.parametrize("template,k1,r1,k2,r2", [
    (FREE_FN, "0", "100", "250", "350"),
    (METHOD_TYPE_SELF, "0", "100", "777", "877"),
])
def test_class_attr_via_name_invalidates(tmp_path, template, k1, r1, k2, r2):
    main = tmp_path / "main.py"
    main.write_text(template.format(k=k1), encoding="utf-8")
    assert _run(tmp_path) == r1

    main.write_text(template.format(k=k2), encoding="utf-8")
    assert _run(tmp_path) == r2, "a class constant read via the class name went stale"


def test_only_user_classes_are_folded():
    """The class-attr fold must exclude stdlib / third-party classes, or an
    upgrade that changes a library class attribute would churn every key."""
    import collections
    import decimal

    from cash import Cash

    # stdlib / builtin classes -> excluded
    assert Cash._is_user_class(decimal.Decimal) is False
    assert Cash._is_user_class(collections.OrderedDict) is False
    assert Cash._is_user_class(int) is False

    # a class defined in this test module resolves to a user module
    assert Cash._is_user_class(_UserClassProbe) is True


class _UserClassProbe:
    CONST = 1
