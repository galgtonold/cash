"""A cached function whose walk reaches a method of a module-level instance
(``model = Model()``, ``model.fit(x)`` called) still has its purity report
stored, and a fresh process reads it instead of walking again.

Such a report was refused ("a method of an object"), and a highly variable
genes step walked its helpers again in every process (225 ms). The name its
caller reads leads to the instance; the report holds while that name leads
to an object of the same kind, defined by the same module text.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app
    from cash.analysis import report_store
    top = cash.cache(app.top)
    value = top(3)
    key = top.explain(3).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "key": key, "value": value}))
    """
)

_APP = """
from models import model


def top(n):
    return model.fit(n)
"""

_MODELS = """
def helper(x):
    return x + 1


class Model:
    def fit(self, x):
        return helper(x) * 2


class Other:
    def fit(self, x):
        return helper(x) * 5


model = Model()
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(code: Path, models: str = _MODELS) -> None:
    code.mkdir(exist_ok=True)
    (code / "models.py").write_text(models, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")


def test_a_report_through_an_instance_method_is_read_in_a_new_process(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert first["value"] == 8
    assert first["stats"]["stored"] >= 1 and not first["stats"]["refused"], first["stats"]
    second = _run(code)
    assert second["stats"]["loaded"] >= 1, second["stats"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 1


def test_an_edit_or_a_rebound_instance_walks_again(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    for edited, value in (
        (_MODELS.replace("helper(x) * 2", "helper(x) * 3"), 12),
        (_MODELS.replace("model = Model()", "model = Other()"), 20),
        (_MODELS.replace("return x + 1", "return x + 2"), 10),
    ):
        _write(code, edited)
        again = _run(code)
        assert again["value"] == value
        assert again["key"] != first["key"]
        assert again["stats"]["loaded"] == 0, again["stats"]


_NT_APP = """
from points import P


def top(n):
    p = P._make([n, 2])
    return [p._replace(a=5)._asdict()["a"], p.a, p.b]
"""


def test_namedtuple_methods_of_a_user_class_are_read_in_a_new_process(tmp_path):
    """``namedtuple`` makes ``P._make``, ``P._replace`` and ``P._asdict`` with
    ``collections`` as their module, for a class in the user's module: found
    again through the class that holds them."""
    code = tmp_path / "code"
    code.mkdir()
    (code / "points.py").write_text('import collections\nP = collections.namedtuple("P", "a b")\n', encoding="utf-8")
    (code / "app.py").write_text(_NT_APP, encoding="utf-8")
    (code / "models.py").write_text(_MODELS, encoding="utf-8")
    first = _run(code)
    assert first["value"] == [5, 3, 2]
    assert not first["stats"]["refused"], first["stats"]
    second = _run(code)
    # What only the second process analyses is stored by it, and read by the third.
    third = _run(code)
    assert third["stats"]["loaded"] >= 3 and not third["stats"]["missed"], third["stats"]
    assert not second["stats"]["refused"] and not third["stats"]["refused"]
    assert (second["key"], second["value"]) == (third["key"], third["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 1
    (code / "points.py").write_text(
        'import collections\nP = collections.namedtuple("P", "a b", defaults=[7])\n', encoding="utf-8"
    )
    edited = _run(code)
    assert edited["key"] != first["key"]
