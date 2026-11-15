"""A cached function whose walk reaches a method a ``dataclass`` generated
still has its purity report stored, and a fresh process reads it instead of
walking again.

``dataclass`` builds ``__repr__`` as ``__create_fn__.<locals>.__repr__`` and
holds it behind ``reprlib.recursive_repr``'s wrapper, so neither its name nor
its module led back to it, and the report of a function using such a class
(a mount object of a solar model) was walked in every process. The class
that holds it, by the method's own name, does.
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
    value = top(3.0)
    key = top.explain(3.0).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "key": key, "value": value}))
    """
)

_APP = """
import mounts


def top(tilt):
    m = mounts.Mount(tilt)
    return [repr(m), m.area(), m == mounts.Mount(tilt)]
"""

_MOUNTS = """
import dataclasses


@dataclasses.dataclass
class Mount:
    tilt: float = 0.0

    def area(self):
        return self.tilt * 2
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(code: Path, mounts: str = _MOUNTS) -> None:
    code.mkdir(exist_ok=True)
    (code / "mounts.py").write_text(mounts, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")


def test_a_report_through_a_dataclass_repr_is_read_in_a_new_process(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert first["value"] == ["Mount(tilt=3.0)", 6.0, True]
    assert first["stats"]["stored"] >= 1 and not first["stats"]["refused"], first["stats"]
    second = _run(code)
    assert second["stats"]["loaded"] >= 1, second["stats"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 1


def test_an_edit_to_the_dataclass_walks_again_and_changes_the_key(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    # A new field changes the generated __repr__ and __eq__; a changed
    # method changes the class's own code.
    for edited in (
        _MOUNTS.replace("tilt: float = 0.0", "tilt: float = 0.0\n    azimuth: float = 180.0"),
        _MOUNTS.replace("self.tilt * 2", "self.tilt * 4"),
    ):
        _write(code, edited)
        again = _run(code)
        assert again["key"] != first["key"]
        assert again["value"] != first["value"]
        assert again["stats"]["loaded"] == 0, again["stats"]
