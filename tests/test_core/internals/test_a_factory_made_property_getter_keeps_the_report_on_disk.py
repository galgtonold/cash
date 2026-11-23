"""The report of a getter a factory made for a property or a
``cached_property`` is stored, and a fresh process reads it instead of
walking again.

scanpy's ``preset_property`` wraps each preset table in a getter made inside
the factory (``functools.wraps`` gives it the table's name) and hands it to
``cached_property``. No name led back to that getter, so the walk of every
method of the enum, which a key folding one of its members reads for their
environment reads, ran again in every process: a tenth of a second for one
enum member. The class attribute holding the descriptor is the way back.
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
    from cash.analysis.purity_analyzer import get_analyzer
    import presets
    # A walk cheaper than reading its report back is not stored, and these
    # getters' walks sit near that line: a busy runner tipped one of them.
    report_store._READ_COST = -1.0
    an = get_analyzer()
    before = report_store.status()
    for getter in (vars(presets.Preset)["table"].func, vars(presets.Preset)["flag"].fget):
        an.analyze(getter)
    getters = {k: v - before[k] for k, v in report_store.status().items()}
    top = cash.cache(app.top)
    value = top(2)
    key = top.explain(2).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "getters": getters, "key": key, "value": value}))
    """
)

_APP = """
import presets


def top(n):
    return [presets.Preset.A.table * n, presets.Preset.B.flag]
"""

_PRESETS = """
import enum
from functools import cached_property, wraps


def _scale():
    return 3


def preset_property(get_map):
    @wraps(get_map)
    def get(self):
        return get_map()[self] * _scale()

    return cached_property(get)


def plain_property(get_map):
    @wraps(get_map)
    def get(self):
        return get_map()[self]

    return property(get)


class Preset(enum.Enum):
    A = "a"
    B = "b"

    @preset_property
    def table():
        return {Preset.A: 1, Preset.B: 2}

    @plain_property
    def flag():
        return {Preset.A: False, Preset.B: True}
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(code: Path, presets: str = _PRESETS) -> None:
    code.mkdir(exist_ok=True)
    (code / "presets.py").write_text(presets, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")


def test_a_factory_made_getter_is_read_back_in_a_new_process(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert first["value"] == [6, True]
    assert first["stats"]["stored"] >= 2, first["stats"]
    second = _run(code)
    # Both getters come back from disk, neither is walked again.
    assert second["getters"]["loaded"] == 2 and second["getters"]["missed"] == 0, second["getters"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 2


def test_an_edit_to_what_the_getter_runs_walks_it_again_and_misses(tmp_path):
    code = tmp_path / "code"
    _write(code)
    _run(code)
    assert _run(code)["getters"]["loaded"] == 2
    _write(code, _PRESETS.replace("return 3", "return 5"))
    edited = _run(code)
    # The stored report of the getter that reaches the edit is not used,
    # and the call is not served from before the edit. The other getter
    # reads nothing the edit touched (the function it reaches is another of
    # the same file): its report is read back.
    assert edited["getters"]["stale"] == 1 and edited["getters"]["loaded"] == 1, edited["getters"]
    assert edited["value"] == [10, True]
    assert _run(code, verify=True)["stats"]["verified"] >= 2
