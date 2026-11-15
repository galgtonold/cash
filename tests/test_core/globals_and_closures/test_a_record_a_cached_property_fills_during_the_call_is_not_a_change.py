"""A record a ``cached_property`` fills while the call runs is not a change.

Reported against scanpy checked out as the user's code: its settings hold a
preset, an enum member whose ``cached_property`` getters return a
``NamedTuple`` of plain values (``HVGPreset(flavor="seurat")``). The first
read inside a call stored that record on the member, the walk of the code
the global holds found one more class there, and with another thread alive
(zarr's, asyncio's) the call read as changing the state its key read: no
call was ever stored (STORE-STATE-CHANGED). The record brings no code but
its class, which the getter's code already reaches.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

MAIN = textwrap.dedent("""
    import enum, sys, threading
    from functools import cached_property
    from typing import NamedTuple
    import cash


    class PcaPreset(NamedTuple):
        n_comps: int


    class Preset(enum.Enum):
        V1 = 1
        V2 = 2

        @cached_property
        def pca(self):
            return {Preset.V1: PcaPreset({V1}), Preset.V2: PcaPreset(3)}[self]


    class Settings:
        def __init__(self):
            self.preset = Preset.V1


    settings = Settings()
    stop = threading.Event()
    threading.Thread(target=stop.wait, daemon=True).start()


    @cash.cache
    def pca(xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return [x * settings.preset.pca.n_comps for x in xs]


    print(pca([1, 2]))
    if sys.argv[1:] == ["v2"]:
        settings.preset = Preset.V2
        print(pca([1, 2]))
    stop.set()
""")


def _run(tmp_path, *argv, v1=2):
    (tmp_path / "main.py").write_text(MAIN.replace("{V1}", str(v1)), encoding="utf-8")
    proc = run_python("main.py", *argv, cwd=tmp_path)
    return proc.stdout.split("\n")[:-1], proc.stderr


def test_the_first_call_is_stored_and_the_next_process_hits(tmp_path):
    out, err = _run(tmp_path)
    assert out == ["[2, 4]"] and "[RUN]" in err
    assert "STORE-STATE-CHANGED" not in err, err
    out, err = _run(tmp_path)
    assert out == ["[2, 4]"]
    assert "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"


def test_another_preset_or_an_edited_getter_recomputes(tmp_path):
    _run(tmp_path)
    out, err = _run(tmp_path, "v2")
    assert out == ["[2, 4]", "[3, 6]"]
    assert err.count("[RUN]") == 1, err
    out, err = _run(tmp_path, v1=5)
    assert out == ["[5, 10]"], f"served the result from before the edit:\n{err}"
