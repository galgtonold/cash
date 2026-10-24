"""A package's read of ``os.environ`` is seen however many bodies ran before.

Cash put its read hook on ``os.environ``'s class when observation started and
took it off when it stopped, around every observed body. On Python 3.13 a
class changed that often stops telling its callers: after about a thousand
changes, ``Mapping.get``'s ``self[key]`` kept calling the ``__getitem__`` it
had seen first, the hook was never called, and a package's
``os.environ.get("TENANT")`` went unnoticed: in a long notebook session the
first tenant's rows were served to every other. The hook now stays on.
"""

from __future__ import annotations

import importlib
import os

from cash.tracking import env_reads, io_watch

READER = "import os\ndef tenant():\n    return os.environ.get('CASH_UT_HOOK_TENANT')\n"


def test_the_hook_stays_on_after_observation_stops():
    io_watch.hold()
    io_watch.release()
    assert type(os.environ).__getitem__ is env_reads._getitem


def test_a_package_read_is_seen_after_many_observed_bodies(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_hook_reader.py").write_text(READER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("CASH_UT_HOOK_TENANT", "acme")
    reader = importlib.import_module("cash_ut_hook_reader")
    for _ in range(1500):  # the bodies of a long session, most reading no variable
        io_watch.hold()
        io_watch.release()
        reader.tenant()  # and reads between them, as anything makes

    io_watch.hold()
    try:
        with env_reads.recording() as reads:
            assert reader.tenant() == "acme"
    finally:
        io_watch.release()
    assert reads == {"CASH_UT_HOOK_TENANT"}
