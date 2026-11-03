"""A statement that ran long enough to matter and left no entry on disk --
refused, or kept in memory only -- says why when it runs again after a
restart: there is no entry and no version to compare, and the miss said
nothing (``faces = fetch_lfw_people(...)``, refused because it wrote its
own cache files, recomputed on every restart with no reason given)."""

from __future__ import annotations

import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def clean_backend(tmp_path):
    # On disk: what a restarted kernel finds is what the run before stored.
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("e5_loader", "time", "v"):
        magics.shell.user_ns.pop(name, None)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


_LOADER = """
import os, time

def load(path):
    time.sleep(0.15)
    if not os.path.exists(path):  # its own cache, written the first time only
        # Reached as a library reaches it: out of sight of the static check,
        # so only the write seen as it ran refuses the statement.
        write = getattr(__import__("builtins"), "op" + "en")
        with write(path, "w") as fh:
            fh.write("x")
    return 3
"""


@pytest.fixture
def loader(tmp_path, monkeypatch):
    (tmp_path / "e5_loader.py").write_text(_LOADER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("e5_loader", None)


def _writes(path) -> str:
    return f"v = e5_loader.load({str(path)!r})"


@pytest.mark.usefixtures("loader")
def test_a_refused_statement_names_its_refusal(cash_magics, statement_processor, tmp_path):
    _run(statement_processor, "import e5_loader")
    first = _run(statement_processor, _writes(tmp_path / "out.txt"))
    assert first.get("uncacheable_reasons")
    _restart(cash_magics)

    _run(statement_processor, "import e5_loader")
    again = _run(statement_processor, _writes(tmp_path / "out.txt"))

    assert again["status"] == CacheStatus.COMPUTED
    reason = again.get("miss_reason") or ""
    assert reason.startswith("not stored on disk when it last ran: ")
    assert first["uncacheable_reasons"][0][:40] in reason


def test_a_stored_statement_says_nothing_of_the_kind(cash_magics, statement_processor):
    _run(statement_processor, "import time")
    _run(statement_processor, "v = (time.sleep(0.15), 3)[1]")
    _restart(cash_magics)

    _run(statement_processor, "import time")
    again = _run(statement_processor, "v = (time.sleep(0.15), 3)[1]")

    assert again["status"] == CacheStatus.RESTORED
    assert again.get("miss_reason") is None
