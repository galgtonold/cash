"""Pytest plugin: record which test each covered line belongs to.

Loaded only by ``run_baseline.py`` (``-p covctx_plugin``). Coverage itself is
started in every Python process by ``sitecustomize.py``; this plugin labels
the data with the running test's node id, in two places:

* the pytest worker process, at the start of every test;
* the Jupyter kernel the test drives, right after ``start_kernel()`` and
  ``restart()`` return. Warm kernels are shared by many tests, so the kernel
  is relabelled for every test that uses it.

Each relabel also saves the kernel's data so far. Windows kills kernels
without running atexit handlers, so data that is only written at exit would
be lost.
"""

from __future__ import annotations

import os

import coverage


def _switch_here(label: str) -> None:
    cov = coverage.Coverage.current()
    if cov is not None:
        cov.switch_context(label)


_KERNEL_SNIPPET = (
    "import coverage as _cov_mod\n"
    "_cov = _cov_mod.Coverage.current()\n"
    "if _cov is not None:\n"
    "    _cov.switch_context({label!r})\n"
    "    _cov.save()\n"
    "del _cov, _cov_mod\n"
)


def _current_test() -> str:
    # "tests/x.py::test_y (call)" -> "tests/x.py::test_y"
    return os.environ.get("PYTEST_CURRENT_TEST", "unknown").rsplit(" ", 1)[0]


def _label_kernel(runner) -> None:
    client = getattr(runner, "client", None)
    kc = getattr(client, "kc", None)
    if kc is None:
        return
    code = _KERNEL_SNIPPET.format(label=_current_test())
    try:
        runner._run_async(kc._async_execute_interactive(code, store_history=False, silent=True))
    except Exception as exc:  # a lost label must never fail the test itself
        print(f"[covctx] could not label kernel: {exc!r}")


def _wrap(cls, name: str) -> None:
    original = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _label_kernel(self)
        return result

    wrapped.__wrapped__ = original
    setattr(cls, name, wrapped)


def pytest_collection_finish(session):
    from tests._nbharness.runner import NotebookTestRunner as runner_cls

    if getattr(runner_cls, "_covctx_wrapped", False):
        return
    _wrap(runner_cls, "start_kernel")
    _wrap(runner_cls, "restart")
    runner_cls._covctx_wrapped = True


def pytest_runtest_setup(item):
    _switch_here(item.nodeid)


# ---------------------------------------------------------------------------
# Per-test results, written by the controlling process only. xdist forwards
# every worker's reports to it, so this sees all of them.

_results: dict[str, dict] = {}


def pytest_runtest_logreport(report):
    entry = _results.setdefault(report.nodeid, {"outcome": "passed", "duration": 0.0, "reruns": 0})
    entry["duration"] += report.duration
    if report.outcome == "rerun":
        entry["reruns"] += 1
    elif report.failed:
        entry["outcome"] = "error" if report.when != "call" else "failed"
    elif report.skipped and entry["outcome"] == "passed":
        entry["outcome"] = "skipped"


def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        return
    out = os.environ.get("TESTSEL_RESULTS")
    if not out:
        return
    import json

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_results, fh, indent=0, sort_keys=True)
