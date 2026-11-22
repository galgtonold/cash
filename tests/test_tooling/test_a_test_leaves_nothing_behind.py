"""What one test leaves in the process is dropped before the next
(`tests/_between_tests.py`, run by the root conftest after every test).

An xdist worker runs thousands of tests in one process, and cash's heap
reads (the share check's ``gc.get_referrers``) slow down with every object
the process holds, so a leak here shows up as a late test running ten times
slower than alone.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

from cash import Cash, install_paths, source_norm
from cash.notebook import string_lookups
from tests import _between_tests


def test_the_memos_a_statement_fills_are_emptied_and_the_environment_kept():
    source_norm.exact_source_digest("x_left_behind = 1")
    string_lookups._STATEMENT_CODES["x_left_behind = 1"] = None
    install_paths.interpreter_roots()
    assert source_norm.exact_source_digest.cache_info().currsize
    _between_tests.forget_cash_memos()
    assert source_norm.exact_source_digest.cache_info().currsize == 0
    assert len(string_lookups._STATEMENT_CODES) == 0
    # A one-entry fact about the interpreter is asked by every test: kept.
    assert install_paths.interpreter_roots.cache_info().currsize == 1


def test_a_collected_cash_no_longer_holds_its_backend_through_atexit():
    cash = Cash(register_magic=False)
    kept = Cash(register_magic=False)
    gone = weakref.ref(cash)
    runs = [run for _, run in _between_tests._exit_work]
    assert cash._exit_work.run in runs and kept._exit_work.run in runs
    del cash, runs
    gc.collect()
    _between_tests.forget_dead_cash_exit_work()
    assert gone() is None
    held = [instance() for instance, _ in _between_tests._exit_work]
    assert any(instance is kept for instance in held)
    assert None not in held


def test_ipython_output_history_is_emptied():
    from IPython.core.history import HistoryManager, HistoryOutput

    HistoryManager.outputs[1].append(HistoryOutput(output_type="execute_result", bundle={"text/plain": "1"}))
    _between_tests.forget_ipython_outputs()
    assert not HistoryManager.outputs


def test_a_test_class_instance_lets_go_of_what_its_setup_made():
    instance = SimpleNamespace(cash=object())
    _between_tests.forget_test_instance(SimpleNamespace(instance=instance))
    assert vars(instance) == {}
    _between_tests.forget_test_instance(SimpleNamespace(instance=None))  # a plain test function
