"""A statement restores after ``cash.configure`` moves the cache.

The statement processor looked entries up in the backend it was built with,
and stored them in the instance's current one. ``cash.configure(cache_dir=...)``
(Cash.reconfigure) in a running notebook builds a new backend, so from then on every statement
was stored where no lookup went: nothing restored until a restart.
"""

from __future__ import annotations

from cash import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


def test_a_statement_stored_after_configure_is_found(mock_shell, tmp_path):
    c = Cash(cache_dir=str(tmp_path / "first"), register_magic=False)
    processor = CashMagics(mock_shell, c)._statement_processor
    processor.process_statement("import time")
    try:
        c.reconfigure(cache_dir=str(tmp_path / "second"))
        code = "y = (time.sleep(0.2), 8)[1]"
        assert processor.process_statement(code)["status"] == CacheStatus.COMPUTED
        mock_shell.user_ns.pop("y")
        again = processor.process_statement(code)
        assert again["status"] == CacheStatus.RESTORED, again.get("miss_reason")
        assert mock_shell.user_ns["y"] == 8
    finally:
        c.backend.shutdown()
