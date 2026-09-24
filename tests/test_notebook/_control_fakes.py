"""A stand-in ``StatementProcessor`` for control-structure unit tests.

Specced on the real class, so the control code can only call what
``StatementProcessor`` has: a renamed or missing member fails here as it
would in a kernel. The state it reads is real (a ``TrackingState`` and a
``Cash`` over an in-memory backend, with the default config); only the
statement execution is faked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import StatementProcessor
from cash.notebook.tracking_state import TrackingState


def computed_metrics() -> dict:
    """What ``process_statement`` returns for a statement that ran."""
    return {
        "status": CacheStatus.COMPUTED,
        "execution_time": 0.01,
        "stdout": "",
        "stderr": "",
        "outputs": [],
    }


def fake_statement_processor() -> MagicMock:
    processor = MagicMock(spec=StatementProcessor)
    processor.tracking_state = TrackingState()
    processor.cash_instance = Cash(backend=InMemoryBackend(), register_magic=False)
    processor.compute_hash = MagicMock(return_value="fakehash")
    processor.process_statement.return_value = computed_metrics()
    return processor
