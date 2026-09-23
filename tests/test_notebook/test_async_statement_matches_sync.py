"""The async statement path takes the same requests as the sync one.

``process_statement_async`` runs a statement holding a top-level ``await``.
Everything but the execution itself is shared with ``process_statement``,
so a request the sync path honours must not be dropped on the async one.
"""

import asyncio


def test_forced_outputs_are_captured_on_the_async_path(statement_processor):
    """The accumulator-loop fast path names its accumulator as a forced
    output. The async path ignored the argument, so the accumulator was
    neither captured nor restored there."""
    statement_processor.shell.user_ns["acc"] = []
    code = "for e in [1, 2]:\n    acc.append(e)"

    sync = statement_processor.process_statement(code, silent=True, force_outputs={"acc", "e"})
    statement_processor.shell.user_ns["acc"] = []
    asynchronous = asyncio.run(
        statement_processor.process_statement_async(code, silent=True, force_outputs={"acc", "e"})
    )

    assert {"acc", "e"} <= set(sync["evaluated_vars"])
    assert set(asynchronous["evaluated_vars"]) == set(sync["evaluated_vars"])
