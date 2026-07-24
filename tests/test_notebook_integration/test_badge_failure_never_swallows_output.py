"""A failing badge renderer must never abort the cell or swallow its output.

The badge is a diagnostic overlay drawn *around* the user's statements — cash
renders it before each statement runs. If building or displaying it raises, that
exception used to propagate out of the statement loop, aborting the cell before
the user's code ran: every cell went blank while cash itself still loaded (you'd
see the ``%cash_on`` message, then nothing). That is exactly how a renderer that
raised at import on Python 3.11 manifested in Binder.

The badge is never worth breaking a cell over, so a render failure now degrades
to "no badge" and the cell runs and shows its output as normal.
"""
from __future__ import annotations


def _text(cell) -> str:
    return "".join(
        o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    )


def _raw_exec(nb_runner, code: str) -> None:
    """Run code straight on the kernel, bypassing cash's cell hook (so this
    setup itself renders no badge)."""
    nb_runner._run_async(nb_runner.client.kc._async_execute_interactive(
        code, store_history=False, output_hook=lambda m: None,
    ))


def test_badge_render_failure_does_not_swallow_output(nb_runner):
    nb_runner.create_notebook(["print('HELLO'); x = 41 + 1"])
    nb_runner.start_kernel()

    # Replace the badge renderer with one that always raises — the same shape as
    # the import-time failure seen in the wild. Installed via a raw kernel exec
    # so this setup does not itself route through cash / render a badge.
    _raw_exec(
        nb_runner,
        "import cash.notebook.badge_renderer as _br\n"
        "def _boom(*a, **k):\n"
        "    raise RuntimeError('badge boom')\n"
        "_br.render_interactive_badge = _boom\n"
        "_br.print_text_badge = _boom\n",
    )

    nb_runner.run_cell(1)  # a perfectly ordinary cell

    out = _text(nb_runner.get_cell(1))
    assert "HELLO" in out, f"a failing badge swallowed the cell's output: {out!r}"

    # ...and the statement actually executed (the cell was not aborted).
    val = []
    nb_runner._run_async(nb_runner.client.kc._async_execute_interactive(
        "print('XIS', x)", store_history=False,
        output_hook=lambda m: val.append(m["content"].get("text", "")) if m["msg_type"] == "stream" else None,
    ))
    assert "XIS 42" in "".join(val), "the statement did not run — the cell was aborted"
