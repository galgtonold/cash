"""Notebook shapes: Run All x3, a restart, Run All x2, in a real kernel
(`tests._nbharness.runner.NotebookTestRunner`), under cash or plain."""

from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
from typing import Any, Iterator

from tools.plain_checker import CONFIGS, RUNS, Outcome, Shape

#: Run in the kernel after cash is on, in the ``floors0`` config.
FLOORS0_SNIPPET = "# @cash:no-cache\nimport cash.notebook.call_unit as _pc_cu; _pc_cu._COST_FLOOR_S = 0; del _pc_cu"

_FINGERPRINT_SOURCE = (Path(__file__).parent / "_fingerprint.py").read_text(encoding="utf-8")

_STATUSES = (
    "[str(s.get('status')).split('.')[-1] for s in "
    "get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics.get('statements', [])]"
)


def assigned_names(cells: list[str]) -> list[str]:
    """The top-level names the cells assign, in order: not ``def``,
    ``class`` or ``import`` targets."""
    names: list[str] = []
    for code in cells:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            targets: list[Any] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                targets = [node.target] if hasattr(node, "target") else [i.optional_vars for i in node.items]
            for target in targets:
                for sub in ast.walk(target) if target is not None else ():
                    if isinstance(sub, ast.Name) and sub.id not in names and not sub.id.startswith("_"):
                        names.append(sub.id)
    return names


@contextlib.contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _shown(cell: Any) -> str:
    """What the cell showed the user: stdout, its value and any error.
    stderr (cash's warnings) and display data (the badge) are left out."""
    parts = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream" and output.get("name") == "stdout":
            parts.append(output.get("text", ""))
        elif kind == "execute_result":
            parts.append(output.get("data", {}).get("text/plain", "") + "\n")
        elif kind == "error":
            parts.append(f"{output.get('ename')}: {output.get('evalue')}\n")
    from tools.plain_checker._fingerprint import _short

    return _short("".join(parts))


def _peek(runner: Any, expr: str) -> Any:
    shown = runner.peek(expr)
    try:
        return ast.literal_eval(shown)
    except (ValueError, SyntaxError):
        return shown


def _values(runner: Any, names: list[str]) -> dict[str, str]:
    expr = f"(lambda d: (exec({_FINGERPRINT_SOURCE!r}, d), d['fingerprints'](globals(), {names!r}))[1])({{}})"
    got = _peek(runner, expr)
    return got if isinstance(got, dict) else {name: f"<peek failed: {got}>" for name in names}


def _cash_on(runner: Any, config: str, *, init: bool = True) -> None:
    if init:
        runner._init_cash()
    if config == "floors0":
        runner._run_async(
            runner.client.kc._async_execute_interactive(
                FLOORS0_SNIPPET, store_history=False, output_hook=lambda _m: None
            )
        )


def run_notebook(shape: Shape, work: Path, config: str | None) -> Outcome:
    """Run *shape* in a fresh kernel under *work*: under cash in *config*,
    or plain Python when *config* is None."""
    from tests._nbharness.runner import NotebookTestRunner

    work.mkdir(parents=True, exist_ok=True)
    names = (
        shape.names
        if shape.names is not None
        else assigned_names([code for run in range(len(RUNS)) for code in shape.cells_for(run)])
    )
    outcome = Outcome()
    runner = NotebookTestRunner(work_dir=work, timeout=300)
    runner._force_fresh_kernel = True
    runner.create_notebook(shape.cells_for(0))
    try:
        with _environment(CONFIGS[config] if config is not None else {}):
            runner.start_kernel(with_cash=config is not None)
        if config is not None:
            _cash_on(runner, config, init=False)  # start_kernel turned cash on
        for index, run_name in enumerate(RUNS):
            if run_name == "restart+run1":
                runner.restart()
                if config is not None:
                    _cash_on(runner, config)  # a restart leaves the kernel without cash
            cells = shape.cells_for(index)
            for number, code in enumerate(cells, start=1):
                if runner.get_cell_source(number) != code:
                    runner.set_cell_source(number, code)
            seen = outcome.seen.setdefault(run_name, {})
            for number in range(1, len(cells) + 1):
                runner.run_cell(number)
                seen[f"cell {number - 1} output"] = _shown(runner.get_cell(number))
                if config is not None:
                    statuses = _peek(runner, _STATUSES)
                    outcome.statuses.setdefault(run_name, {})[number - 1] = (
                        statuses if isinstance(statuses, list) else [str(statuses)]
                    )
            for name, value in _values(runner, names).items():
                seen[f"{name}"] = value
    finally:
        runner.shutdown()
    return outcome
