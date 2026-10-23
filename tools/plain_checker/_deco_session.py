"""One decorator session, run as a script: the job (JSON on stdin) names
the setup, the steps and the runs; prints one JSON line of what it saw."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fingerprint import fingerprint

job = json.loads(sys.stdin.read())
config = job["config"]
#: The code objects of the decorated functions: a step that enters one ran it.
bodies: set = set()
entered: list = []


def _watch(frame, event, _arg):
    if event == "call" and frame.f_code in bodies:
        entered.append(frame.f_code)


def _remember(fn):
    code = getattr(fn, "__code__", None)
    if code is not None:
        bodies.add(code)
    return fn


if config is None:

    def cache(fn=None, **_kwargs):
        return _remember(fn) if fn is not None else _remember

else:
    import cash

    if config == "floors0":
        import cash.notebook.call_unit as _cu

        _cu._COST_FLOOR_S = 0

    def cache(fn=None, **kwargs):
        if fn is None:
            return lambda f: cash.cache(_remember(f), **kwargs)
        return cash.cache(_remember(fn), **kwargs)


ns = {"__name__": "__main__", "cache": cache}
exec(compile(job["setup"], "<setup>", "exec"), ns)
seen, statuses = {}, {}
for run_name in job["runs"]:
    out = seen.setdefault(run_name, {})
    for index, step in enumerate(job["steps"]):
        tree = ast.parse(step)
        printed = io.StringIO()
        entered.clear()
        sys.setprofile(_watch)
        try:
            with contextlib.redirect_stdout(printed):
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    value = eval(compile(ast.Expression(tree.body[0].value), "<step>", "eval"), ns)
                    out[f"step {index} value"] = fingerprint(value)
                else:
                    exec(compile(tree, "<step>", "exec"), ns)
        except Exception as exc:
            out[f"step {index} value"] = f"raised {type(exc).__name__}: {exc}"
        finally:
            sys.setprofile(None)
        if config is not None:
            statuses.setdefault(run_name, {})[index] = ["ran"] if entered else ["-"]
        if printed.getvalue():
            out[f"step {index} output"] = printed.getvalue()
print(json.dumps({"seen": seen, "statuses": statuses}))
