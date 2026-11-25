"""A type checker reads ``@cash.cache`` the way the docs show it.

cash ships ``py.typed``, so mypy and pyright (the engine behind VS Code's
Pylance) trust its annotations. The sample below is the documented use: the
decorator bare and with options, on a function, a method and a coroutine,
and the three methods the decorated function gets. A checker must accept
every line but the ones marked ``# wrong``, and must flag exactly those: a
decorated function that lost its signature would pass them.

The checkers are not in the dev extra (pyright downloads Node). Without
them the tests skip; CI's docs job installs both and sets
``CASH_TEST_REQUIRE_TYPE_CHECKERS=1``, which turns a skip into a failure.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from tests._scripts import CASH_SRC, run_tied

SAMPLE = """\
import cash
from cash import Cash


@cash.cache
def render(x: int) -> str:
    return str(x)


render(1).upper()
render("one")  # wrong
render.cache_info()["hits"]
render.cache_clear()
render.explain(1).would_hit
render.__wrapped__(1)


@cash.cache(ttl=60, ignore=["verbose"])
def fit(x: int, *, verbose: bool = False) -> float:
    return float(x)


fit(1, verbose=True).is_integer()
fit(1, verbose="yes")  # wrong
fit.explain(1)

app = Cash()


class Model:
    @app.cache
    def predict(self, x: int) -> int:
        return x

    @cash.cache(ttl=5)
    async def fetch(self, url: str) -> bytes:
        return b""


Model().predict(1).bit_length()
Model().predict("one")  # wrong
Model.predict(Model(), 1)
Model().predict.cache_info()


async def main() -> None:
    body: bytes = await Model().fetch("u")
    await Model().fetch(1)  # wrong


def typed(f: cash.CachedCallable[[int], str]) -> None:
    f.cache_clear()


typed(render)
cash.show_stats()
cash.register_hasher(int, lambda value: str(value))
"""

WRONG_LINES = {n for n, line in enumerate(SAMPLE.splitlines(), 1) if line.endswith("# wrong")}


def _checker(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        if os.environ.get("CASH_TEST_REQUIRE_TYPE_CHECKERS"):
            pytest.fail(f"{name} is not installed, and CASH_TEST_REQUIRE_TYPE_CHECKERS is set")
        pytest.skip(f"{name} is not installed")
    return path


def _error_lines(output: str, pattern: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(pattern, output)}


def test_the_sample_marks_some_lines_wrong():
    assert len(WRONG_LINES) == 4


@pytest.mark.timeout(300)
def test_mypy_keeps_the_signature_and_knows_the_cache_methods(tmp_path: Path):
    mypy = _checker("mypy")
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    done = run_tied(
        [
            mypy,
            "--python-executable",
            sys.executable,
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            # Errors in the sample only: cash's own source is read for its
            # types, not checked here.
            "--follow-imports=silent",
            "--ignore-missing-imports",
            "sample.py",
        ],
        timeout=280,
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": CASH_SRC},
    )
    assert _error_lines(done.stdout, r"sample\.py:(\d+): error") == WRONG_LINES, done.stdout + done.stderr


@pytest.mark.timeout(300)
def test_pyright_accepts_the_decorator_line_and_the_cache_methods(tmp_path: Path):
    pyright = _checker("pyright")
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "pyrightconfig.json").write_text(
        '{"extraPaths": [%s], "reportMissingImports": false, "reportMissingModuleSource": false}'
        % ('"' + CASH_SRC.replace("\\", "/") + '"'),
        encoding="utf-8",
    )
    done = run_tied([pyright, "sample.py"], timeout=280, cwd=tmp_path, env=dict(os.environ))
    assert _error_lines(done.stdout, r"sample\.py:(\d+):\d+ - error") == WRONG_LINES, done.stdout + done.stderr
