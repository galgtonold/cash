"""A read that failed because the file was missing is an input like any other.

Found while attacking the decorator before round 26: ``docs/decorator.md``
promises "A file that was **not** there counts as well", and the
``os.path.exists`` spelling does record it -- but the equally common

    try:
        return open(path).read()
    except FileNotFoundError:
        return DEFAULT

did not, so the default was served for ever after the file appeared.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import os, time, json
    import pandas as pd
    import cash
    cash.configure(cache_dir=CACHE_DIR, disable=os.environ.get("NOCACHE") == "1")
    F = PATH

    @cash.cache
    def with_try(n):
        time.sleep(0.3)
        try:
            with open(F) as fh:
                return fh.read().strip()
        except FileNotFoundError:
            return "MISSING"

    @cash.cache
    def with_pandas(n):
        time.sleep(0.3)
        try:
            return pd.read_csv(F).iloc[0, 0]
        except FileNotFoundError:
            return "MISSING"

    @cash.cache
    def with_pathlib(n):
        time.sleep(0.3)
        from pathlib import Path
        try:
            return Path(F).read_text().strip()
        except FileNotFoundError:
            return "MISSING"

    print(json.dumps([with_try(1), str(with_pandas(1)), with_pathlib(1)]))
""")


@pytest.mark.timeout(300)
def test_the_file_appearing_invalidates_each_spelling(tmp_path):
    data = tmp_path / "cfg.csv"
    script = tmp_path / "run.py"
    script.write_text(
        PROGRAM.replace("CACHE_DIR", repr(str(tmp_path / ".cash"))).replace("PATH", repr(str(data))), encoding="utf-8"
    )

    def run():
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path)
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert run() == ["MISSING", "MISSING", "MISSING"]
    data.write_text("value\nNOWTHERE\n", encoding="utf-8")
    assert run() == ["value\nNOWTHERE", "NOWTHERE", "value\nNOWTHERE"]
