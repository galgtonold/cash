"""Reading a global by string, or a class held in one, is still a dependency.

Found while stress-testing the decorator: `globals()["K"]` and
`vars(conf)["K"]` never reached the key, so editing the constant served the old
answer (20 where an uncached run gives 500). A dict of CLASSES had the same
hole, while the identical dict of functions was followed -- `TABLE = {"fast":
impl.Fast}` with `TABLE["fast"]().run(x)`.
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

from tests._scripts import run_python

MOD = textwrap.dedent("""
    import time
    import cash
    cash.configure(cache_dir=CACHE)
    import conf
    import impl

    K = MULT
    TABLE = {"fast": impl.Fast}
    MAKERS = {"fast": impl.build}

    @cash.cache
    def by_globals(x):
        return x * globals()["K"]

    @cash.cache
    def by_vars(x):
        return x * vars(conf)["K"]

    @cash.cache
    def by_class_table(x):
        return TABLE["fast"]().run(x)

    @cash.cache
    def by_function_table(x):
        return MAKERS["fast"](x)
""")
IMPL = "class Fast:\n    def run(self, x):\n        return x * MULT\n\n\ndef build(x):\n    return x * MULT\n"
RUN = "import mod\nprint(mod.by_globals(10), mod.by_vars(10), mod.by_class_table(10), mod.by_function_table(10))\n"


def _write(project, mult):
    (project / "mod.py").write_text(
        MOD.replace("MULT", str(mult)).replace("CACHE", repr(str(project / ".cash"))), encoding="utf-8"
    )
    (project / "impl.py").write_text(IMPL.replace("MULT", str(mult)), encoding="utf-8")
    (project / "conf.py").write_text(f"K = {mult}\n", encoding="utf-8")
    (project / "run.py").write_text(RUN, encoding="utf-8")


def _run(project):
    shutil.rmtree(project / "__pycache__", ignore_errors=True)
    done = run_python("run.py", cwd=project, timeout=180)
    return done.stdout.strip()


@pytest.mark.timeout(300)
def test_editing_what_a_string_named_global_reaches_recomputes(tmp_path):
    _write(tmp_path, 2)
    assert _run(tmp_path) == "20 20 20 20"
    _write(tmp_path, 50)
    assert _run(tmp_path) == "500 500 500 500"
