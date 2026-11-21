"""The analysis a process worked out for the calls of a ``Cash(cache_dir=...)``
of its own is written into that instance's folder, even when the default
instance's folder exists too.

What waits to be written (purity reports, text answers, module scans) is
written in batches and at exit, where no call runs: the store folder was
looked up then, from the default instance's settings. While the default
folder did not exist the last folder a call used stood in for it; once it
did (a ``.cash`` a project made earlier, or one in a parent directory), the
analysis went there. Every later process looked in the instance's own
folder, found nothing, and walked every helper again.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app
    from cash.analysis import helper_walk, report_store, text_store
    walked = []
    plain = helper_walk.HelperWalk._walk_one
    def counted(self, entry, qualname):
        walked.append(qualname)
        return plain(self, entry, qualname)
    helper_walk.HelperWalk._walk_one = counted
    own = cash.Cash(cache_dir=sys.argv[2])
    top = own.cache(app.top)
    value = top(3)
    print(json.dumps({"walked": walked, "value": value, "report": report_store.status(),
                      "text": text_store.status()}))
    """
)

_APP = """
import lib


def step(x):
    return lib.leaf(x) + 1


def top(n):
    return step(n) * 2
"""

_LIB = """
FACTOR = 3


def leaf(x):
    return x * FACTOR
"""


def _run(code_dir: Path, own_dir: Path, default_dir: Path) -> dict:
    out = run_python("-c", _RUNNER, str(code_dir), str(own_dir), cwd=code_dir, cache_dir=default_dir)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_next_process_reads_the_report_from_the_own_folder(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "app.py").write_text(_APP, encoding="utf-8")
    (code / "lib.py").write_text(_LIB, encoding="utf-8")
    own = tmp_path / "own-cache"
    # The default instance's folder exists, but nothing here uses it.
    default = tmp_path / "default-cache"
    default.mkdir()
    first = _run(code, own, default)
    assert "lib.leaf" in first["walked"]
    assert first["report"]["loaded"] == 0
    second = _run(code, own, default)
    assert second["walked"] == []
    assert second["report"]["loaded"] >= 1
    assert second["text"]["loaded"] >= 1
    assert second["value"] == first["value"] == 20
    # Nothing went into the folder the program did not choose.
    assert list(default.iterdir()) == []
