"""The purity reports and text answers a process worked out for the calls of
a ``Cash(cache_dir=...)`` of its own are written at exit into that
instance's folder, so the next process reads them instead of walking the
helpers again.

They are written at exit (`report_store.flush`, `text_store.flush`), where
no call runs: the store folder was looked up from the default instance's
settings, not the instance whose calls worked them out, so they went to a
folder that does not exist and were dropped. Every process walked every
helper of such an instance again, though nothing had changed.
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


def _run(code_dir: Path, own_dir: Path) -> dict:
    # The default instance's folder (``cache_dir``) is not the one the
    # program's own instance writes to, and is never made.
    out = run_python("-c", _RUNNER, str(code_dir), str(own_dir), cwd=code_dir, cache_dir=code_dir / "default-unused")
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_next_process_reads_the_report_of_an_own_instance(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "app.py").write_text(_APP, encoding="utf-8")
    (code / "lib.py").write_text(_LIB, encoding="utf-8")
    own = tmp_path / "own-cache"
    first = _run(code, own)
    assert "lib.leaf" in first["walked"]
    assert first["report"]["loaded"] == 0
    second = _run(code, own)
    assert second["walked"] == []
    assert second["report"]["loaded"] >= 1
    assert second["text"]["loaded"] >= 1
    assert second["value"] == first["value"] == 20
    assert not (code / "default-unused").exists()
