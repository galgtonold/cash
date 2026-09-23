"""A Cash instance is not kept alive by the process, and what it learned
stays with it.

Every instance registered its own bound methods with ``atexit``, so none was
ever collected; and warn-once marks lived on the class, so one instance's
warning silenced every other's.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref

import pytest

from cash import Cash

pytestmark = pytest.mark.core


def test_a_discarded_instance_is_collected(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False, summary=True)

    @c.cache
    def f(x):
        return x

    f(1)
    ref = weakref.ref(c)
    del c, f
    gc.collect()
    assert ref() is None


def test_a_discarded_instances_writes_still_land_at_exit(tmp_path):
    script = textwrap.dedent(
        """
        import gc, time
        from cash import Cash

        def run():
            c = Cash(cache_dir=".c", register_magic=False)

            @c.cache
            def slow(x):
                time.sleep(0.15)  # @cash:assume-safe
                return x

            slow(1)

        run()
        gc.collect()
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    p = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-2000:]
    assert os.listdir(tmp_path / ".c" / ".keys"), "the stored-key record was not written"


def test_one_instances_warning_does_not_silence_anothers(tmp_path, caplog):
    class Opaque:
        __call__ = staticmethod(abs)

    def warnings_from(c):
        caplog.clear()

        @c.cache(assume_safe=True)
        def apply(fn, x):
            return fn(x)

        apply(Opaque(), 3)
        return [r for r in caplog.records if "[KEY-OPAQUE-CALLABLE]" in r.getMessage()]

    with caplog.at_level("WARNING", logger="cash"):
        assert len(warnings_from(Cash(cache_dir=str(tmp_path / "a"), register_magic=False))) == 1
        assert len(warnings_from(Cash(cache_dir=str(tmp_path / "b"), register_magic=False))) == 1
