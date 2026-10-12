"""A class attribute that names a library class or function is keyed by
what it names.

``Settings.dtype = np.float32`` or ``rounder = staticmethod(math.floor)``:
the class's code channels key the user's own code, and the class data fold
left every class, function and descriptor out as code, so rebinding the
attribute at run time, or picking it from the environment when the class is
made, served the old result. A plain data attribute next to it was keyed.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from tests._scripts import run_python

np = pytest.importorskip("numpy")


def test_rebinding_it_in_this_process_recomputes(cash_instance):
    class Settings:
        dtype = np.float32
        rounder = staticmethod(math.floor)

    @cash_instance.cache
    def convert(xs):
        return np.asarray(xs, dtype=Settings.dtype).dtype.name, Settings.rounder(xs[0])

    assert convert([1.5, 2.5]) == ("float32", 1)
    Settings.dtype = np.float64
    assert convert([1.5, 2.5]) == ("float64", 1)
    Settings.rounder = staticmethod(math.ceil)
    assert convert([1.5, 2.5]) == ("float64", 2)


METHODS = textwrap.dedent("""
    import numpy as np
    import cash

    class Conv:
        dtype = np.float32

        @classmethod
        @cash.cache
        def conv(cls, xs):
            return np.asarray(xs, dtype=cls.dtype).dtype.name

        @cash.cache
        def conv_m(self, xs):
            return np.asarray(xs, dtype=self.dtype).dtype.name

    print(Conv.conv([1.5]), Conv().conv_m([1.5]))
    Conv.dtype = np.float64
    print(Conv.conv([1.5]), Conv().conv_m([1.5]))
""")


def test_a_cached_method_reading_it_through_cls_or_self_recomputes(tmp_path):
    script = tmp_path / "methods.py"
    script.write_text(METHODS, encoding="utf-8")
    done = run_python(script, cwd=tmp_path, cache_dir=tmp_path / ".cash")
    assert done.stdout.split() == ["float32", "float32", "float64", "float64"], done.stderr


SCRIPT = textwrap.dedent("""
    import os
    import numpy as np
    import cash

    class K:
        DT = getattr(np, os.environ["PICKED_DTYPE"])

    @cash.cache
    def zeros(n):
        return np.zeros(n, dtype=K.DT).dtype.name

    print(zeros(2))
""")


def test_picking_it_from_the_environment_recomputes_in_the_next_run(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT, encoding="utf-8")
    cache_dir = tmp_path / ".cash"
    first = run_python(script, cwd=tmp_path, cache_dir=cache_dir, env={"PICKED_DTYPE": "float32"})
    second = run_python(script, cwd=tmp_path, cache_dir=cache_dir, env={"PICKED_DTYPE": "float64"})
    assert first.stdout.split() == ["float32"]
    assert second.stdout.split() == ["float64"], second.stderr
