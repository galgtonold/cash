"""A class reached only through a type annotation is a code dependency.

Annotations name classes, so shouldn't they count? Cash
ignored them on the reasoning that ``value: B`` never runs. For a whole family
of libraries it does: pydantic validates ``A``'s field ``b: B`` by running
``B``'s validators, and anything built on ``typing.get_type_hints`` (cattrs,
dacite, FastAPI, a ten-line builder of your own) constructs ``B`` from ``A``'s
hints. ``B`` is named nowhere in the cached function, so editing its validator
served the result from before the edit.

Each case runs in fresh interpreters on one cache: the edit must recompute,
and a run without an edit must hit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

PYDANTIC_MODELS = textwrap.dedent("""
    from pydantic import BaseModel, field_validator

    class B(BaseModel):
        v: int

        @field_validator("v")
        @classmethod
        def scale(cls, x):
            return x * {FACTOR}

    class A(BaseModel):
        b: B
""")

PYDANTIC_MAIN = textwrap.dedent("""
    import sys, time
    import cash
    from models import A

    @cash.cache
    def parse(d):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.3)  # @cash:assume-safe
        return A.model_validate(d).b.v

    print(parse({"b": {"v": 2}}))
""")

HINTS_MODELS = textwrap.dedent("""
    import dataclasses
    import typing

    @dataclasses.dataclass
    class B:
        v: int

        def __post_init__(self):
            self.v = self.v * {FACTOR}

    @dataclasses.dataclass
    class A:
        b: B

    def build(cls, data):
        hints = typing.get_type_hints(cls)
        return cls(**{{k: build(hints[k], v) if dataclasses.is_dataclass(hints[k]) else v
                       for k, v in data.items()}})
""")

HINTS_MAIN = textwrap.dedent("""
    import sys, time
    import cash
    from models import A, build

    @cash.cache
    def parse(d):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.3)  # @cash:assume-safe
        return build(A, d).b.v

    print(parse({"b": {"v": 2}}))
""")


def _run(tmp_path, models, main, factor):
    (tmp_path / "models.py").write_text(models.replace("{FACTOR}", str(factor)).replace("{{", "{").replace("}}", "}"))
    (tmp_path / "main.py").write_text(main)
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip(), proc.stderr


@pytest.mark.parametrize(
    "models, main",
    [
        pytest.param(PYDANTIC_MODELS, PYDANTIC_MAIN, id="pydantic_nested_model"),
        pytest.param(HINTS_MODELS, HINTS_MAIN, id="get_type_hints_builder"),
    ],
)
def test_editing_a_class_reached_through_an_annotation_recomputes(tmp_path, models, main):
    if "pydantic" in models:
        pytest.importorskip("pydantic")
    out, err = _run(tmp_path, models, main, 10)
    assert out == "20" and "[RUN]" in err
    out, err = _run(tmp_path, models, main, 10)
    assert out == "20" and "[RUN]" not in err, "an unedited run must hit"

    out, _ = _run(tmp_path, models, main, 100)
    assert out == "200", f"served the result from before the edit: {out}"
