"""Functions held in another module's tables, or in a class attribute, key the
module constants they read.

``import steps; steps.STEPS`` (or ``steps.HANDLERS["x"]``, a pipeline object
``steps.PIPE``, a class registry ``steps.MODELS["lin"]()``) and a list held
as a class attribute (``Pipeline.DEFAULT_STEPS``) keyed the table by its
pickle, which names the functions: editing the ``RATE`` they read served the
old result, while ``from steps import STEPS`` recomputed.
"""

from __future__ import annotations

import pytest

from . import _steps_module as steps


class LocalPipeline:
    DEFAULT_STEPS = [steps.scale]


@pytest.fixture
def rate(monkeypatch):
    def set_rate(value):
        monkeypatch.setattr(steps, "RATE", value)

    return set_rate


def run_module_attr(x):
    for step in steps.STEPS:
        x = step(x)
    return x


def dispatch_module_attr(x):
    return steps.HANDLERS["scale"](x)


def pipeline_object(x):
    return steps.PIPE.run(x)


def class_attr_list(x):
    return steps.Pipeline.DEFAULT_STEPS[0](x)


def local_class_attr_list(x):
    return LocalPipeline.DEFAULT_STEPS[0](x)


def class_registry(x):
    return steps.MODELS["lin"]().predict(x)


@pytest.mark.parametrize(
    "body",
    [run_module_attr, dispatch_module_attr, pipeline_object, class_attr_list, local_class_attr_list, class_registry],
)
def test_editing_the_constant_recomputes(cash_instance, rate, body):
    f = cash_instance.cache(body)
    assert f(10) == 20
    assert f.explain(10).would_hit
    rate(3)
    assert f(10) == 30
