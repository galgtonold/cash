"""The variable sharing parts with a library object is found without reading
the heap, and the climb up from those parts does not go back through the
library object's own insides.

``optimizer = torch.optim.SGD(model.parameters())`` holds ``model``'s
parameters inside its ``param_groups``. Each check of a statement that left
``optimizer`` or ``model`` climbed up ``gc.get_referrers`` from the
parameters: a read of the whole heap per level, up the optimizer's own group
dict, list and ``__dict__`` and through ``model``'s layers, four or five
reads (70 ms each on a PyTorch notebook) for every cell that touched either.
The variables are found going down from the namespace first; the count taken
again with them joined still decides, so a holder the guess misses is
climbed to as before. Pins the work, not the behaviour: a refactor may
expect this to fail.
"""

from __future__ import annotations

import sys

import pytest

import cash.notebook.shared_objects as shared_objects

#: A library's own classes: not the notebook's, an estimator or a user module's.
Parameter = type("Parameter", (), {"__module__": "somelib.nn"})
Linear = type("Linear", (), {"__module__": "somelib.nn"})
SGD = type("SGD", (), {"__module__": "somelib.optim"})


class Model:
    """The notebook's own module of layers: the share check reads its attributes."""


@pytest.fixture
def heap_reads(monkeypatch):
    reads = []
    real = shared_objects._referrers

    def counted(objs, stop):
        reads.append(len(objs))
        return real(objs, stop)

    monkeypatch.setattr(shared_objects, "_referrers", counted)
    return reads


def _namespace() -> dict:
    model = Model()
    model.fc = Linear()
    model.fc.weight = Parameter()
    model.fc.bias = Parameter()
    model.fc._parameters = {"weight": model.fc.weight, "bias": model.fc.bias}
    optimizer = SGD()
    optimizer.param_groups = [{"params": [model.fc.weight, model.fc.bias], "lr": 0.1}]
    optimizer.state = {}
    return {"model": model, "optimizer": optimizer}


@pytest.mark.parametrize(("output", "holder"), [("optimizer", "model"), ("model", "optimizer")])
def test_the_variable_sharing_a_library_objects_parts_joins_with_no_read(heap_reads, output, holder):
    user_ns = _namespace()
    holders, shared = shared_objects.share_group([output], {output: user_ns[output]}, user_ns)
    assert set(holders) == {holder}
    assert shared == set()
    assert heap_reads == []  # 4 reads before


def test_the_climb_does_not_go_back_up_the_library_objects_parts(heap_reads, monkeypatch):
    # Without the guess: up from the parameters to the layer's dicts, the
    # layer and the model (3 reads), not on up the optimizer's group dict,
    # its list and its ``__dict__`` back to the optimizer (4 reads before).
    monkeypatch.setattr(shared_objects, "_variables_reaching", lambda *args: set())
    user_ns = _namespace()
    holders, shared = shared_objects.share_group(["optimizer"], {"optimizer": user_ns["optimizer"]}, user_ns)
    assert set(holders) == {"model"}
    assert shared == set()
    # Python 3.10 holds an object's attributes in a ``__dict__`` of its own,
    # one more step on the way up from the layer to the model.
    assert len(heap_reads) <= (3 if sys.version_info >= (3, 11) else 4)


def test_a_holder_the_guess_cannot_reach_is_still_climbed_to(heap_reads):
    user_ns = _namespace()
    # Held where no walk down from a variable goes: past the guess's depth,
    # in nested lists.
    user_ns["deep"] = [user_ns["optimizer"].param_groups[0]["params"][0]]
    for _ in range(shared_objects._REACH_DEPTH):
        user_ns["deep"] = [user_ns["deep"]]
    holders, shared = shared_objects.share_group(["optimizer"], {"optimizer": user_ns["optimizer"]}, user_ns)
    assert set(holders) == {"model", "deep"}
    assert shared == set()
    assert heap_reads  # the count after the guess still saw the deep holder
