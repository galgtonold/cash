"""A pydantic model passed as an output spec must cache, and must key on its
field declarations.

Pydantic is what most people reach for when they want structured output, so a
`BaseModel` handed to a cached extraction function is the common case. Two
things have to hold, and before this fix only the first did:

* editing a field `description` -- the instruction sent to the model -- must
  invalidate;
* an UNEDITED model must hit across processes.

Pydantic v2 compiles `__pydantic_core_schema__`, `__pydantic_serializer__` and
`__pydantic_validator__` onto every model. The class surface folded all three,
and their digest is different in every process -- measured: an unedited model
ran 2 times across 2 runs where the equivalent dataclass ran 1. Tracked, never
cached. Safe and useless.

They are skipped now, and `model_fields` is folded in their place: same
declarations, stable digest. `model_fields` is a property on the class, so
`vars(cls)` never saw it -- which is why dropping the compiled trio without
adding it would have made descriptions invisible instead, the stale-answer bug
in `test_dataclass_spec_as_argument.py` all over again.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash

pydantic = pytest.importorskip("pydantic")


def _model(desc: str = "Grand total", *, extra: bool = False, doc: str = "Invoice."):
    fields = {
        "__annotations__": {"vendor": str, "total": str},
        "vendor": pydantic.Field(description="Who issued it"),
        "total": pydantic.Field(description=desc),
        "__doc__": doc,
    }
    if extra:
        fields["__annotations__"]["currency"] = str
        fields["currency"] = pydantic.Field(description="ISO code")
    return type("Invoice", (pydantic.BaseModel,), fields)


@pytest.fixture
def render(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache
    def build(spec: type) -> str:
        return " | ".join(f"{n}:{f.description}" for n, f in spec.model_fields.items())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield build


def test_two_identical_models_get_the_same_surface(tmp_path):
    """The half that was broken, as a unit test.

    Two separately-built classes with identical declarations stand in for the
    same model re-imported in a fresh process. Their surfaces must match; with
    the compiled trio folded they did not, which is why nothing ever hit.
    """
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    first = c._class_surface_parts(_model())
    second = c._class_surface_parts(_model())

    names = [n for _, n, _ in first]
    assert not any(n.startswith("__pydantic_core") or n.startswith("__pydantic_valid")
                   for n in names), names
    assert any(n.startswith("__pydantic_field__:") for n in names), names
    assert first == second, "identical declarations must give an identical surface"


def test_rewriting_a_field_description_invalidates(render):
    """The description is prompt text. Serving the old answer is a wrong answer."""
    first = render(_model("Grand total"))
    second = render(_model("Grand total INCLUDING tax"))
    assert first == "vendor:Who issued it | total:Grand total"
    assert second == "vendor:Who issued it | total:Grand total INCLUDING tax"


def test_adding_a_field_invalidates(render):
    assert "currency" not in render(_model())
    assert "currency" in render(_model(extra=True))


def test_two_models_differing_only_in_description_do_not_collide(render):
    assert render(_model("alpha")) != render(_model("beta"))


def test_a_class_with_a_hostile_model_fields_property_does_not_break(tmp_path):
    """`model_fields` is read on every class the surface walk sees, so a
    non-pydantic class that happens to define it -- as a property, raising --
    must not take the call down with it. Hashing is never allowed to be the
    thing that breaks caching."""
    class Hostile:
        @property
        def model_fields(self):
            raise RuntimeError("not yours to read")

        def work(self):
            return 1

    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache
    def use(obj) -> int:
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert use(Hostile()) == 1
    assert c._class_surface_parts(Hostile) is not None
