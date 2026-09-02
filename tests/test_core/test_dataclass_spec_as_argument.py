"""A dataclass passed as an output SPECIFICATION must key on its declaration.

    fields = extract(document, InvoiceFields)

This is how every structured-output library is used, and the field descriptions
in such a class are not documentation -- they are the instruction sent to the
model. Change one and the answer changes.

``@dataclass`` moves that declaration off the class attribute into
``__dataclass_fields__``; the attribute left behind is only the default value.
``__dataclass_fields__`` cannot be pickled -- ``Field.metadata`` is a
``mappingproxy`` -- so the generic member fold in ``_class_surface_parts``
caught the ``TypeError``, folded nothing, and dropped it without a word.

Measured before the fix: rewriting a field description left the digest
unchanged and the cached answer, built from the OLD description, was served
back. The class docstring and adding a NEW field both invalidated (they move
``__doc__`` and ``__init__``), which is what made the hole easy to miss -- the
two edits a person tries first both behave.

The specs are written to a FILE and re-imported rather than redefined inline.
An inline redefinition produces class objects that already differ in ways the
digest picks up, so the first version of this test passed against the unfixed
code and proved nothing.
"""
from __future__ import annotations

import importlib
import sys
import textwrap
import warnings

import pytest

from cash import Cash

MODULE = """
from dataclasses import dataclass, field

@dataclass
class Invoice:
    \"\"\"{doc}\"\"\"
    vendor: str = field(default="", metadata={{"desc": "Who issued it"}})
    total: str = field(default="", metadata={{"desc": "{total_desc}"}})
{extra}
"""


@pytest.fixture
def spec_module(tmp_path, monkeypatch):
    """Write, import, and re-write a spec module -- the real edit cycle."""
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "spec_under_test.py"

    def write(total_desc="Grand total", doc="Invoice fields.", extra=""):
        path.write_text(
            MODULE.format(doc=doc, total_desc=total_desc,
                          extra=textwrap.indent(extra, "    ")),
            encoding="utf-8",
        )
        sys.modules.pop("spec_under_test", None)
        importlib.invalidate_caches()
        return importlib.import_module("spec_under_test").Invoice

    yield write
    sys.modules.pop("spec_under_test", None)


@pytest.fixture
def render(tmp_path):
    """A cached function that renders the spec the way a prompt builder would."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache
    def build(spec: type) -> str:
        from dataclasses import fields
        return " | ".join(f"{f.name}:{f.metadata['desc']}" for f in fields(spec))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield build


def test_rewriting_a_field_description_invalidates(spec_module, render):
    """The description is prompt text. Serving the old answer is a wrong answer."""
    first = render(spec_module(total_desc="Grand total"))
    second = render(spec_module(total_desc="Grand total INCLUDING tax"))
    assert first == "vendor:Who issued it | total:Grand total"
    assert second == "vendor:Who issued it | total:Grand total INCLUDING tax", (
        "served an answer built from the previous field description"
    )


def test_adding_a_field_invalidates(spec_module, render):
    """Control arm -- this one always worked, through __init__."""
    before = render(spec_module())
    after = render(spec_module(
        extra='currency: str = field(default="", metadata={"desc": "ISO code"})'))
    assert "currency" not in before
    assert "currency" in after


def test_a_comment_only_edit_still_hits(spec_module, render):
    """The other half of the requirement. Folding the declaration must not
    resurrect comment sensitivity -- re-running a corpus because somebody
    tidied an indent is its own kind of wrong."""
    first = render(spec_module())
    second = render(spec_module(extra="# a comment that changes nothing"))
    info = render.cache_info()
    assert first == second
    assert info["hits"] >= 1, info


def test_two_specs_differing_only_in_metadata_do_not_collide(spec_module, render):
    assert render(spec_module(total_desc="alpha")) != render(
        spec_module(total_desc="beta"))
