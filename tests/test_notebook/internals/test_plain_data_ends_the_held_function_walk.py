"""The walk for the notebook functions a value holds (`held_notebook_code`)
ends at its first question when the value is JSON-like data alone (exact
dicts, lists and tuples over strings, numbers and dates), found complete
with nothing: such data holds no function, and one pass at C speed says so.

A million ``{'i': i, 'v': [i, i + 1]}`` records were walked object by
object (0.75 s at the first statement reading them) as soon as the
notebook defined any function; without one, every statement reading them
scanned every object the process tracks again (0.1-0.2 s each), since the
answer "no function exists" holds only until new code runs. A function
kept in one record is still found.
"""

from __future__ import annotations

import uuid

from cash.notebook import held_functions as hf


def _namespace(defines: bool) -> dict:
    namespace: dict = {"__name__": "__main__"}
    if defines:
        exec("def f(x):\n    return x + 1\n", namespace)
    return namespace


def _records(n: int = 30_000) -> list:
    assert n > hf.EXISTENCE_CHECK_AT
    return [{"i": i, "v": [i, i + 1], "when": ("a", 2.5)} for i in range(n)]


def _lineage() -> str:
    return f"lineage-{uuid.uuid4().hex}"


def test_records_beside_a_notebook_function_are_not_walked(monkeypatch):
    namespace, records = _namespace(defines=True), _records()
    levels = [0]
    next_level = hf._next_level

    def counted(children, seen):
        levels[0] += 1
        return next_level(children, seen)

    monkeypatch.setattr(hf, "_next_level", counted)
    held = hf.held_notebook_code(records, namespace, _lineage())
    assert held.codes == () and held.complete and held.module_classes == ()
    assert levels[0] == 0


def test_records_read_again_ask_nothing_of_the_heap(monkeypatch):
    # As in a notebook importing no module of the user's (the test files
    # here are such modules).
    monkeypatch.setattr(hf, "module_class_objects_exist", lambda: False)
    namespace, records, lineage = _namespace(defines=False), _records(), _lineage()
    assert hf.held_notebook_code(records, namespace, lineage).codes == ()
    asked = [0]

    def existence(namespace):
        asked[0] += 1
        return False

    monkeypatch.setattr(hf, "notebook_functions_exist", existence)
    assert hf.held_notebook_code(records, namespace, lineage).codes == ()
    assert asked[0] == 0


def test_a_function_kept_in_one_record_is_still_found():
    namespace, records = _namespace(defines=True), _records()
    records[12_345]["v"].append(namespace["f"])
    held = hf.held_notebook_code(records, namespace, _lineage())
    assert held.codes == (namespace["f"].__code__,)
    assert held.complete
