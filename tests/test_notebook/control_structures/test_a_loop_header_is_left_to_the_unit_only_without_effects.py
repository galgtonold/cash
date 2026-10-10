"""When may a long loop's header be left to the unit that runs the loop?

A header the policy can size from its text (``range(n)``,
``tqdm(list(zip(a, b)))``) is not evaluated before the loop runs as one unit:
the unit evaluates it, once, as plain Python does. Building it first only to
size the loop doubled the work of a 2M-pair ``list(zip(...))``.

The unit only runs on a cache miss, so such a header must have no effect a
hit would lose. Everything else is evaluated first, and the unit iterates that
value (see ``test_a_long_loop_evaluates_its_header_once``).

Defects pinned here:

* **Too strict** -- `len` was not on the list of pure builtins, so the
  commonest header in Python, ``range(0, len(frame), STEP)``, was evaluated
  for nothing.
* **Too loose** -- ``sorted(g)`` drains ``g``; a method call
  (``np.random.permutation(n)``, ``inbox.drain()``) may draw or consume; a
  notebook function handed to ``map`` runs per item; a property runs code.
"""

import ast
import random

import pytest

from cash.notebook.control_structures.single_unit_policy import header_may_be_left_to_the_unit


def _left(header, user_ns):
    node = ast.parse("for x in %s:\n    pass" % header).body[0]
    return header_may_be_left_to_the_unit(node.iter, user_ns)


class TestTooStrict:
    """Headers with no effect, which may be left to the unit."""

    def test_a_range_bounded_by_len(self):
        """The reported header, verbatim."""
        ns = {"frame": list(range(3131)), "STEP": 5}
        assert _left("range(0, len(frame), STEP)", ns), (
            "len() cannot drain or mutate anything; refusing this header "
            "builds it once more than plain Python does"
        )

    @pytest.mark.parametrize(
        "header",
        [
            "range(min(n, 100))",
            "range(max(n, 1))",
            "range(int(n / 2))",
            "range(abs(n))",
            "range(round(n / 3))",
            "range(int(float(n)))",
        ],
    )
    def test_the_other_builtins_a_bound_is_computed_with(self, header):
        assert _left(header, {"n": 300})


class TestTooLoose:
    """Headers with an effect, which are evaluated first."""

    @pytest.mark.parametrize("header", ["sorted(g)", "list(g)", "tuple(g)", "set(g)", "range(len(list(g)))"])
    def test_a_generator_inside_a_container_call(self, header):
        ns = {"g": (i for i in range(400))}
        assert not _left(header, ns), "the evaluation drains g, which a cache hit would skip"

    def test_a_file_object_read_through_a_method(self, tmp_path):
        path = tmp_path / "rows.txt"
        path.write_text("a\nb\n", encoding="utf-8")
        with open(path, encoding="utf-8") as fh:
            assert not _left("range(len(fh.readlines()))", {"fh": fh})

    def test_a_shadowed_builtin_is_not_the_builtin(self):
        """A notebook's own `len` gets no benefit of the doubt from its name."""

        def len(x):
            return 3

        assert not _left("range(len(data))", {"len": len, "data": [1, 2, 3]})

    @pytest.mark.parametrize(
        "header",
        [
            "list(np.random.permutation(300))",
            "list(random.sample(items, 300))",
            "list(df.sample(n=300).index)",
            "sorted(inbox.drain())",
            "sorted(d.items())",
        ],
    )
    def test_a_method_call_may_draw_or_consume(self, header):
        ns = {"np": pytest.importorskip("numpy"), "random": random, "items": list(range(1000))}
        assert not _left(header, ns), "a method call may draw or change state: the header is evaluated first"

    @pytest.mark.parametrize("header", ["list(b.gen)", "list(gens['a'])", "sorted(b.gen)"])
    def test_a_stored_iterator_reached_through_an_attribute_or_a_subscript(self, header):
        class Box:
            pass

        b = Box()
        b.gen = (i for i in range(300))
        ns = {"b": b, "gens": {"a": (i for i in range(300))}}
        assert not _left(header, ns)

    @pytest.mark.parametrize("header", ["list(map(f, range(300)))", "sorted(rows, key=f)", "list(map(Row, rows))"])
    def test_a_notebook_function_handed_to_a_builtin(self, header):
        calls = []

        def f(x):
            calls.append(x)
            return x

        class Row:
            def __init__(self, x):
                calls.append(x)

        assert not _left(header, {"f": f, "Row": Row, "rows": [1, 2]}), "f's calls would be skipped on a hit"

    def test_a_property_is_not_read_to_judge_it(self):
        reads = []

        class Loader:
            @property
            def batch(self):
                reads.append(1)
                return [1, 2, 3]

        assert not _left("enumerate(loader.batch)", {"loader": Loader()})
        assert reads == [], "judging the header ran the property"


class TestUnchanged:
    """What was already right must stay right."""

    @pytest.mark.parametrize(
        "header,ns",
        [
            ("range(n)", {"n": 300}),
            ("list(rows)", {"rows": list(range(300))}),
            ("zip(a, b)", {"a": [1], "b": [2]}),
            ("enumerate(cfg.rows)", {"cfg": type("C", (), {"rows": [1, 2]})()}),
            ("list(map(str, rows))", {"rows": [1, 2]}),
            ("enumerate(table['rows'])", {"table": {"rows": [1, 2]}}),
        ],
    )
    def test_headers_with_no_effect_are_left_to_the_unit(self, header, ns):
        assert _left(header, ns)

    def test_zip_over_a_stored_iterator_is_still_refused(self):
        assert not _left("zip(a, g)", {"a": [1, 2], "g": iter([3, 4])})

    def test_an_unknown_call_is_still_refused(self):
        assert not _left("drain()", {"drain": lambda: [1, 2, 3]})

    def test_a_bare_generator_is_still_refused(self):
        assert not _left("g", {"g": (i for i in range(10))})
