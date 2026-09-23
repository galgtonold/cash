"""When may a loop header be evaluated a second time?

The single-unit fast path runs a loop from source, so its header is evaluated
twice: once to measure the loop, once to run it. That is only correct when the
second evaluation yields the same iteration.

Two defects, found together:

* **Too strict** -- `len` was not on the list of pure builtins, so the
  commonest header in Python, ``range(0, len(frame), STEP)``, was refused the
  fast path and every iteration was decomposed: 16.9 s cached against
  1.0 s uncached, BLOCKING, and slower again on a re-run.
* **Too loose** -- the one-shot check looked only at the RESULT. ``sorted(g)``
  returns a list, so the first evaluation drained ``g`` and the loop then ran
  zero times. Silent, first run, no cache involved.
"""

import ast
import types

import pytest

from cash.notebook.control_structures.for_handler import ForLoopHandler


def _handler(user_ns):
    shell = types.SimpleNamespace(user_ns=user_ns)
    return ForLoopHandler(shell, statement_processor=None, debug=False, dispatcher=None)


def _safe(header, user_ns):
    node = ast.parse("for x in %s:\n    pass" % header).body[0]
    iterable = eval(header, dict(user_ns), dict(user_ns))
    return _handler(user_ns)._iter_header_safe_to_reevaluate(node.iter, iterable)


class TestTooStrict:
    """Headers that are safe and were refused."""

    def test_a_range_bounded_by_len(self):
        """The reported header, verbatim."""
        ns = {"frame": list(range(3131)), "STEP": 5}
        assert _safe("range(0, len(frame), STEP)", ns), (
            "len() cannot drain or mutate anything; refusing this header is "
            "what decomposed 627 iterations one statement at a time"
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
        assert _safe(header, {"n": 300})


class TestTooLoose:
    """Headers that were allowed and must not be."""

    @pytest.mark.parametrize("header", ["sorted(g)", "list(g)", "tuple(g)", "set(g)", "range(len(list(g)))"])
    def test_a_generator_inside_a_container_call(self, header):
        ns = {"g": (i for i in range(400))}
        assert not _safe(header, ns), (
            "the first evaluation drains g, so the second -- the one that runs the loop -- iterates nothing"
        )

    def test_a_file_object_read_through_a_method(self, tmp_path):
        path = tmp_path / "rows.txt"
        path.write_text("a\nb\n", encoding="utf-8")
        with open(path, encoding="utf-8") as fh:
            assert not _safe("range(len(fh.readlines()))", {"fh": fh})

    def test_a_shadowed_builtin_is_not_the_builtin(self):
        """A notebook's own `len` gets no benefit of the doubt from its name."""
        calls = []

        def len(x):  # noqa: A001 - the point of the test
            calls.append(x)
            return 3

        assert not _safe("range(len(data))", {"len": len, "data": [1, 2, 3]})


class TestUnchanged:
    """What was already right must stay right."""

    @pytest.mark.parametrize(
        "header,ns",
        [
            ("range(n)", {"n": 300}),
            ("sorted(d.items())", {"d": {i: i for i in range(300)}}),
            ("list(rows)", {"rows": list(range(300))}),
            ("zip(a, b)", {"a": [1], "b": [2]}),
        ],
    )
    def test_re_iterable_headers_stay_on_the_fast_path(self, header, ns):
        # zip() returns a self-iterator, and was refused on that RESULT. But the
        # header CALLS zip, so evaluating it again builds a fresh zip over the
        # same two lists -- as `df.itertuples()` builds a fresh iterator, which
        # the same rule kept a real 631-iteration loop off the fast path for.
        # What still refuses is a header over a STORED iterator:
        # see test_zip_over_a_stored_iterator_is_still_refused.
        assert _safe(header, ns)

    def test_zip_over_a_stored_iterator_is_still_refused(self):
        assert not _safe("zip(a, g)", {"a": [1, 2], "g": iter([3, 4])})

    def test_an_unknown_call_is_still_refused(self):
        assert not _safe("drain()", {"drain": lambda: [1, 2, 3]})

    def test_a_bare_generator_is_still_refused(self):
        assert not _safe("g", {"g": (i for i in range(10))})
