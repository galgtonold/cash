"""Source retrieval must degrade, never raise, when a function's file is not Python.

``inspect.getsource`` does not merely read the file — it runs ``tokenize`` over
it to find where the function's block ends. So a function whose ``co_filename``
names a file that is not valid Python end-to-end raises ``tokenize.TokenError``,
which derives straight from ``Exception``: an ``except (OSError, TypeError)``
handler does not catch it, and neither does ``except SyntaxError``.

Cash reaches for source at *decoration* time, so before the fix this aborted
``@cash.cache`` itself rather than falling back to the bytecode hash the call
sites were written to fall back to. It is not an exotic setup — it is any
literate-source workflow (Jupytext, Quarto, a docs page exec'd under its own
path, a saved REPL transcript). It surfaced as ten red pages in the docs-parity
CI job on Python 3.12.

The repro compiles a function with a ``.md`` path as its filename, which is
exactly what the docs harness does.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import cash
from cash.exceptions import SOURCE_RETRIEVAL_ERRORS
from cash.notebook.analysis import CodeAnalyzer


NON_PYTHON_SOURCE = "Cash doesn't tokenize this line as Python.\nMore prose.\n"


def _func_from_non_python_file(tmp_path: Path, name: str = "page.md"):
    """Build a real function whose ``co_filename`` points at a non-Python file.

    The apostrophe in the prose is what breaks the tokenizer: it opens a string
    literal that never closes.
    """
    doc = tmp_path / name
    doc.write_text(NON_PYTHON_SOURCE, encoding="utf-8")

    namespace: dict = {}
    exec(compile("def demo(x):\n    return x * 2\n", str(doc), "exec"), namespace)
    return namespace["demo"]


class TestSourceRetrievalDegrades:
    def test_repro_is_valid_getsource_really_raises(self, tmp_path):
        """Guard the guard: if this stops raising, the tests below prove nothing."""
        demo = _func_from_non_python_file(tmp_path)

        with pytest.raises(Exception) as exc_info:
            inspect.getsource(demo)

        # The point of the bug: not an OSError, not a SyntaxError.
        assert not isinstance(exc_info.value, (OSError, SyntaxError))
        assert isinstance(exc_info.value, SOURCE_RETRIEVAL_ERRORS)

    def test_get_source_hash_falls_back_instead_of_raising(self, tmp_path):
        demo = _func_from_non_python_file(tmp_path)

        digest = CodeAnalyzer.get_source_hash(demo)

        assert isinstance(digest, str) and len(digest) == 64

    def test_hash_is_stable_across_calls(self, tmp_path):
        """The bytecode fallback must be deterministic or every call is a miss."""
        demo = _func_from_non_python_file(tmp_path)

        assert CodeAnalyzer.get_source_hash(demo) == CodeAnalyzer.get_source_hash(demo)

    def test_hash_still_distinguishes_different_bodies(self, tmp_path):
        """Degrading must not collapse distinct functions onto one key."""
        doc = tmp_path / "page.md"
        doc.write_text(NON_PYTHON_SOURCE, encoding="utf-8")

        ns_a: dict = {}
        ns_b: dict = {}
        exec(compile("def demo(x):\n    return x * 2\n", str(doc), "exec"), ns_a)
        exec(compile("def demo(x):\n    return x * 3\n", str(doc), "exec"), ns_b)

        assert CodeAnalyzer.get_source_hash(ns_a["demo"]) != CodeAnalyzer.get_source_hash(
            ns_b["demo"]
        )

    def test_find_called_functions_degrades_to_empty(self, tmp_path):
        demo = _func_from_non_python_file(tmp_path)

        assert CodeAnalyzer.find_called_functions(demo) == set()


class TestDecoratorSurvivesNonPythonSource:
    """The end-to-end shape: decoration is where users hit this."""

    def test_cache_decoration_does_not_raise(self, tmp_path):
        doc = tmp_path / "page.md"
        doc.write_text(NON_PYTHON_SOURCE, encoding="utf-8")

        instance = cash.Cash(cache_dir=str(tmp_path / "cache"))
        namespace: dict = {"cash_instance": instance}
        source = (
            "@cash_instance.cache\n"
            "def compute(x):\n"
            "    calls.append(x)\n"
            "    return x * 2\n"
        )
        namespace["calls"] = []

        # Before the fix this raised TokenError out of the decorator.
        exec(compile(source, str(doc), "exec"), namespace)

        compute = namespace["compute"]
        assert compute(21) == 42

    def test_cached_function_still_hits(self, tmp_path):
        """Degrading the *hash source* must not degrade caching itself."""
        doc = tmp_path / "page.md"
        doc.write_text(NON_PYTHON_SOURCE, encoding="utf-8")

        instance = cash.Cash(cache_dir=str(tmp_path / "cache"))
        namespace: dict = {"cash_instance": instance, "calls": []}
        source = (
            "@cash_instance.cache\n"
            "def compute(x):\n"
            "    calls.append(x)\n"
            "    return x * 2\n"
        )
        exec(compile(source, str(doc), "exec"), namespace)

        compute = namespace["compute"]
        assert compute(21) == 42
        assert compute(21) == 42
        assert namespace["calls"] == [21], "second call should have hit the cache"


class TestErrorTupleContract:
    """These entries are load-bearing; narrowing the tuple is how the bug got in."""

    def test_covers_tokenize_error(self):
        from tokenize import TokenError

        assert TokenError in SOURCE_RETRIEVAL_ERRORS

    @pytest.mark.parametrize(
        "exc", [OSError, TypeError, SyntaxError, IndentationError, UnicodeDecodeError]
    )
    def test_covers_known_failure_modes(self, exc):
        assert issubclass(exc, SOURCE_RETRIEVAL_ERRORS)
