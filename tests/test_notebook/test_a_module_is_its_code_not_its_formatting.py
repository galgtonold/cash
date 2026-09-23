"""A module's identity is what it says, not what it looks like.

The digest of a local module lands in the lineage of every name bound from it
and in the key of every statement that reads one, so whatever it covers
re-runs work when it moves. It used to be ``sha256`` of the file's bytes, so a
comment, a blank line or a reformat re-ran everything built on the module.

Measured 2026-09-21: adding ONE comment to a module re-executed a 1.2 s call
that used a function the edit did not touch -- both for ``import lib`` and
``import lib as x``. Round 27 r27s2 hit the same thing at scale: editing one
helper re-read all 10,000 of their ticket files, 48.7 s against a 17.3 s
control, later 9.1x.

Docstrings are prose too, the module's own included, and do not count.

What must still count is the part nobody should have to think about twice:
``@cash:`` directives, which are instructions to cash --
``# @cash:assume-safe`` waives a purity check, and cash's own diagnostic says
directives are part of a function's source identity.
"""

import os

import pytest

from cash.notebook.statement import file_deps
from cash.notebook.statement.file_deps import _module_identity, read_module_source_hash

BASE = "def load(n):\n    return list(range(n))\n\ndef report(rows):\n    return len(rows)\n"


def _id(text):
    return _module_identity(text.encode("utf-8"))


class TestWhatStopsMattering:
    """Formatting. The whole point."""

    def test_a_comment_does_not_change_a_module_s_identity(self):
        assert _id(BASE) == _id("# a note to myself\n" + BASE)

    def test_a_trailing_comment_does_not_either(self):
        assert _id(BASE) == _id(BASE + "# TODO: tidy this up\n")

    def test_blank_lines_do_not(self):
        assert _id(BASE) == _id(BASE.replace("\n\n", "\n\n\n\n"))

    def test_a_docstring_does_not(self):
        assert _id(BASE) == _id(BASE.replace("def load(n):\n", 'def load(n):\n    """rows"""\n'))

    def test_nor_does_rewording_one(self):
        a = BASE.replace("def load(n):\n", 'def load(n):\n    """rows"""\n')
        b = BASE.replace("def load(n):\n", 'def load(n):\n    """The rows, n of them."""\n')
        assert _id(a) == _id(b)

    def test_nor_does_the_module_s_own(self):
        assert _id(BASE) == _id('"""Loading and reporting."""\n' + BASE)

    def test_nor_does_reflowing_a_call(self):
        wrapped = BASE.replace("    return list(range(n))", "    return list(\n        range(n)\n    )")
        assert _id(BASE) == _id(wrapped)


class TestWhatStillMatters:
    """Everything a change to the file can actually do."""

    def test_an_edit_to_a_function_body(self):
        assert _id(BASE) != _id(BASE.replace("range(n)", "range(n + 1)"))

    def test_a_new_function(self):
        assert _id(BASE) != _id(BASE + "def extra():\n    return 1\n")

    def test_a_string_the_function_returns(self):
        assert _id(BASE) != _id(BASE.replace("return len(rows)", "return 'rows: ' + str(len(rows))"))

    def test_a_cash_directive_appearing(self):
        assert _id(BASE) != _id(BASE.replace("def load(n):", "def load(n):  # @cash:assume-safe"))

    def test_a_cash_directive_moving_to_another_function(self):
        """Kept as whole lines precisely so this is visible.

        Keeping only the directive text would make the two identical, and a
        waived purity check would silently apply to the wrong function.
        """
        on_load = BASE.replace("def load(n):", "def load(n):  # @cash:assume-safe")
        on_report = BASE.replace("def report(rows):", "def report(rows):  # @cash:assume-safe")
        assert _id(on_load) != _id(on_report)

    def test_a_cash_directive_changing_its_value(self):
        a = BASE.replace("def load(n):", "def load(n):  # @cash:ttl=60")
        b = BASE.replace("def load(n):", "def load(n):  # @cash:ttl=600")
        assert _id(a) != _id(b)


class TestItNeverBreaks:
    """A file cash cannot parse must behave exactly as it did before."""

    def test_something_that_is_not_python_falls_back_to_its_bytes(self):
        raw = b"\xff\xfe this is not a python file"
        assert _module_identity(raw) == raw

    def test_a_syntax_error_falls_back_to_its_bytes(self):
        raw = b"def broken(:\n"
        assert _module_identity(raw) == raw

    def test_a_missing_file_still_returns_none(self, tmp_path):
        assert read_module_source_hash(str(tmp_path / "nope.py")) is None


class TestTheMemo:
    """Re-reading and re-parsing per statement is what this replaces."""

    def test_an_unchanged_file_keeps_its_digest(self, tmp_path):
        path = tmp_path / "memo_lib.py"
        path.write_text(BASE, encoding="utf-8")
        assert read_module_source_hash(str(path)) == read_module_source_hash(str(path))

    def test_a_changed_file_gets_a_new_one(self, tmp_path):
        path = tmp_path / "memo_lib2.py"
        path.write_text(BASE, encoding="utf-8")
        before = read_module_source_hash(str(path))
        path.write_text(BASE.replace("range(n)", "range(n + 1)"), encoding="utf-8")
        assert read_module_source_hash(str(path)) != before, (
            "the memo is keyed on the same stat cash uses to notice a module "
            "changed at all; if this fails it is holding a stale digest"
        )

    def test_dependency_files_are_folded_in(self, tmp_path):
        mod = tmp_path / "with_dep.py"
        mod.write_text(BASE, encoding="utf-8")
        dep = tmp_path / "dep.py"
        dep.write_text("HELPER = 1\n", encoding="utf-8")

        alone = read_module_source_hash(str(mod))
        with_dep = read_module_source_hash(str(mod), {str(dep)})
        assert alone != with_dep

        dep.write_text("HELPER = 2\n", encoding="utf-8")
        assert read_module_source_hash(str(mod), {str(dep)}) != with_dep

    def test_a_dependency_that_is_not_python_is_still_counted(self, tmp_path):
        """Data files sit among the dependencies; they must not be dropped."""
        mod = tmp_path / "with_data.py"
        mod.write_text(BASE, encoding="utf-8")
        data = tmp_path / "table.bin"
        data.write_bytes(b"\x00\x01\x02")

        before = read_module_source_hash(str(mod), {str(data)})
        data.write_bytes(b"\x00\x01\x03")
        assert read_module_source_hash(str(mod), {str(data)}) != before

    def test_an_edit_inside_one_timestamp_tick_is_still_seen(self, tmp_path):
        """The memo's key cannot see an edit that keeps the size and the mtime.

        On Windows the mtime moves in timer ticks (~15.6 ms), so a same-size
        rewrite moments after the first write can keep it exactly: this memo
        then served the old digest, and the tests above failed intermittently
        on Windows CI for 3.10-3.14. A user saving twice within one tick is
        the same edit. Forced here by putting the mtime back.
        """
        path = tmp_path / "racy_lib.py"
        path.write_text(BASE, encoding="utf-8")
        st = os.stat(path)
        before = read_module_source_hash(str(path))

        path.write_text(BASE.replace("range(n)", "range(9)"), encoding="utf-8")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert os.stat(path).st_size == st.st_size

        assert read_module_source_hash(str(path)) != before, (
            "same size, same mtime, new code: a digest memoised while the file "
            "was still being written must not be trusted"
        )

    def test_a_file_that_has_settled_is_memoised(self, tmp_path):
        """The other half: the rule above must not turn the memo off."""
        path = tmp_path / "settled_lib.py"
        path.write_text(BASE, encoding="utf-8")
        old = os.stat(path).st_mtime_ns - 60 * 10**9
        os.utime(path, ns=(old, old))

        read_module_source_hash(str(path))
        assert str(path) in file_deps._IDENTITY_CACHE


@pytest.mark.parametrize(
    "directive",
    [
        "# @cash:persist",
        "# @cash:no-cache",
        "# @cash:assume-safe",
        "# @cash: allow-random",
        "# @cash:nocache",
    ],
)
def test_every_spelling_of_a_directive_counts(directive):
    """Matched with the parser's own pattern, so the two cannot drift."""
    assert _id(BASE) != _id(BASE.replace("def load(n):", "def load(n):  " + directive))
