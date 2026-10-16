"""A ``functools`` memo of the user's that already held results at ``import
cash`` is named once (``KEY-MEMO-BEFORE-IMPORT``).

cash watches file reads from ``import cash`` on, so a config loader memoised
with ``lru_cache`` and called ahead of it read its file unseen: a cached
function that gets the config from the memo records no file. Quiet when the
memo is empty at import, and for a library's memos.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

CODE = "[KEY-MEMO-BEFORE-IMPORT]"

LOADER = textwrap.dedent(
    """
    import functools

    @functools.lru_cache
    def load_cfg():
        with open("config.txt", encoding="utf-8") as fh:
            return fh.read()

    @functools.cache
    def unused():
        return 1
    """
)


def _project(tmp_path, main: str):
    (tmp_path / "config.txt").write_text("a=1\n", encoding="utf-8")
    (tmp_path / "settings.py").write_text(LOADER, encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(main), encoding="utf-8")
    return run_python("main.py", cwd=tmp_path)


def test_a_memo_called_before_import_cash_is_named_with_the_fix(tmp_path):
    done = _project(
        tmp_path,
        """
        import settings
        settings.load_cfg()
        import cash
        """,
    )
    assert done.stderr.count(CODE) == 1, done.stderr
    line = next(line for line in done.stderr.splitlines() if CODE in line)
    assert "settings.load_cfg" in line
    assert "settings.unused" not in line, "an empty memo was named"
    assert "settings.py:4" in line, "the warning does not point at the memo"
    assert "import cash before" in line


def test_a_memo_first_called_after_import_cash_is_quiet(tmp_path):
    done = _project(
        tmp_path,
        """
        import settings
        import cash
        settings.load_cfg()
        """,
    )
    assert CODE not in done.stderr, done.stderr


def test_a_library_memo_holding_results_is_quiet(tmp_path):
    """The standard library fills its own memos (``re``, ``fnmatch``,
    ``ipaddress``) long before ``import cash``; those are not the user's."""
    done = _project(
        tmp_path,
        """
        import fnmatch, settings
        fnmatch.fnmatch("a.txt", "*.txt")
        import cash
        """,
    )
    assert CODE not in done.stderr, done.stderr
