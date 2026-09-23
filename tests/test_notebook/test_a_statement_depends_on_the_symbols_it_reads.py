"""What reading some names from a module depends on.

``closure_digest`` is what lets editing one function in a module leave the
statements that read OTHER functions alone (2.8x then 9.1x in one notebook).
Leaving something out of the closure serves a value computed from code that
has since changed, so the tests are mostly the other direction: every way a
name's behaviour can change without its own source changing must move the
digest, and every way cash cannot bound the answer must return None.
"""

from __future__ import annotations

import textwrap

import pytest

from cash.notebook.module_symbols import closure_digest_of_source, static_attribute_reads


def _d(source, *names):
    return closure_digest_of_source(textwrap.dedent(source), names)


BASE = """
    import numpy as np
    THRESHOLD = 3

    def build_table():
        return 2

    TABLE = build_table()

    def _helper(n):
        return n + 1

    def load(n):
        return [_helper(i) * THRESHOLD * TABLE for i in range(n)]

    def report(rows):
        return "v1:" + str(len(rows))
"""


class TestWhatDoesNotMatter:
    """The point of the feature."""

    def test_an_unrelated_function(self):
        assert _d(BASE, "load") == _d(BASE.replace('"v1:"', '"v2:"'), "load")

    def test_a_comment_or_a_blank_line(self):
        assert _d(BASE, "load") == _d(BASE + "\n\n    # a note\n", "load")

    def test_a_new_function_nothing_calls(self):
        assert _d(BASE, "load") == _d(BASE + "\n    def extra():\n        return 9\n", "load")

    def test_an_import_nothing_in_the_closure_reads(self):
        src = BASE.replace("import numpy as np", "import numpy as np\n    import math")
        assert _d(BASE, "load") == _d(src, "load")


class TestWhatDoes:
    """Every way `load` can change without `load` changing."""

    @pytest.mark.parametrize(
        "old,new,what",
        [
            ("return [_helper(i)", "return [_helper(i + 0)", "the function itself"),
            ("return n + 1", "return n + 2", "a helper it calls"),
            ("THRESHOLD = 3", "THRESHOLD = 4", "a constant it reads"),
            ("return 2", "return 5", "a function run at import time to build a value it reads"),
        ],
    )
    def test_a_dependency_moves_the_digest(self, old, new, what):
        assert _d(BASE, "load") != _d(BASE.replace(old, new), "load"), what

    def test_a_redefinition_below_the_first(self):
        """The later def is the live one; both are the name's source."""
        src = BASE + "\n    def _helper(n):\n        return n + 7\n"
        assert _d(BASE, "load") != _d(src, "load")

    def test_a_conditional_rebinding(self):
        src = BASE + "\n    if THRESHOLD > 1:\n        TABLE = 99\n"
        assert _d(BASE, "load") != _d(src, "load")

    def test_import_time_code_that_mutates_what_it_reads(self):
        src = BASE.replace("TABLE = build_table()", "TABLE = [build_table()]") + "\n    TABLE.append(4)\n"
        before = BASE.replace("TABLE = build_table()", "TABLE = [build_table()]")
        assert _d(before, "load") != _d(src, "load")

    def test_import_time_code_is_in_every_closure_whatever_it_names(self):
        """A bare call at import time can do anything; it is never left out."""
        a = BASE + "\n    print('loaded')\n"
        b = BASE + "\n    print('loaded!')\n"
        assert _d(a, "report") != _d(b, "report")

    def test_a_class_and_what_its_methods_read(self):
        src = """
            RATE = 2
            class Model:
                def fit(self, x):
                    return x * RATE
            def other():
                return 1
        """
        assert _d(src, "Model") != _d(src.replace("RATE = 2", "RATE = 3"), "Model")
        assert _d(src, "Model") == _d(src.replace("return 1", "return 2"), "Model")

    def test_a_decorator_defined_in_the_module(self):
        src = """
            def deco(f):
                return f
            @deco
            def load():
                return 1
        """
        assert _d(src, "load") != _d(src.replace("return f", "return lambda: 2"), "load")

    def test_a_default_argument_read_from_the_module(self):
        src = """
            LIMIT = 10
            def load(n=LIMIT):
                return n
        """
        assert _d(src, "load") != _d(src.replace("LIMIT = 10", "LIMIT = 11"), "load")

    def test_a_cash_directive_anywhere_in_the_file(self):
        """Instructions to cash, not commentary."""
        src = BASE.replace("def report(rows):", "def report(rows):  # @cash:assume-safe")
        assert _d(BASE, "load") != _d(src, "load")

    def test_reading_two_names_covers_both(self):
        assert _d(BASE, "load", "report") != _d(BASE.replace('"v1:"', '"v2:"'), "load", "report")


class TestWhenItCannotSay:
    """None: depend on the whole module, which is always right."""

    def test_a_name_the_module_does_not_bind(self):
        assert _d(BASE, "missing") is None

    def test_a_star_import(self):
        assert _d("from os.path import *\ndef load():\n    return 1\n", "load") is None

    def test_a_module_level_getattr(self):
        assert _d("def __getattr__(name):\n    return 1\ndef load():\n    return 2\n", "load") is None

    @pytest.mark.parametrize(
        "body",
        [
            "return globals()['x']",
            "return eval('1')",
            "exec('x = 1')",
            "setattr(obj, 'a', 1)",
            "return sys.modules[__name__]",
            "return obj.__dict__",
        ],
    )
    def test_code_in_the_closure_that_reaches_the_namespace_by_string(self, body):
        src = "import sys\nobj = object()\ndef load():\n    " + body + "\n"
        assert _d(src, "load") is None

    def test_dynamic_code_outside_the_closure_does_not_matter(self):
        src = "def other():\n    return eval('1')\ndef load():\n    return 2\n"
        assert _d(src, "load") is not None

    def test_source_that_does_not_parse(self):
        assert _d("def load(:\n", "load") is None


class TestStaticAttributeReads:
    """Which names a statement reads from a module -- or that it does more."""

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("x = lib.load(3)", {"load"}),
            ("x = lib.load(lib.SIZE)", {"load", "SIZE"}),
            ("x = [lib.f(i) for i in range(3)]", {"f"}),
            ("x = lib.sub.fn()", {"sub"}),
        ],
    )
    def test_plain_reads(self, code, expected):
        assert static_attribute_reads(code, "lib") == expected

    @pytest.mark.parametrize(
        "code",
        [
            "x = run(lib)",  # passed whole
            "x = lib",  # rebound
            "lib.load = other",  # stored
            "del lib.load",
            "x = getattr(lib, name)",
            "x = lib.__dict__['load']",
            "x = 1",  # not read at all
        ],
    )
    def test_anything_else_is_not_static(self, code):
        assert static_attribute_reads(code, "lib") is None


class TestModuleReadLineage:
    """The one function all three valuations of a module input go through.

    Every None here means "the module's whole lineage", the old behaviour,
    which is always correct: these are the cases narrowing must NOT apply.
    """

    @staticmethod
    def _setup(tmp_path, name="mrl_lib", tracked=True):
        import importlib.util
        import types as _types

        path = tmp_path / (name + ".py")
        path.write_text(textwrap.dedent(BASE), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        tracker = _types.SimpleNamespace(
            _tracked_modules={name} if tracked else set(),
            _dep_file_to_parents={},
        )
        return path, module, tracker

    def test_a_plain_attribute_read_is_narrowed(self, tmp_path):
        from cash.notebook.lineage_formula import module_read_lineage

        _path, module, tracker = self._setup(tmp_path)
        assert module_read_lineage(tracker, "lib", module, "x = lib.load(3)") is not None

    def test_the_value_ignores_an_unrelated_function_and_follows_the_read_one(self, tmp_path):
        from cash.notebook.lineage_formula import module_read_lineage

        path, module, tracker = self._setup(tmp_path)
        before = module_read_lineage(tracker, "lib", module, "x = lib.load(3)")
        path.write_text(textwrap.dedent(BASE).replace('"v1:"', '"v2:"'), encoding="utf-8")
        assert module_read_lineage(tracker, "lib", module, "x = lib.load(3)") == before
        path.write_text(textwrap.dedent(BASE).replace("return n + 1", "return n + 2"), encoding="utf-8")
        assert module_read_lineage(tracker, "lib", module, "x = lib.load(3)") != before

    @pytest.mark.parametrize("case", ["untracked", "not a module", "bare use", "no code", "no tracker"])
    def test_whole_lineage_when_narrowing_does_not_apply(self, tmp_path, case):
        from cash.notebook.lineage_formula import module_read_lineage

        _path, module, tracker = self._setup(tmp_path, tracked=(case != "untracked"))
        value = object() if case == "not a module" else module
        code = {"bare use": "x = run(lib)", "no code": None}.get(case, "x = lib.load(3)")
        trk = None if case == "no tracker" else tracker
        assert module_read_lineage(trk, "lib", value, code) is None

    def test_a_tracked_dependency_file_still_counts_whole(self, tmp_path):
        """Another local module the closure uses is not narrowed away."""
        from cash.notebook.lineage_formula import module_read_lineage

        _path, module, tracker = self._setup(tmp_path)
        dep = tmp_path / "helper_dep.py"
        dep.write_text("VALUE = 1\n", encoding="utf-8")
        tracker._dep_file_to_parents = {str(dep): {"mrl_lib"}}
        before = module_read_lineage(tracker, "lib", module, "x = lib.load(3)")
        dep.write_text("VALUE = 2\n", encoding="utf-8")
        assert module_read_lineage(tracker, "lib", module, "x = lib.load(3)") != before

    def test_an_edit_inside_one_timestamp_tick_is_still_seen(self, tmp_path):
        """Same size, same mtime, new code: what a Windows timer tick allows.

        The closure and the analysis under it are memoised on the file's
        stat, and a same-size rewrite within one tick keeps it -- the two
        tests above failed intermittently on Windows CI that way. Forced here
        by putting the mtime back.
        """
        import os

        from cash.notebook.lineage_formula import module_read_lineage

        path, module, tracker = self._setup(tmp_path)
        st = os.stat(path)
        before = module_read_lineage(tracker, "lib", module, "x = lib.load(3)")
        path.write_text(textwrap.dedent(BASE).replace("return n + 1", "return n + 2"), encoding="utf-8")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert os.stat(path).st_size == st.st_size

        assert module_read_lineage(tracker, "lib", module, "x = lib.load(3)") != before


class TestNondeterministicImportTimeCode:
    """A hole narrowing would open, closed.

    Keying on the whole module reloaded it only on a change, which re-keyed
    everything. Narrowing reloads it on an edit to ANY part, so import-time
    code whose result differs run to run (``STAMP = time.time()``) would put
    a new value under an old key. Such a closure is unbounded: the old
    behaviour.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "STAMP = time.time()",
            "STAMP = datetime.datetime.now()",
            "STAMP = random.random()",
            "STAMP = uuid.uuid4()",
            "STAMP = np.random.rand()",
        ],
    )
    def test_a_nondeterministic_value_in_the_closure(self, line):
        src = "import time, datetime, random, uuid\nimport numpy as np\n" + line + "\ndef load():\n    return STAMP\n"
        assert _d(src, "load") is None

    def test_a_deterministic_call_at_import_time_is_fine(self):
        src = "def build():\n    return 2\nTABLE = build()\ndef load():\n    return TABLE\n"
        assert _d(src, "load") is not None

    def test_a_clock_read_inside_a_function_is_not_import_time_code(self):
        """A helper that times its own steps made
        every edit to its module -- even appending an unrelated function --
        re-run everything built on it. A clock read that runs only when the
        function is CALLED gives no new value on a reload."""
        src = (
            "import time\ndef summary(rows):\n    t0 = time.perf_counter()\n"
            "    print(time.perf_counter() - t0)\n    return sum(rows)\n"
        )
        assert _d(src, "summary") is not None
        assert _d(src + "def unrelated():\n    return 1\n", "summary") == _d(src, "summary")

    @pytest.mark.parametrize(
        "src",
        [
            "def make():\n    return time.time()\nSTAMP = make()\n",
            "def inner():\n    return time.time()\ndef make():\n    return inner()\nSTAMP = make()\n",
            "def make():\n    return time.time()\nclass C:\n    stamp = make()\nSTAMP = C.stamp\n",
            "def make(t=time.time()):\n    return t\nSTAMP = 1\n",
        ],
        ids=["called_at_import", "called_through_another", "called_in_a_class_body", "a_default"],
    )
    def test_a_function_run_at_import_time_still_counts(self, src):
        src = "import time\n" + src + "def load():\n    return STAMP, make\n"
        assert _d(src, "load") is None

    def test_nondeterminism_nothing_in_the_closure_reaches(self):
        """Only what the read name reaches counts -- except import-time code,
        which is in every closure; a def that is never called is not."""
        src = "import time\ndef stamp():\n    return time.time()\ndef load():\n    return 1\n"
        assert _d(src, "load") is not None
