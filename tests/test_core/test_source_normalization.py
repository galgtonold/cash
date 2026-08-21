"""Comment- and format-insensitive code identity.

A comment edit, a reformat, or a blank line must NOT invalidate a cache
entry: the compiled behaviour is unchanged, so recomputing is pure waste.
Two things must still move the digest -- a real code change, and a
``# @cash:`` annotation, because those are directives that change caching
behaviour rather than ordinary prose.
"""

import textwrap
import warnings

import pytest

from cash.source_norm import normalize_source_for_hash


def _same(a: str, b: str) -> bool:
    return normalize_source_for_hash(textwrap.dedent(a)) == normalize_source_for_hash(
        textwrap.dedent(b)
    )


BASE = """
    def f(n):
        total = n * 2
        return total
    """


# --- edits that must NOT change identity -------------------------------

def test_added_comment_keeps_identity():
    edited = """
        def f(n):
            # doubling, because the caller halves it later
            total = n * 2
            return total
        """
    assert _same(BASE, edited)


def test_trailing_comment_keeps_identity():
    edited = """
        def f(n):
            total = n * 2  # doubled
            return total
        """
    assert _same(BASE, edited)


def test_blank_lines_keep_identity():
    edited = """
        def f(n):

            total = n * 2

            return total
        """
    assert _same(BASE, edited)


def test_trailing_whitespace_keeps_identity():
    edited = "\ndef f(n):   \n    total = n * 2\t\n    return total  \n"
    assert normalize_source_for_hash(
        textwrap.dedent(BASE)
    ) == normalize_source_for_hash(edited)


def test_reindentation_keeps_identity():
    """A consistent 4-space -> 2-space reformat does not change behaviour."""
    edited = """
        def f(n):
          total = n * 2
          return total
        """
    assert _same(BASE, edited)


# --- edits that MUST change identity -----------------------------------

def test_real_code_change_breaks_identity():
    edited = """
        def f(n):
            total = n * 3
            return total
        """
    assert not _same(BASE, edited)


def test_docstring_change_breaks_identity():
    """Docstrings are ordinary constants; a function may return one."""
    a = '''
        def f(n):
            "one"
            return n
        '''
    b = '''
        def f(n):
            "two"
            return n
        '''
    assert not _same(a, b)


def test_indentation_that_changes_structure_breaks_identity():
    """Dedenting a statement out of a block is a real behaviour change."""
    a = """
        def f(n):
            if n:
                total = 1
                return total
        """
    b = """
        def f(n):
            if n:
                total = 1
            return total
        """
    assert not _same(a, b)


# --- cash annotations stay load-bearing --------------------------------

@pytest.mark.parametrize(
    "annotation",
    [
        "# @cash: no-cache",
        "# @cash:persist",
        "# @cash: allow-random",
        "# @cash: ttl=300",
    ],
)
def test_cash_annotation_breaks_identity(annotation):
    """These comments change caching behaviour, so they must move the key."""
    edited = f"""
        def f(n):
            {annotation}
            total = n * 2
            return total
        """
    assert not _same(BASE, edited)


def test_differing_annotation_values_differ():
    a = """
        def f(n):
            # @cash: ttl=300
            return n
        """
    b = """
        def f(n):
            # @cash: ttl=600
            return n
        """
    assert not _same(a, b)


def test_annotation_spacing_is_not_load_bearing():
    """Same directive, different spacing -- still the same directive."""
    a = """
        def f(n):
            # @cash:no-cache
            return n
        """
    b = """
        def f(n):
            #   @cash:   no-cache
            return n
        """
    assert _same(a, b)


def test_annotation_hashing_does_not_warn():
    """A malformed ttl warns when PARSED; hashing must stay silent.

    ``parse_annotation_line`` emits ``CashCacheIneffectiveWarning`` for
    ``ttl=5m``. Hashing runs on every call, so routing it through that
    parser would spray a warning per cache lookup.
    """
    src = textwrap.dedent(
        """
        def f(n):
            # @cash: ttl=5m
            return n
        """
    )
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        normalize_source_for_hash(src)
    assert not log, f"hashing emitted warnings: {[str(w.message) for w in log]}"


# --- never raise from inside a hasher ----------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "def f(:\n    pass",          # syntax error mid-edit
        "    return 1",               # bare fragment, unexpected indent
        "",                            # empty
        "def f():\n    return '''un",  # unterminated string
    ],
)
def test_unparseable_source_falls_back_without_raising(bad):
    out = normalize_source_for_hash(bad)
    assert isinstance(out, str)


def test_unparseable_source_still_distinguishes_content():
    """The fallback must not collapse different broken sources together."""
    a = normalize_source_for_hash("def f(:\n    return 1")
    b = normalize_source_for_hash("def f(:\n    return 2")
    assert a != b


# --- end to end, through the real decorated call -----------------------

_MODULE = '''
class Schema:
    FIELD = "a"

def helper(x):
    return x * 2

@cash_instance.cache
def compute(n, schema):
    return helper(n) + len(schema.FIELD)
'''


def _call(tmp_path, c, source):
    """Write *source* as the same module, import it fresh, call once.

    Returns True when the call was served from cache. A fresh import means
    a fresh wrapper with zero'd stats, so one call reads unambiguously.
    """
    import importlib.util
    import linecache
    import sys

    path = tmp_path / "usermod.py"
    path.write_text(source)
    importlib.invalidate_caches()
    linecache.clearcache()
    spec = importlib.util.spec_from_file_location("usermod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["usermod"] = mod
    mod.__dict__["cash_instance"] = c
    try:
        spec.loader.exec_module(mod)
        mod.compute(5, mod.Schema)
        return mod.compute.cache_info()["hits"] == 1
    finally:
        sys.modules.pop("usermod", None)


@pytest.fixture()
def cache_env(tmp_path):
    from cash import Cash

    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    work = tmp_path / "work"
    work.mkdir()
    return c, work


def test_identical_source_hits(cache_env):
    """Null control: without this the arms below prove nothing."""
    c, work = cache_env
    assert _call(work, c, _MODULE) is False, "first call must compute"
    assert _call(work, c, _MODULE) is True, "identical source must hit"


def test_comment_in_cached_function_hits(cache_env):
    c, work = cache_env
    _call(work, c, _MODULE)
    edited = _MODULE.replace(
        "def compute(n, schema):\n    return",
        "def compute(n, schema):\n    # explanatory comment\n    return",
    )
    assert edited != _MODULE
    assert _call(work, c, edited) is True


def test_comment_in_transitive_helper_hits(cache_env):
    """The surprising half: editing something the cached function CALLS."""
    c, work = cache_env
    _call(work, c, _MODULE)
    edited = _MODULE.replace(
        "def helper(x):\n    return", "def helper(x):\n    # a note\n    return"
    )
    assert edited != _MODULE
    assert _call(work, c, edited) is True


def test_reformatting_hits(cache_env):
    c, work = cache_env
    _call(work, c, _MODULE)
    edited = _MODULE.replace(
        "def compute(n, schema):\n    return", "def compute(n, schema):\n\n    return"
    )
    assert edited != _MODULE
    assert _call(work, c, edited) is True


def test_real_body_change_still_recomputes(cache_env):
    """Control: normalization must not swallow a genuine edit."""
    c, work = cache_env
    _call(work, c, _MODULE)
    edited = _MODULE.replace("helper(n) + len", "helper(n) + 100 + len")
    assert edited != _MODULE
    assert _call(work, c, edited) is False


def test_real_helper_change_still_recomputes(cache_env):
    """Control: a behaviour change in a helper must still invalidate."""
    c, work = cache_env
    _call(work, c, _MODULE)
    edited = _MODULE.replace("return x * 2", "return x * 3")
    assert edited != _MODULE
    assert _call(work, c, edited) is False


def test_normalization_is_memoized():
    """The memo is load-bearing, not an optimization detail.

    Tokenizing costs ~47us against ~0.5us for hashing raw text, and this
    runs once per transitive helper per cached call. Without the memo,
    ``test_cfd_loop_overhead`` fails on CPU-time overhead -- measured, not
    predicted. Guard it so it cannot be quietly dropped.
    """
    src = textwrap.dedent(BASE)
    normalize_source_for_hash(src)
    before = normalize_source_for_hash.cache_info().hits
    normalize_source_for_hash(src)
    after = normalize_source_for_hash.cache_info().hits
    assert after == before + 1, "repeat normalization must be served from the memo"


def test_source_hash_memo_is_keyed_by_identity_not_value(tmp_path):
    """Two value-equal code objects must not share one memo entry.

    ``_hash_callable_source`` memoizes its digest so a helper's source is not
    re-read and re-tokenized on every cache hit. That memo must key on the
    code object's IDENTITY: ``CodeType`` implements ``__eq__``/``__hash__`` BY
    VALUE and ``co_filename`` is not part of it, so two functions from
    different FILES can occupy one dict slot.

    Their digests can still differ. A TRAILING ``# @cash:`` annotation changes
    the normalized source without changing a byte of bytecode or a line
    number, so a value-keyed memo served one file's digest for the other --
    silently dropping or applying a caching directive. Found as a reproducible
    xdist-only failure of ``test_real_helper_change_still_recomputes``.

    Real files on disk are required: ``inspect.getsource`` must SUCCEED here,
    or both sides fall back to ``bytecode_identity`` and compare equal for a
    reason that has nothing to do with the memo. That mistake made the first
    version of this test fail against correct code.
    """
    import importlib.util
    import inspect
    import sys

    from cash import Cash

    def load(name, body):
        path = tmp_path / (name + ".py")
        path.write_text(body, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.f

    plain = "def f(x):\n    return x * 2\n"
    annotated = "def f(x):\n    return x * 2  # @cash: no-cache\n"
    fn_a = load("memo_mod_a", plain)
    fn_b = load("memo_mod_b", annotated)
    try:
        assert inspect.getsource(fn_a), "premise: source really is retrievable"
        assert fn_a.__code__ == fn_b.__code__, "premise: code objects are value-equal"
        assert hash(fn_a.__code__) == hash(fn_b.__code__), "premise: one dict slot"
        assert fn_a.__code__ is not fn_b.__code__
        assert normalize_source_for_hash(plain) != normalize_source_for_hash(annotated)

        cash = Cash(register_magic=False)
        assert cash._hash_callable_source(fn_a) != cash._hash_callable_source(fn_b), (
            "the memo conflated two value-equal code objects with different sources"
        )
    finally:
        for name in ("memo_mod_a", "memo_mod_b"):
            sys.modules.pop(name, None)
