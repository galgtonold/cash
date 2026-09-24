"""The per-code source memo keeps taking new functions once it is full.

A helper's source digest is most of what a key costs (`inspect.getsource`
re-tokenises the file on every call). The memo stopped taking entries at its
size, so in a long session every function defined after that paid the full
cost on every call.
"""

from cash.decorator import code_identity


def _helpers():
    def a(x):
        return x + 1

    def b(x):
        return x + 2

    def c(x):
        return x + 3

    def d(x):
        return x + 4

    return [a, b, c, d]


def test_a_function_defined_after_the_memo_filled_is_memoized(monkeypatch):
    monkeypatch.setattr(code_identity.SOURCE_HASH_MEMO, "maxsize", 2)
    helpers = _helpers()
    for fn in helpers:
        code_identity.hash_callable_source(fn)

    reads = []
    real = code_identity.source_digest
    monkeypatch.setattr(code_identity, "source_digest", lambda fn: reads.append(fn) or real(fn))
    newest = helpers[-1]
    code_identity.hash_callable_source(newest)

    assert id(newest.__code__) in code_identity.SOURCE_HASH_MEMO
    assert reads == [], "the newest function's source was read again"
    assert len(code_identity.SOURCE_HASH_MEMO) <= 2
