"""A helper made by a factory is keyed by what its closure and defaults hold
now, not by what they held the first time it was hashed in the process.

``score = make_scorer(weights)`` with ``weights["a"] = 3`` later, a ``get``
whose captured variable a sibling ``nonlocal`` setter re-sets, or a helper's
keyword-only default dict changed in place: the helper's identity was
remembered per function object while its defaults were the same objects, so
the cached caller kept serving the old result in that process (a new
process recomputed).
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent("""
    import cash

    weights = {"a": 2}
    def make_scorer(w):
        def score(x):
            return x * w["a"]          # only reads the captured dict
        return score
    score = make_scorer(weights)

    def make_setting():
        m = 2
        def get(x):
            return x * m
        def set_m(v):
            nonlocal m
            m = v
        return get, set_m
    get, set_m = make_setting()

    def scaled(x, *, opts={"k": 2}):
        return x * opts["k"]

    def shifted(x, extra=[2]):
        return x * extra[0]

    @cash.cache
    def by_dict(x):
        return score(x)

    @cash.cache
    def by_cell(x):
        return get(x)

    @cash.cache
    def by_kwdefault(x):
        return scaled(x)

    @cash.cache
    def by_default(x):
        return shifted(x)

    before = [by_dict(5), by_cell(5), by_kwdefault(5), by_default(5)]
    weights["a"] = 3
    set_m(3)
    scaled.__kwdefaults__["opts"]["k"] = 3
    shifted.__defaults__[0][0] = 3
    after = [by_dict(5), by_cell(5), by_kwdefault(5), by_default(5)]
    plain = [score(5), get(5), scaled(5), shifted(5)]
    print(before, after, plain)
""")


def test_a_helper_s_capture_or_default_changed_at_run_time_recomputes(tmp_path):
    (tmp_path / "t.py").write_text(SCRIPT, encoding="utf-8")
    out = run_python("t.py", cwd=tmp_path).stdout.strip()
    assert out == "[10, 10, 10, 10] [15, 15, 15, 15] [15, 15, 15, 15]"


IMMUTABLE = textwrap.dedent("""
    import cash

    def make_scaler(k):
        def scale(x):
            return x * k
        return scale
    scale = make_scaler(2)

    weights = {"a": 2}
    def make_scorer(w):
        def score(x):
            return x * w["a"]
        return score
    score = make_scorer(weights)

    @cash.cache
    def total(x):
        return scale(x)

    @cash.cache
    def scored(x):
        return score(x)

    print(total(5), total(5), scored(5), scored(5), total.cache_info()["hits"], scored.cache_info()["hits"])
""")


def test_an_unchanged_capture_is_still_a_hit(tmp_path):
    # Positive control: hashing a mutable capture on every call keys it,
    # it does not make every call miss.
    (tmp_path / "t.py").write_text(IMMUTABLE, encoding="utf-8")
    assert run_python("t.py", cwd=tmp_path).stdout.split() == ["10", "10", "10", "10", "1", "1"]
