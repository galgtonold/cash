"""A pint quantity keys by its magnitude, its units and its registry's
definitions.

pint builds a ``Quantity`` class per unit registry and names its module
``pint``, where ``pint.Quantity`` is another class. It counted as exec'd
user code, so a quantity argument or default folded the class's data -- the
whole registry, down to the weak index each context keeps -- and no key
could be built (KEY-UNHASHABLE-DEFAULT, about a ``WeakValueDictionary``
callback). A quantity's pickle names neither the registry nor what its unit
names mean there, so two registries defining ``widget`` differently must
key apart by their definitions.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash, FileBackend

pint = pytest.importorskip("pint")
np = pytest.importorskip("numpy")

UREG = pint.UnitRegistry()
Q_ = UREG.Quantity


def _in_meters(x):
    return x.to("meter").magnitude


def _with_default(depth, gradient=Q_(30.0, "delta_degC/km")):
    return (gradient * depth).to("delta_degC").magnitude


def _reads_global_class(n):
    return _in_meters(_REG_CLASS(n, "widget"))


_REG_CLASS = Q_


def _cache(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _codes(fn):
    return [w["code"] for w in fn.cache_info()["warnings"]]


def test_the_same_quantity_hits_and_builds_its_key(tmp_path):
    f = _cache(tmp_path).cache(_in_meters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(Q_(1500.0, "m")) == 1500.0
        assert f(Q_(1500.0, "m")) == 1500.0
    assert f.cache_info()["hits"] == 1
    assert _codes(f) == []


def test_a_quantity_default_is_keyed(tmp_path):
    f = _cache(tmp_path).cache(_with_default)
    assert f(Q_(2.0, "km")) == pytest.approx(60.0)
    assert f(Q_(2.0, "km")) == pytest.approx(60.0)
    assert f(Q_(2.0, "km"), Q_(25.0, "delta_degC/km")) == pytest.approx(50.0)
    info = f.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2)
    assert _codes(f) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (Q_(1.0, "m"), Q_(2.0, "m")),
        (Q_(1.0, "m"), Q_(1.0, "cm")),
        (Q_(np.array([1.0, 2.0]), "m"), Q_(np.array([1.0, 3.0]), "m")),
    ],
    ids=["magnitude", "unit", "array-magnitude"],
)
def test_a_different_magnitude_or_unit_misses(tmp_path, first, second):
    f = _cache(tmp_path).cache(_in_meters)
    np.testing.assert_allclose(f(first), first.to("m").magnitude)
    np.testing.assert_allclose(f(second), second.to("m").magnitude)
    assert f.cache_info()["hits"] == 0


def _registry(widget):
    ureg = pint.UnitRegistry()
    ureg.define(f"widget = {widget} * meter")
    return ureg


def test_registries_with_different_definitions_key_apart(tmp_path):
    f = _cache(tmp_path).cache(_in_meters)
    assert f(_registry(2).Quantity(1, "widget")) == 2
    assert f(_registry(3).Quantity(1, "widget")) == 3
    assert f.cache_info()["hits"] == 0


def test_registries_with_the_same_definitions_share_entries(tmp_path):
    f = _cache(tmp_path).cache(_in_meters)
    assert f(_registry(2).Quantity(1, "widget")) == 2
    assert f(_registry(2).Quantity(1, "widget")) == 2
    assert f.cache_info()["hits"] == 1


def test_a_definition_added_to_the_registry_moves_the_key(tmp_path):
    ureg = _registry(2)
    f = _cache(tmp_path).cache(_in_meters)
    f(ureg.Quantity(1, "widget"))
    ureg.define("gadget = 7 * meter")
    f(ureg.Quantity(1, "widget"))
    assert f.cache_info()["hits"] == 0


def test_an_active_context_moves_the_key(tmp_path):
    f = _cache(tmp_path).cache(_in_meters)
    ureg = pint.UnitRegistry()
    f(ureg.Quantity(1, "m"))
    with ureg.context("sp"):
        f(ureg.Quantity(1, "m"))
    f(ureg.Quantity(1, "m"))
    assert f.cache_info()["hits"] == 1


def test_a_global_registry_class_keys_by_its_registry(tmp_path):
    global _REG_CLASS
    f = _cache(tmp_path).cache(_reads_global_class)
    try:
        _REG_CLASS = _registry(2).Quantity
        assert f(1) == 2
        _REG_CLASS = _registry(3).Quantity
        assert f(1) == 3
        assert f(1) == 3
    finally:
        _REG_CLASS = Q_
    assert f.cache_info()["hits"] == 1


def _context_with(fn):
    ureg = pint.UnitRegistry()
    ctx = pint.Context("custom")
    ctx.add_transformation("[length]", "[time]", fn)
    ureg.add_context(ctx)
    return ureg


def test_a_python_function_in_a_context_keys_by_its_code(tmp_path):
    f = _cache(tmp_path).cache(_in_meters)
    light = _context_with(lambda reg, x: x / reg.Quantity(3e8, "m/s"))
    sound = _context_with(lambda reg, x: x / reg.Quantity(343.0, "m/s"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(light.Quantity(1, "m")) == 1
        assert f(light.Quantity(1, "m")) == 1
        assert f(sound.Quantity(1, "m")) == 1
    assert f.cache_info()["hits"] == 1
    assert f.cache_info()["misses"] == 2


ALIASES = {"thing": "meter"}


def _alias(text):
    return ALIASES.get(text, text)


def _to_unit(x, unit):
    return x.to(unit).magnitude


def test_a_preprocessor_keys_by_the_globals_it_reads(tmp_path):
    ureg = pint.UnitRegistry()
    ureg.preprocessors.append(_alias)
    f = _cache(tmp_path).cache(_to_unit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(ureg.Quantity(2000.0, "m"), "thing") == 2000.0
        assert f(ureg.Quantity(2000.0, "m"), "thing") == 2000.0
        ALIASES["thing"] = "kilometer"
        try:
            # pint remembers what it parsed; a registry that has not met
            # "thing" yet reads the new alias, as plain Python would.
            ureg._cache.parse_unit.clear()
            expected = ureg.Quantity(2000.0, "m").to("thing").magnitude
            assert expected == 2.0
            assert f(ureg.Quantity(2000.0, "m"), "thing") == expected
        finally:
            ALIASES["thing"] = "meter"
    assert f.cache_info()["hits"] == 1
    assert f.cache_info()["misses"] == 2


class _Table:
    """An object with no value form: its repr is its address."""

    def __init__(self):
        self.names = {"thing": "meter"}


TABLE = _Table()


def _alias_from_table(text):
    return TABLE.names.get(text, text)


def test_a_function_reading_an_object_cash_cannot_key_runs_uncached_with_a_warning(tmp_path):
    ureg = pint.UnitRegistry()
    ureg.preprocessors.append(_alias_from_table)
    f = _cache(tmp_path).cache(_in_meters)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert f(ureg.Quantity(1, "m")) == 1
        assert f(ureg.Quantity(1, "m")) == 1
    assert f.cache_info()["hits"] == 0
    assert "KEY-UNHASHABLE-ARG" in _codes(f)


def test_a_metpy_quantity_hits(tmp_path):
    metpy_units = pytest.importorskip("metpy.units")
    units = metpy_units.units
    f = _cache(tmp_path).cache(_in_meters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(units.Quantity(np.array([1.5, 2.0]), "km")).tolist() == [1500.0, 2000.0]
        assert f(units.Quantity(np.array([1.5, 2.0]), "km")).tolist() == [1500.0, 2000.0]
        assert f(units.Quantity(np.array([1.5, 2.0]), "m")).tolist() == [1.5, 2.0]
    assert f.cache_info()["hits"] == 1
    assert _codes(f) == []


def _registry_with_pattern_preprocessor(pattern):
    """A registry with the preprocessor xclim adds: a compiled pattern's
    ``sub`` method, bound and partially applied."""
    import functools
    import re

    ureg = pint.UnitRegistry()
    ureg.preprocessors.append(functools.partial(re.compile(pattern).sub, "**"))
    return ureg


def test_a_bound_pattern_preprocessor_keys_by_its_pattern(tmp_path):
    first = _registry_with_pattern_preprocessor(r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")
    same = _registry_with_pattern_preprocessor(r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")
    other = _registry_with_pattern_preprocessor(r"(?<=[A-Za-z])(?=[0-9])")
    f = _cache(tmp_path).cache(_in_meters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(first.Quantity(3.0, "km")) == 3000.0
        assert f(same.Quantity(3.0, "km")) == 3000.0
        assert f(other.Quantity(3.0, "km")) == 3000.0
    info = f.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2)
    assert _codes(f) == []


def test_an_xclim_quantity_hits_and_a_changed_one_misses(tmp_path):
    units = pytest.importorskip("xclim.core.units").units
    f = _cache(tmp_path).cache(_in_meters)
    calls = [
        units.Quantity(np.array([1.5, 2.0]), "km"),
        units.Quantity(np.array([1.5, 2.0]), "km"),
        units.Quantity(np.array([1.5, 3.0]), "km"),
        units.Quantity(np.array([1.5, 2.0]), "m"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = [f(q).tolist() for q in calls]
    assert got == [_in_meters(q).tolist() for q in calls]
    info = f.cache_info()
    assert (info["hits"], info["misses"]) == (1, 3)
    assert _codes(f) == []
