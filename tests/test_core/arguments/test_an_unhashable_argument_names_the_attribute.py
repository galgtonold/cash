"""KEY-UNHASHABLE-ARG says which attribute of the argument is the reason.

A cached method on an object that holds one lock, client or logger runs
uncached on every call. The warning named the class and told the user to
write ``__cash_key__``; it named neither the parameter (``self``) nor the
attribute to leave out, so the user had to bisect their own object.
"""

from __future__ import annotations

import threading
import warnings


class Model:
    def __init__(self):
        self.weights = [1, 2, 3]
        self.name = "m"
        self._lock = threading.Lock()

    def predict(self, x):
        return sum(self.weights) * x


class Session:
    def __init__(self):
        self.url = "db://x"
        self.guard = threading.Lock()


class Service:
    def __init__(self):
        self.retries = 3
        self.session = Session()


class OnlyALock:
    def __init__(self):
        self.lock = threading.Lock()


class Wide:
    def __init__(self):
        self.a, self.b, self.c, self.d, self.e = 1, 2, 3, 4, 5
        self.lock = threading.Lock()


class Guarded:
    """An attribute read that is not in ``__dict__`` must not run."""

    def __init__(self):
        self.n = 1
        self.lock = threading.Lock()

    @property
    def expensive(self):
        raise AssertionError("the walk read a property")

    def __getattr__(self, name):
        raise AssertionError(f"the walk asked __getattr__ for {name}")


def _full_warning(cached, *args, **kwargs) -> str:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        cached(*args, **kwargs)
    [logged] = [w["message"] for w in cached.cache_info()["warnings"] if w["code"] == "KEY-UNHASHABLE-ARG"]
    [line] = [str(w.message) for w in shown if getattr(w.message, "code", None) == "KEY-UNHASHABLE-ARG"]
    assert "\n" not in line
    return logged


def test_a_method_names_self_and_the_attribute(disk_cash):
    predict = disk_cash.cache(Model.predict)
    text = _full_warning(predict, Model(), 2)
    assert "the argument `self`, of type Model, could not be hashed" in text
    assert "its attribute `_lock` (lock) cannot be hashed" in text
    assert "leaves `_lock` out" in text
    assert "`def __cash_key__(self): return (self.weights, self.name)`" in text
    assert predict.cache_info()["hits"] == 0


def test_the_one_line_form_names_the_attribute_too(disk_cash):
    predict = disk_cash.cache(Model.predict)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        predict(Model(), 2)
    [line] = [str(w.message) for w in shown if getattr(w.message, "code", None) == "KEY-UNHASHABLE-ARG"]
    assert "`self`" in line and "`_lock` (lock)" in line and "leaves `_lock` out" in line


def test_a_keyword_argument_is_named_by_its_parameter(disk_cash):
    @disk_cash.cache
    def score(x, model=None):
        return x

    text = _full_warning(score, 1, model=Model())
    assert "the argument `model`, of type Model, could not be hashed" in text


def test_an_attribute_of_an_attribute_is_named_by_its_path(disk_cash):
    @disk_cash.cache
    def call(service):
        return service.retries

    text = _full_warning(call, Service())
    assert "its attribute `session.guard` (lock) cannot be hashed" in text
    assert "leaves `session` out" in text
    assert "return (self.retries,)`" in text, "a one-attribute key is written as a tuple"


def test_with_nothing_else_to_key_on_no_example_is_made_up(disk_cash):
    @disk_cash.cache
    def call(thing):
        return 1

    text = _full_warning(call, OnlyALock())
    assert "leaves `lock` out; otherwise register a hasher" in text
    assert "such as" not in text


def test_a_long_attribute_list_is_cut(disk_cash):
    @disk_cash.cache
    def call(thing):
        return 1

    assert "return (self.a, self.b, self.c, self.d, ...)`" in _full_warning(call, Wide())


def test_the_walk_reads_the_instance_dict_only(disk_cash):
    @disk_cash.cache
    def call(thing):
        return 1

    assert "its attribute `lock` (lock)" in _full_warning(call, Guarded())


def test_an_argument_that_is_itself_the_problem_reads_as_before(disk_cash):
    @disk_cash.cache
    def hold(lock):
        return 1

    text = _full_warning(hold, threading.Lock())
    assert "the argument `lock`, of type lock, could not be hashed, so this call" in text
    assert "its attribute" not in text and "leaves" not in text
