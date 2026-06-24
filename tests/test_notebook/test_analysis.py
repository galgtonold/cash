import math
from cash.notebook.analysis import CodeAnalyzer
from cash import Cash

# Dummy functions for testing
def func_a():
    pass

def func_b():
    func_a()

def func_with_import():
    return math.sqrt(4)

from math import sqrt as my_sqrt
def func_with_alias():
    return my_sqrt(4)


class _AmbiguousBool:
    """Stand-in for pandas DataFrame / numpy ndarray: ``__bool__`` raises
    instead of returning True/False."""
    def __bool__(self):
        raise ValueError("The truth value is ambiguous")

    def method(self):
        return None

# A module global bound to an object with a raising __bool__, referenced by the
# function below — mirrors a decorated function that calls df.sum() while `df`
# (a DataFrame) lives in its globals.
_ambiguous_global = _AmbiguousBool()
def func_references_ambiguous():
    return _ambiguous_global.method()

# Module-qualified key for func_a (matches the format returned by find_called_functions)
_func_a_key = Cash._get_func_key(func_a)


def test_simple_call():
    '''Test that simple function calls are detected.'''
    deps = CodeAnalyzer.find_called_functions(func_b)
    # func_a is in globals, so it should be resolved with module-qualified key
    assert _func_a_key in deps, f'Should detect func_a ({_func_a_key}) called by func_b, got {deps}'


def test_known_functions_filter():
    '''Test that known_functions filter works correctly.'''
    # When func_a is known (using module-qualified key), it should be in dependencies
    known = {_func_a_key: func_a}
    deps = CodeAnalyzer.find_called_functions(func_b, known)
    assert _func_a_key in deps, f'Should include func_a when it is in known functions, got {deps}'

    # When func_a is not known, it should not be in dependencies
    known = {}
    deps = CodeAnalyzer.find_called_functions(func_b, known)
    assert _func_a_key not in deps, 'Should not include func_a when it is not in known functions'


def test_import_call():
    '''Test that imported module functions are detected.'''
    # math.sqrt is a builtin function
    deps = CodeAnalyzer.find_called_functions(func_with_import)
    # The analyzer returns all resolved qualnames
    # This test is intentionally left as a placeholder for future analysis logic
    # Currently just verifying it doesn't crash
    assert isinstance(deps, set), 'Should return a set of dependencies'


def test_alias_call():
    '''Test that aliased imports are detected.'''
    # Should resolve to 'sqrt' (the qualname of the object)
    deps = CodeAnalyzer.find_called_functions(func_with_alias)
    # This test is intentionally left as a placeholder for future analysis logic
    # Currently just verifying it doesn't crash
    assert isinstance(deps, set), 'Should return a set of dependencies'


def test_find_called_functions_handles_ambiguous_bool_globals():
    '''Analyzing a function whose globals hold an object with a raising
    __bool__ (DataFrame/ndarray) must not crash. Regression for the
    @cash.cache + DataFrame "truth value is ambiguous" failure.'''
    result = CodeAnalyzer.find_called_functions(func_references_ambiguous)
    assert isinstance(result, set)
