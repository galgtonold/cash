"""Which bare calls count as changing a setting a module keeps.

The integration arm is
``test_notebook_integration/test_module_settings_survive_a_restart.py``.
"""
import ast

import pytest

from cash.notebook.cacheability import module_setting_receivers


def _names(code):
    return module_setting_receivers(ast.parse(code))


@pytest.mark.parametrize("code, name", [
    ("plt.rcParams.update({'axes.grid': True})", "plt"),
    ("pd.set_option('display.max_columns', 3)", "pd"),
    ("plt.style.use('ggplot')", "plt"),
    ("np.seterr(all='raise')", "np"),
    ("warnings.filterwarnings('ignore')", "warnings"),
    ("sns.set_theme()", "sns"),
    ("sns.set()", "sns"),
    ("logging.basicConfig(level=10)", "logging"),
    ("sys.path.append('lib')", "sys"),
    ("os.environ.update({'A': '1'})", "os"),
    ("matplotlib.use('Agg')", "matplotlib"),
])
def test_a_setting_counts(code, name):
    assert _names(code) == {name}


@pytest.mark.parametrize("code", [
    "np.append(a, 1)",                   # a function on the module, returning a new array
    "np.random.seed(0)",                 # the seed has its own handling
    "time.sleep(1)",
    "plt.show()",
    "x = pd.set_option('display.width', 80)",   # not a bare call
    "for k in ks:\n    pd.set_option(k, 1)",     # not top level
    "print(pd.get_option('display.width'))",
])
def test_anything_else_does_not(code):
    assert _names(code) == frozenset()
