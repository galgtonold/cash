from cash.notebook.analysis import CodeAnalyzer

def test_function_def_scope():
    """Test that function definition isolates local variables."""
    code = """
def bla(x):
    r = 0
    for a in range(x):
        r += 1
    return r
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    
    # Inputs: range (global). x is argument (local). r, a are local.
    assert 'range' in inputs, "range should be detected as input"
    assert 'x' not in inputs, "x (argument) should not be input"
    assert 'r' not in outputs, "r (local) should not be output"
    assert 'a' not in outputs, "a (local loop var) should not be output"
    assert 'bla' in outputs, "bla (function name) should be output"
    
def test_global_access_in_function():
    """Test that global variables accessed in function are inputs."""
    code = """
def foo():
    return global_var + 1
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'global_var' in inputs
    assert 'foo' in outputs
    assert 'foo' not in inputs

def test_shadowing():
    """Test that local variables shadow globals."""
    code = """
x = 10
def foo():
    x = 5
    return x
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    # x is defined in block (outputs).
    assert 'x' in outputs
    assert 'foo' in outputs
    # Inside foo, x is assigned. So usage 'return x' uses local x.
    # Outer x=10 is assignment.
    # No external inputs.
    assert 'x' not in inputs 

def test_shadowing_argument():
    """Test that arguments shadow globals."""
    code = """
def foo(x):
    return x + y
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'x' not in inputs
    assert 'y' in inputs
    assert 'foo' in outputs

def test_class_scope():
    """Test class definition scope."""
    code = """
class MyClass:
    class_var = 1
    def method(self):
        return self.class_var
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'MyClass' in outputs
    assert 'class_var' not in outputs
    assert 'method' not in outputs
    assert 'self' not in inputs
    assert len(inputs) == 0

def test_aug_assign_local():
    """Test augmented assignment on local variable."""
    code = """
def func():
    x = 0
    x += 1
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'x' not in outputs
    assert 'x' not in inputs

def test_aug_assign_global():
    """Test augmented assignment on global variable (not defined in block)."""
    code = """
x += 1
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'x' in inputs # Read before write
    assert 'x' in outputs # Write

def test_loop_scope():
    """Test loop variables in global scope."""
    code = """
for i in range(10):
    pass
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'i' in outputs
    assert 'range' in inputs

def test_class_keyword_base_is_input():
    """A metaclass/keyword base expression is visited as an input (analysis.py 141)."""
    code = """
class Widget(metaclass=RegistryMeta):
    pass
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'Widget' in outputs
    assert 'RegistryMeta' in inputs, "metaclass keyword value should be detected as input"


def test_attribute_store_on_call_result():
    """Attribute store whose base is a call (no simple name) records no modified object.

    Exercises the falsy `_extract_base_name` branch in visit_Attribute (arc 216->222).
    """
    code = "get_obj().attr = 5"
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    # The base is a Call, so no module-level object name is captured as output.
    assert 'attr' not in outputs
    assert 'get_obj' in inputs


def test_import_scope():
    """Test imports."""
    code = """
import os
from math import sqrt
def foo():
    return sqrt(os.getpid())
    """
    inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert 'os' in outputs
    assert 'sqrt' in outputs
    assert 'foo' in outputs
    assert 'os' not in inputs # It is defined here
    assert 'sqrt' not in inputs


# --- Walrus operator (PEP 572 := / ast.NamedExpr) ---

def test_walrus_self_reference_is_input_and_output():
    """``n := n + 1`` reads the old n, so n is both an input and an output
    (mirrors ``n = n + 1``)."""
    inputs, outputs = CodeAnalyzer.analyze_code_block("if (n := n + 1):\n    pass")
    assert 'n' in inputs, "self-referential walrus must register n as an input"
    assert 'n' in outputs, "walrus target n must be an output"


def test_walrus_plain_binding_target_is_output():
    """``m = (d := base * 2)`` binds both m and d; base is the only input."""
    inputs, outputs = CodeAnalyzer.analyze_code_block("m = (d := base * 2)")
    assert inputs == {'base'}
    assert {'m', 'd'} <= outputs


def test_walrus_in_comprehension_binds_enclosing_scope():
    """PEP 572: a walrus inside a comprehension binds in the ENCLOSING scope,
    not the comprehension scope. ``[(t := t + i) for i in range(4)]`` must
    therefore expose t as both a cell-level input and output."""
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "vals = [(t := t + i) for i in range(4)]"
    )
    assert 't' in inputs, "walrus self-read t must be a cell input"
    assert 't' in outputs, "walrus target t must be a cell output (enclosing scope)"
    assert 'vals' in outputs
    assert 'i' not in outputs, "comprehension loop var i stays local"


def test_walrus_reassigned_names_includes_target():
    """A walrus target is a fresh rebinding, so reassigned_names includes it."""
    assert 'n' in CodeAnalyzer.reassigned_names("while (n := n - 1) > 0:\n    pass")
