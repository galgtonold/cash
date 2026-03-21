"""
Integration tests for @cash.cache decorator invalidation when
a dependency function changes in a module file.

Scenarios:
- Module has dep() and fun() decorated with @cash.cache
- fun() calls dep()
- After changing dep() source code, decorator should cache-miss
  both for dep() and for fun() (transitive invalidation)
"""
import pytest
import time


@pytest.mark.integration
@pytest.mark.modules
class TestDecoratorModuleReloadSameCell:
    """Tests for decorator cache invalidation when import + call are in the SAME cell."""

    def test_same_cell_import_and_call_invalidates_on_dep_change(self, nb_runner, tmp_path):
        """
        Reproduces the user's exact scenario:
        - Cell 1: sys.path setup
        - Cell 2: import calc; calc.fun(...) — import + calls in SAME cell
        - User modifies dep() in module file
        - Re-run ONLY cell 2 → fun() should cache-miss because dep() changed

        This is the critical user-reported bug: re-running the same cell that
        has both the import and the function calls doesn't invalidate the
        decorator's internal cache.
        """
        mod_dir = tmp_path / "mymod"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: Setup path
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            # Cell 2: Import + call in SAME cell (user's pattern)
            "import calc\nresult = calc.fun(5, 1)\nprint('result=' + str(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # First run: dep(5) = 6, fun(5,1) = 5+1+6 = 12
        output = nb_runner.get_output(2)
        assert "result=12" in output, f"Expected result=12, got: {output}"

        # Change dep() in the module
        time.sleep(0.5)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100  # CHANGED

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        # Re-run ONLY cell 2 (same cell has import + call)
        nb_runner.run_cell(2)
        output2 = nb_runner.get_output(2)
        assert "result=111" in output2, (
            f"Expected result=111 after dep change (decorator should invalidate), got: {output2}"
        )

    def test_same_cell_multiple_calls_all_invalidate(self, nb_runner, tmp_path):
        """
        Similar to above but with multiple calls to fun() (like the user's
        50-call list comprehension). All calls should cache-miss after dep() changes.
        """
        mod_dir = tmp_path / "mymod2"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc2.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            # Cell 2: import + multiple calls in same cell
            "import calc2\nresults = [calc2.fun(5, i % 3) for i in range(10)]\nprint('sum=' + str(sum(results)))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # First run: dep(5)=6, fun(5,0)=5+0+6=11, fun(5,1)=5+1+6=12, fun(5,2)=5+2+6=13
        # Pattern: 11,12,13,11,12,13,11,12,13,11 → sum = 11*4+12*3+13*3 = 44+36+39 = 119
        output = nb_runner.get_output(2)
        assert "sum=119" in output, f"Expected sum=119, got: {output}"

        # Change dep
        time.sleep(0.5)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100  # CHANGED

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        # Re-run ONLY cell 2
        nb_runner.run_cell(2)
        output2 = nb_runner.get_output(2)
        # dep(5)=105, fun(5,0)=5+0+105=110, fun(5,1)=111, fun(5,2)=112
        # Pattern: 110,111,112,110,111,112,110,111,112,110 → sum = 110*4+111*3+112*3 = 440+333+336 = 1109
        assert "sum=1109" in output2, (
            f"Expected sum=1109 after dep change, got: {output2}"
        )

    def test_same_cell_dep_and_import_already_imported(self, nb_runner, tmp_path):
        """
        Scenario where the module was already imported in a previous cell,
        then a later cell re-imports and calls. This is the user's exact pattern:
        Cell 30 does the first import, Cell 35 re-imports and calls.
        """
        mod_dir = tmp_path / "mymod3"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc3.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: sys.path + first import
            f"import sys; sys.path.insert(0, '{mod_dir_str}')\nimport calc3",
            # Cell 2: Some other code using the module
            "first_result = calc3.dep(5)\nprint('dep=' + str(first_result))",
            # Cell 3: Re-import + call in same cell (user's cell 35 pattern)
            "import calc3\nresult = calc3.fun(5, 1)\nprint('result=' + str(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # First run: dep(5) = 6, fun(5,1) = 12
        assert "dep=6" in nb_runner.get_output(2)
        assert "result=12" in nb_runner.get_output(3)

        # Change dep() in the module
        time.sleep(0.5)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100  # CHANGED

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        # Re-run ONLY cell 3 (the one with import + call)
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)
        assert "result=111" in output3, (
            f"Expected result=111 after dep change (re-running import+call cell), got: {output3}"
        )

    def test_qualname_collision_notebook_and_module(self, nb_runner, tmp_path):
        """
        Reproduces the user bug: the notebook defines its own dep() and
        fun() with @cash.cache (same qualnames as module's dep/fun), then
        a later cell imports the module and calls module.fun().

        Before the module-qualified keys fix, the qualname collision
        ("dep" → "dep") caused the decorator's source_hashes to be
        overwritten by the notebook's function, so changes to the
        module's dep() were invisible to the cache key.
        """
        mod_dir = tmp_path / "mymod_collision"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc_col.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: sys.path setup
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            # Cell 2: Notebook defines its OWN dep() and fun() with same names
            "import cash\n@cash.cache\ndef dep(a):\n    return a + 1\n\n@cash.cache\ndef fun(a, b):\n    return a + b + dep(a)",
            # Cell 3: Call notebook's fun to warm its cache
            "nb_result = fun(5, 1)\nprint('nb_result=' + str(nb_result))",
            # Cell 4: Import module + call module's fun (same cell)
            "import calc_col\nresult = calc_col.fun(5, 1)\nprint('mod_result=' + str(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Both should compute the same result initially
        assert "nb_result=12" in nb_runner.get_output(3)
        assert "mod_result=12" in nb_runner.get_output(4)

        # Now change ONLY the module's dep() — the notebook's dep() is unchanged
        time.sleep(0.5)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100  # CHANGED in module

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        # Re-run cell 4 (import + module call)
        nb_runner.run_cell(4)
        output4 = nb_runner.get_output(4)
        
        # Module's dep(5) should now return 105, so module's fun(5,1) = 5+1+105 = 111
        assert "mod_result=111" in output4, (
            f"Expected mod_result=111 after module dep change, "
            f"but got: {output4}. "
            f"This likely means the qualname collision between notebook's dep() "
            f"and module's dep() caused incorrect cache key computation."
        )


@pytest.mark.integration
@pytest.mark.modules
class TestDecoratorModuleReloadInvalidation:

    def test_decorator_cache_invalidates_on_dep_change(self, nb_runner, tmp_path):
        """
        When dep() changes in a module, fun() which calls dep() should
        also see a decorator cache miss because fun's behavior depends
        on dep transitively.
        """
        mod_dir = tmp_path / "mymod"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: Setup path
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            # Cell 2: Import module
            "import calc",
            # Cell 3: Call fun and print result
            "result = calc.fun(5, 1)\nprint('result=' + str(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # First run: dep(5) = 5+1 = 6, fun(5,1) = 5+1+6 = 12
        output = nb_runner.get_output(3)
        assert "result=12" in output, f"Expected result=12, got: {output}"

        # Change dep() in the module
        time.sleep(0.5)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100  # CHANGED

@cash.cache
def fun(a, b):
    return a + b + dep(a)
''')

        # Re-run: dep(5) should now = 105, fun(5,1) = 5+1+105 = 111
        nb_runner.run_cells([1, 2, 3])
        out3 = nb_runner.get_output(3)
        assert "result=111" in out3, (
            f"Expected result=111 after dep change (transitive invalidation), got: {out3}"
        )

    def test_decorator_cache_dep_direct_call_invalidates(self, nb_runner, tmp_path):
        """
        When dep() changes, calling dep() directly should also see a miss.
        """
        mod_dir = tmp_path / "mymod2"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc2.py"
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 1
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            "import calc2",
            "result = calc2.dep(5)\nprint('result=' + str(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=6" in output, f"Expected result=6, got: {output}"

        # Change dep
        time.sleep(0.1)
        mod_file.write_text('''
import cash

@cash.cache
def dep(a):
    return a + 100
''')

        nb_runner.run_cells([2, 3])
        output2 = nb_runner.get_output(3)
        assert "result=105" in output2, (
            f"Expected result=105 after dep change, got: {output2}"
        )

    def test_decorator_unchanged_fun_preserved_granular(self, nb_runner, tmp_path):
        """
        When only a helper function changes but a completely independent
        function doesn't use it, the independent function should be preserved
        (granular invalidation).
        """
        mod_dir = tmp_path / "mymod3"
        mod_dir.mkdir()
        mod_file = mod_dir / "calc3.py"
        mod_file.write_text('''
import cash

@cash.cache
def helper(a):
    return a + 1

@cash.cache
def independent(a):
    return a * 2
''')

        mod_dir_str = str(mod_dir).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys; sys.path.insert(0, '{mod_dir_str}')",
            "import calc3",
            "r1 = calc3.helper(5)\nr2 = calc3.independent(5)\nprint('helper=' + str(r1) + ',independent=' + str(r2))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "helper=6" in output, f"Expected helper=6, got: {output}"
        assert "independent=10" in output, f"Expected independent=10, got: {output}"

        # Change only helper — independent should be preserved
        time.sleep(0.1)
        mod_file.write_text('''
import cash

@cash.cache
def helper(a):
    return a + 100  # CHANGED

@cash.cache
def independent(a):
    return a * 2  # UNCHANGED
''')

        nb_runner.run_cells([1, 2, 3])
        output2 = nb_runner.get_output(3)
        assert "helper=105" in output2, f"Expected helper=105 after change, got: {output2}"
        assert "independent=10" in output2, f"Expected independent=10 (preserved), got: {output2}"
