"""Tests for the SideEffectDetector module.

Covers file write detection, network call detection, system call detection,
and the has_side_effects / get_side_effect_reasons convenience methods.
"""
import pytest
from cash.notebook.side_effects import SideEffectDetector, SideEffectInfo


class TestSideEffectDetectorFileWrites:
    """Detection of file-writing operations."""

    def test_open_write_mode(self):
        effects = SideEffectDetector.detect_side_effects("with open('out.txt', 'w') as f: f.write('x')")
        kinds = [e.kind for e in effects]
        assert 'file_write' in kinds

    def test_open_append_mode(self):
        effects = SideEffectDetector.detect_side_effects("open('log.txt', 'a')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_open_read_mode_not_detected(self):
        effects = SideEffectDetector.detect_side_effects("f = open('data.csv', 'r')")
        assert not any(e.kind == 'file_write' for e in effects)

    def test_os_remove(self):
        effects = SideEffectDetector.detect_side_effects("import os; os.remove('file.txt')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_os_makedirs(self):
        effects = SideEffectDetector.detect_side_effects("import os; os.makedirs('new_dir')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_shutil_copy(self):
        effects = SideEffectDetector.detect_side_effects("import shutil; shutil.copy('a', 'b')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_pandas_to_csv(self):
        effects = SideEffectDetector.detect_side_effects("df.to_csv('output.csv')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_pandas_to_parquet(self):
        effects = SideEffectDetector.detect_side_effects("df.to_parquet('data.parquet')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_json_dump(self):
        effects = SideEffectDetector.detect_side_effects("import json; json.dump(data, f)")
        assert any(e.kind == 'file_write' for e in effects)

    def test_pickle_dump(self):
        effects = SideEffectDetector.detect_side_effects("import pickle; pickle.dump(obj, f)")
        assert any(e.kind == 'file_write' for e in effects)

    def test_savefig(self):
        effects = SideEffectDetector.detect_side_effects("plt.savefig('plot.png')")
        assert any(e.kind == 'file_write' for e in effects)

    def test_numpy_save(self):
        effects = SideEffectDetector.detect_side_effects("np.save('arr.npy', data)")
        assert any(e.kind == 'file_write' for e in effects)

    def test_write_method(self):
        effects = SideEffectDetector.detect_side_effects("f.write('content')")
        assert any(e.kind == 'file_write' for e in effects)


class TestSideEffectDetectorNetworkAndSystem:
    """Detection of network and system-level operations."""

    def test_subprocess_run(self):
        effects = SideEffectDetector.detect_side_effects("import subprocess; subprocess.run(['ls'])")
        assert any(e.kind == 'system' for e in effects)

    def test_subprocess_popen(self):
        effects = SideEffectDetector.detect_side_effects("subprocess.Popen(['cmd'])")
        assert any(e.kind == 'system' for e in effects)

    def test_os_system(self):
        effects = SideEffectDetector.detect_side_effects("os.system('rm -f file')")
        assert any(e.kind == 'system' for e in effects)

    def test_requests_post(self):
        effects = SideEffectDetector.detect_side_effects("requests.post('https://api.example.com', data=payload)")
        assert any(e.kind == 'network' for e in effects)

    def test_requests_delete(self):
        effects = SideEffectDetector.detect_side_effects("requests.delete(url)")
        assert any(e.kind == 'network' for e in effects)


class TestSideEffectDetectorConvenienceMethods:
    """has_side_effects and get_side_effect_reasons convenience methods."""

    def test_has_side_effects_true(self):
        assert SideEffectDetector.has_side_effects("df.to_csv('output.csv')") is True

    def test_has_side_effects_false(self):
        assert SideEffectDetector.has_side_effects("x = 1 + 2") is False

    def test_get_side_effect_reasons_nonempty(self):
        reasons = SideEffectDetector.get_side_effect_reasons("os.remove('file.txt')")
        assert len(reasons) > 0
        assert any('file_write' in r or 'remove' in r for r in reasons)

    def test_get_side_effect_reasons_empty_for_pure_code(self):
        reasons = SideEffectDetector.get_side_effect_reasons("result = [x**2 for x in range(10)]")
        assert reasons == []

    def test_syntax_error_returns_empty(self):
        effects = SideEffectDetector.detect_side_effects("def (:")
        assert effects == []

    def test_pre_parsed_tree(self):
        import ast
        code = "df.to_csv('output.csv')"
        tree = ast.parse(code)
        effects = SideEffectDetector.detect_side_effects(code, tree=tree)
        assert any(e.kind == 'file_write' for e in effects)

    def test_line_numbers_captured(self):
        code = "df.to_csv('out.csv')\nx = 1"
        effects = SideEffectDetector.detect_side_effects(code)
        assert effects[0].line == 1

    def test_multiple_effects_detected(self):
        code = "os.remove('a')\ndf.to_csv('b')"
        effects = SideEffectDetector.detect_side_effects(code)
        assert len(effects) >= 2
