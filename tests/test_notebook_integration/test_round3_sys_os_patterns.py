"""Sys/OS interaction — cash caching with sys, os, platform operations."""

import textwrap

import pytest


@pytest.mark.stress
class TestSysPatterns:
    """Test sys module interaction."""

    def test_sys_path_manipulation(self, nb_runner, tmp_path):
        """sys.path append and import from custom path."""
        mod_dir = tmp_path / "custom_lib"
        mod_dir.mkdir()
        (mod_dir / "helper.py").write_text("VALUE = 42\ndef compute(x): return x * VALUE\n")
        mod_path = str(mod_dir).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, '{mod_path}')
            """),
                textwrap.dedent("""\
                import helper
                result = helper.compute(3)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=126" in nb_runner.get_output(2)


@pytest.mark.stress
class TestOsPatterns:
    """Test os module interaction."""

    def test_os_environ_read(self, nb_runner):
        """Read environment variables across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import os
                os.environ['CASH_TEST_VAR'] = 'hello_cash'
                val = os.environ.get('CASH_TEST_VAR', 'missing')
                print(f"val={val}")
            """),
                textwrap.dedent("""\
                val2 = os.environ.get('CASH_TEST_VAR', 'missing')
                print(f"val2={val2}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=hello_cash" in nb_runner.get_output(1)
        assert "val2=hello_cash" in nb_runner.get_output(2)

    def test_os_path_operations(self, nb_runner, tmp_path):
        """os.path operations across cells."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "import os",
                textwrap.dedent(f"""\
                base = '{dir_path}'
                full = os.path.join(base, 'subdir', 'file.txt')
                dirname = os.path.dirname(full)
                basename = os.path.basename(full)
                print(f"basename={{basename}}")
            """),
                textwrap.dedent("""\
                ext = os.path.splitext(basename)[1]
                print(f"ext={ext}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "basename=file.txt" in nb_runner.get_output(2)
        assert "ext=.txt" in nb_runner.get_output(3)

    def test_tempdir_creation(self, nb_runner, tmp_path):
        """Create temp directories and track files."""
        work_dir = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "import os\nimport tempfile",
                textwrap.dedent(f"""\
                td = tempfile.mkdtemp(dir='{work_dir}')
                for i in range(3):
                    with open(os.path.join(td, f'file_{{i}}.txt'), 'w') as f:
                        f.write(f'content_{{i}}')
                file_count = len(os.listdir(td))
                print(f"files={{file_count}}")
            """),
                textwrap.dedent("""\
                contents = []
                for fn in sorted(os.listdir(td)):
                    with open(os.path.join(td, fn)) as f:
                        contents.append(f.read())
                print(f"contents={contents}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "files=3" in nb_runner.get_output(2)
        assert "content_0" in nb_runner.get_output(3)
