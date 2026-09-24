"""os.path and pathlib path manipulation across cells."""

import textwrap

import pytest


@pytest.mark.stress
class TestSysPatterns:
    """Test sys module interaction."""

    def test_sys_path_manipulation(self, nb_runner, tmp_path):
        """sys.path append and import from custom path."""
        mod_dir = tmp_path / "custom_lib"
        mod_dir.mkdir()
        (mod_dir / "helper.py").write_text("VALUE = 42\ndef compute(x): return x * VALUE\n", encoding="utf-8")
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


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOsPathPathlibCross:
    """Test os.path and pathlib cross-usage across cells."""

    def test_path_cross_usage(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: build paths with both APIs
                "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('home', 'user', 'docs', 'file.txt')\npp = PurePosixPath('home') / 'user' / 'docs' / 'file.txt'\nprint(f'op_base={os.path.basename(op)}')\nprint(f'pp_name={pp.name}')",
                # Cell 2: extract components
                "op_dir = os.path.dirname(op)\npp_parent = str(pp.parent)\nop_ext = os.path.splitext(op)[1]\npp_suffix = pp.suffix\nprint(f'op_dir={op_dir}')\nprint(f'pp_parent={pp_parent}')\nprint(f'ext_match={op_ext == pp_suffix}')",
                # Cell 3: manipulate
                "new_pp = pp.with_suffix('.md')\nnew_op = os.path.splitext(op)[0] + '.md'\nprint(f'new_pp={new_pp}')\nprint(f'names_match={new_pp.name == os.path.basename(new_op)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "op_base=file.txt" in out1
        assert "pp_name=file.txt" in out1
        out2 = nb_runner.get_output(2)
        assert "ext_match=True" in out2
        out3 = nb_runner.get_output(3)
        assert "names_match=True" in out3

    def test_path_edit_propagation(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('home', 'user', 'docs', 'file.txt')\npp = PurePosixPath('home') / 'user' / 'docs' / 'file.txt'\nprint(f'pp_name={pp.name}')",
                "parts_count = len(pp.parts)\nprint(f'parts={parts_count}')",
                "depth = op.count(os.sep) + op.count('/')\nprint(f'depth={depth}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=4" in nb_runner.get_output(2)

        # Change base path depth
        nb_runner.set_cell_source(
            1,
            "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('srv', 'data', 'output.csv')\npp = PurePosixPath('srv') / 'data' / 'output.csv'\nprint(f'pp_name={pp.name}')",
        )
        nb_runner.run_cells([1, 2, 3])
        assert "pp_name=output.csv" in nb_runner.get_output(1)
        assert "parts=3" in nb_runner.get_output(2)

    def test_path_cache_correctness(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import os\nfrom pathlib import PurePosixPath\npp = PurePosixPath('a') / 'b' / 'c.txt'\nprint(f'stem={pp.stem}')",
                "result = pp.with_name('d.csv')\nprint(f'new_name={result.name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stem=c" in nb_runner.get_output(1)
        assert "new_name=d.csv" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "new_name=d.csv" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestOsPathPathlib:
    """os.path and pathlib manipulation."""

    def test_os_path_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import os.path",
                "p = '/home/user/docs/file.txt'\nd = os.path.dirname(p)\nb = os.path.basename(p)\nname, ext = os.path.splitext(b)\nprint(f'd={d} b={b} name={name} ext={ext}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=/home/user/docs" in out
        assert "b=file.txt" in out
        assert "name=file" in out
        assert "ext=.txt" in out

    def test_pathlib_parts(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath",
                "p = PurePosixPath('/usr/local/bin/app')\nprint(f'parent={p.parent} stem={p.stem} suffix={p.suffix} name={p.name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "parent=/usr/local/bin" in out
        assert "stem=app" in out
        assert "name=app" in out

    def test_path_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import os.path",
                "p = '/a/b/c.py'\nprint(f'ext={os.path.splitext(p)[1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ext=.py" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "p = '/x/y/data.csv'\nprint(f'ext={os.path.splitext(p)[1]}')")
        nb_runner.run_all()
        assert "ext=.csv" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPathManipulation:
    """os.path and pathlib path manipulation edits."""

    def test_pathlib_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\np = PurePosixPath('/home/user/docs/file.txt')",
                "parts = list(p.parts)\nstem = p.stem\nsuffix = p.suffix\nparent = str(p.parent)\nprint(f'stem={stem} suffix={suffix} parent={parent}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=file" in out
        assert "suffix=.txt" in out
        assert "parent=/home/user/docs" in out

    def test_pathlib_edit_path(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\np = PurePosixPath('/a/b/c.py')",
                "new_p = p.with_suffix('.txt')\nresult = str(new_p)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=/a/b/c.txt" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from pathlib import PurePosixPath\np = PurePosixPath('/x/y/data.csv')")
        nb_runner.run_all()
        assert "result=/x/y/data.txt" in nb_runner.get_output(2)

    def test_path_join_resolve(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\nbase = PurePosixPath('/project')\nsub = 'src/main.py'",
                "full = base / sub\nname = full.name\nprint(f'full={full} name={name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "full=/project/src/main.py" in out
        assert "name=main.py" in out


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPathlibInteraction:
    """Test pathlib/os.path patterns with cache invalidation."""

    def test_pathlib_join_edit(self, nb_runner):
        """Editing path components should propagate."""
        nb_runner.create_notebook(
            [
                "from pathlib import Path\nbase = Path('/home/user')",
                "full = base / 'documents' / 'report.txt'",
                "result = str(full)",
                "print(f'path={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        # pathlib normalizes to OS-specific separator
        assert "report.txt" in out
        assert "documents" in out

        nb_runner.set_cell_source(1, "from pathlib import Path\nbase = Path('/srv/data')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "report.txt" in out
        assert "data" in out

    def test_path_parts_edit(self, nb_runner):
        """Editing a path and using its parts should propagate."""
        nb_runner.create_notebook(
            [
                "from pathlib import Path\np = Path('/a/b/c/file.txt')",
                "stem = p.stem\nsuffix = p.suffix\nparent_name = p.parent.name",
                "info = f'{stem}{suffix} in {parent_name}'",
                "print(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=file.txt in c" in out

        nb_runner.set_cell_source(1, "from pathlib import Path\np = Path('/x/y/z/data.csv')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=data.csv in z" in out

    def test_os_path_manipulation_edit(self, nb_runner):
        """Editing os.path operations should propagate."""
        nb_runner.create_notebook(
            [
                "import os.path\ndir_part = '/home/user'\nfile_part = 'notes.md'",
                "joined = os.path.join(dir_part, file_part)",
                "base = os.path.basename(joined)\next = os.path.splitext(base)[1]",
                "print(f'base={base},ext={ext}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "base=notes.md,ext=.md" in out

        nb_runner.set_cell_source(1, "import os.path\ndir_part = '/var/log'\nfile_part = 'access.log'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "base=access.log,ext=.log" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPathlibOps:
    """pathlib operations for path manipulation."""

    def test_pathlib_parts(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\np = PurePosixPath('/home/user/docs/file.txt')",
                "stem = p.stem\nsuffix = p.suffix\nparent = str(p.parent)\nprint(f'stem={stem} suffix={suffix} parent={parent}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=file" in out
        assert "suffix=.txt" in out
        assert "parent=/home/user/docs" in out

    def test_pathlib_join(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\nbase = PurePosixPath('/data')",
                "full = base / 'subdir' / 'output.csv'\nprint(f'full={full}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "full=/data/subdir/output.csv" in nb_runner.get_output(2)

    def test_pathlib_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath\nf = PurePosixPath('report.pdf')",
                "new_f = f.with_suffix('.docx')\nprint(f'new={new_f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "new=report.docx" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from pathlib import PurePosixPath\nf = PurePosixPath('data.csv')")
        nb_runner.run_all()
        assert "new=data.docx" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPathlibPathOps:
    """Test pathlib Path operations across cells."""

    def test_pathlib_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: path construction and parts
                "from pathlib import PurePosixPath as PP\np = PP('/home/user/documents/report.pdf')\nprint(f'name={p.name}')\nprint(f'stem={p.stem}')\nprint(f'suffix={p.suffix}')\nprint(f'parts={p.parts}')",
                # Cell 2: parent and ancestors
                "parent = p.parent\ngrandparent = p.parent.parent\nprint(f'parent={parent}')\nprint(f'grandparent={grandparent}')",
                # Cell 3: path manipulation
                "new_path = p.with_suffix('.txt')\nrenamed = p.with_name('summary.docx')\nprint(f'new_suffix={new_path}')\nprint(f'renamed={renamed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "name=report.pdf" in out1
        assert "stem=report" in out1
        assert "suffix=.pdf" in out1
        out2 = nb_runner.get_output(2)
        assert "parent=/home/user/documents" in out2
        assert "grandparent=/home/user" in out2
        out3 = nb_runner.get_output(3)
        assert "new_suffix=/home/user/documents/report.txt" in out3
        assert "renamed=/home/user/documents/summary.docx" in out3

    def test_pathlib_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath as PP\np = PP('/data/input.csv')\ninfo = f'{p.stem}{p.suffix}'\nprint(f'info={info}')",
                "full = str(p)\nprint(f'full={full}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=input.csv" in nb_runner.get_output(1)
        assert "full=/data/input.csv" in nb_runner.get_output(2)

        # Edit path
        nb_runner.set_cell_source(
            1,
            "from pathlib import PurePosixPath as PP\np = PP('/output/results.json')\ninfo = f'{p.stem}{p.suffix}'\nprint(f'info={info}')",
        )
        nb_runner.run_cells([1, 2])
        assert "info=results.json" in nb_runner.get_output(1)
        assert "full=/output/results.json" in nb_runner.get_output(2)

    def test_pathlib_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from pathlib import PurePosixPath as PP\np = PP('/a/b/c/d.txt')\ndepth = len(p.parts) - 1  # minus root\nprint(f'depth={depth}')",
                "is_deep = depth > 2\nprint(f'is_deep={is_deep}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth=4" in nb_runner.get_output(1)
        assert "is_deep=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_deep=True" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestPathLibPatterns:
    """Test pathlib usage across cells."""

    def test_pathlib_operations(self, nb_runner, tmp_path):
        """pathlib.Path operations across cells."""
        path_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                f"base = Path('{path_str}')",
                textwrap.dedent("""\
                # Create some files
                (base / 'a.txt').write_text('hello')
                (base / 'b.txt').write_text('world')
                (base / 'c.py').write_text('# code')
            """),
                textwrap.dedent("""\
                txt_files = sorted(base.glob('*.txt'))
                print(f"txt_count={len(txt_files)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "txt_count=2" in nb_runner.get_output(4)

    def test_pathlib_change_propagation(self, nb_runner, tmp_path):
        """Change base path → downstream glob updates."""
        sub1 = tmp_path / "sub1"
        sub2 = tmp_path / "sub2"
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / "f1.txt").write_text("a", encoding="utf-8")
        (sub2 / "f2.txt").write_text("b", encoding="utf-8")
        (sub2 / "f3.txt").write_text("c", encoding="utf-8")

        path1 = str(sub1).replace("\\", "/")
        path2 = str(sub2).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                f"base = Path('{path1}')",
                textwrap.dedent("""\
                count = len(list(base.glob('*.txt')))
                print(f"count={count}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, f"base = Path('{path2}')")
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(3)
