"""``os.walk(FOLDER)`` and ``Path(FOLDER).walk()`` read the folder, as
``os.listdir`` does: a cell listing an output folder with ``os.walk`` was not
seen reading it, so the export cell filling it was not run again for it after
an edit above, and the listing showed the old files."""

from pathlib import Path

from cash.analysis.namespace_effects import statement_read_paths

NS = {"OUT": "out", "Path": Path}


def test_os_walk_reads_its_folder():
    assert statement_read_paths("files = [f for _, _, fs in os.walk('out') for f in fs]") == {"out"}
    assert statement_read_paths("files = list(os.walk(top=OUT))", namespace=NS) == {"out"}


def test_path_walk_reads_its_folder():
    assert statement_read_paths("files = list(Path('out').walk())", namespace=NS) == {"out"}


def test_a_walk_of_a_syntax_tree_reads_no_file():
    assert statement_read_paths("names = [n for n in ast.walk(tree)]") == set()
