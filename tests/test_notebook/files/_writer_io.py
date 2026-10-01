"""A second stand-in project module: the low-level writer that
``_writer_lib.build_report`` reaches through ``_writer_io.write_table(...)``."""

from __future__ import annotations


def write_table(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"{row}\n" for row in rows)
