"""A read credited to code after an equal code is gone is kept.

Reads are remembered per code, in a map held weakly by the code: two equal
codes (one source compiled twice) share one entry. A read credited again
to the same code takes the entry the last lookup found, without hashing
and comparing the code again, but only while no entry of the map has gone
since: when the code that made the entry goes, the entry goes with it,
and a read written into it then would be lost to `credited_reads`.
"""

from __future__ import annotations

import gc

from cash.tracking import read_credit


def test_the_read_after_the_entry_went_is_credited():
    source = "def load():\n    return 1\n"
    first = compile(source, "<one source>", "exec")
    second = compile(source, "<one source>", "exec")
    assert first == second and first is not second
    read_credit._record_read(first, "/data/a.csv", ("a",))
    read_credit._record_read(second, "/data/b.csv", ("b",))  # the entry *first* made
    assert set(read_credit.credited_reads(second)) == {"/data/a.csv", "/data/b.csv"}
    del first
    gc.collect()
    read_credit._record_read(second, "/data/c.csv", ("c",))
    assert read_credit.credited_reads(second) == {"/data/c.csv": ("c",)}


def test_a_read_credited_again_goes_into_the_codes_entry():
    code = compile("x = 1\n", "<again>", "exec")
    for i in range(3):
        read_credit._record_read(code, f"/data/{i}.csv", (i,))
    assert read_credit.credited_reads(code) == {f"/data/{i}.csv": (i,) for i in range(3)}


def test_a_new_code_on_a_gone_codes_id_gets_its_own_entry():
    """Code equal to a live code is credited, then goes; a new, unequal code
    takes its id. Its reads are its own, not the live code's entry."""
    for i in range(300):
        source = f"k = {i}\n"
        kept = compile(source, "<kept>", "exec")
        read_credit._record_read(kept, "/data/kept.csv", ("kept",))
        equal = compile(source, "<kept>", "exec")
        read_credit._record_read(equal, "/data/equal.csv", ("equal",))
        gone = id(equal)
        entry = read_credit._found_reads[gone]  # the lookup's: *kept*'s reads
        del equal
        # Unequal code of the same size, made until one takes the gone id.
        held = []
        for j in range(64):
            new = compile(f"k = {100_000 + j}\n", "<new>", "exec")
            if id(new) == gone:
                break
            held.append(new)
        if id(new) != gone:
            # Which freed block the allocator hands out next differs by
            # platform and by what ran before (on macOS, 3.14, none of 300
            # took it once): the entry is put where a reuse of the id leaves it.
            read_credit._found_reads[id(new)] = entry
        assert read_credit._found_reads[id(new)] is entry  # the case
        read_credit._record_read(new, "/data/new.csv", ("new",))
        assert read_credit.credited_reads(new) == {"/data/new.csv": ("new",)}
        assert "/data/new.csv" not in read_credit.credited_reads(kept)
        del kept, new, held, entry
