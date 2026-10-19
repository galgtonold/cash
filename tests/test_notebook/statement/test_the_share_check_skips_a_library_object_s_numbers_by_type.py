"""The share check reads down a library object for what something else holds
of it (`shared_objects._held_from_outside`). ``with tarfile.open(...) as tar:
tar.extractall()`` binds a TarFile holding a TarInfo per member, and each
holds a dozen names, sizes, modes and its class: 380,000 objects for 20,000
members, each asked about in a call that answered "a value" from its type.
Numbers, strings and classes are now passed over by their type in the walk
itself, and the answer is the same.
"""

from __future__ import annotations

import io
import tarfile

import cash.notebook.shared_objects as shared_objects

MEMBERS = 200


def _tar() -> tarfile.TarFile:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as out:
        for i in range(MEMBERS):
            info = tarfile.TarInfo(f"part{i}.txt")
            info.size = 1
            out.addfile(info, io.BytesIO(b"x"))
    data.seek(0)
    tar = tarfile.open(fileobj=data, mode="r")
    tar.getmembers()
    return tar


def test_a_tarfile_s_members_are_read_without_a_call_per_number(monkeypatch):
    tar = _tar()
    asked = []
    real = shared_objects._read_down
    monkeypatch.setattr(shared_objects, "_read_down", lambda obj, types: asked.append(type(obj)) or real(obj, types))
    inside, owners = shared_objects._held_from_outside({id(tar): tar}, {id(tar): "tar"}, shared_objects.VALUE_TYPES)
    assert inside == {} and owners == set()
    # A member and its header dict each; nothing they hold that is a number,
    # a string or a class.
    assert len(asked) < 3 * MEMBERS
    # (Types, not the objects: a reference kept here would hold them from outside.)
    assert not {int, str, bytes, type, type(None)} & set(asked)
