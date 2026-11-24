"""A cached call that binds an attribute to an object the instance already
held as another attribute, before the call and still after it, is stored,
and a hit binds that attribute to the caller's own object at that path.

nibabel's ``_get_data`` does ``img._data_cache = np.asanyarray(img._dataobj)``,
which for an in-memory image is the very array ``img._dataobj`` holds. That
was refused ("bound img._data_cache to an object that was there before the
call"), so every nilearn masker call on an image ran. A hit now leaves
``img._data_cache is img._dataobj``, exactly as plain Python does, with no
copy. Changing that object in place still runs every time.
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pytest

from cash import rewrite_in_place
from cash.decorator import attribute_replay
from cash.rewrite_in_place import Relink
from tests._scripts import run_python


class Img:
    """The shape of a nibabel image: an array and a lazily filled cache."""

    def __init__(self, data=None):
        self._dataobj = np.arange(6.0) if data is None else data
        self._data_cache = None
        self.extra = {"affine": [1, 2]}


def get_data(img):
    if img._data_cache is None:
        img._data_cache = np.asanyarray(img._dataobj)
    return float(img._data_cache.sum())


def get_data_and_scale(img):
    img._dataobj *= 2
    img._data_cache = img._dataobj
    return float(img._data_cache.sum())


def _state(img, result):
    """What a run left: each attribute in order, by content, and which
    attributes are one object."""
    names = list(vars(img))
    aliases = sorted((a, b) for a in names for b in names if a < b and vars(img)[a] is vars(img)[b])
    content = [(k, v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(img).items()]
    return content, aliases, result


def _cached(disk_cash, func):
    runs = []

    def body(img):
        runs.append(1)
        return func(img)

    body.__qualname__ = body.__name__ = func.__name__
    return disk_cash.cache(body), runs


def test_an_attribute_bound_to_one_the_instance_already_held_is_replayed(disk_cash):
    plain_img = Img()
    plain = _state(plain_img, get_data(plain_img))
    assert plain_img._data_cache is plain_img._dataobj
    cached, runs = _cached(disk_cash, get_data)
    for _ in range(3):
        img = Img()
        held = img._dataobj
        assert _state(img, cached(img)) == plain
        assert img._data_cache is img._dataobj is held
    assert len(runs) == 1


def test_without_the_c_watcher_the_alias_is_replayed_too(disk_cash, monkeypatch):
    monkeypatch.setattr(rewrite_in_place, "cwatch", None)
    cached, runs = _cached(disk_cash, get_data)
    for _ in range(2):
        img = Img()
        held = img._dataobj
        cached(img)
        assert img._data_cache is held and img._dataobj is held
    assert len(runs) == 1


def test_a_cached_bound_method_aliases_its_own_attribute(disk_cash):
    """``c.cache(img.get_data)``: the instance is ``self``, which the key folds."""
    runs = []

    def counted(self):
        runs.append(1)
        return get_data(self)

    plain_img = Img()
    plain = _state(plain_img, get_data(plain_img))
    for _ in range(3):
        img = Img()
        held = img._dataobj
        assert _state(img, disk_cash.cache(counted.__get__(img))()) == plain
        assert img._data_cache is held and img._dataobj is held
    assert len(runs) == 1


def test_a_caller_that_rebinds_the_held_attribute_gets_its_new_object_bound(disk_cash):
    cached, runs = _cached(disk_cash, get_data)
    first = Img()
    cached(first)
    img = Img()
    img._dataobj = np.arange(6.0)  # a new object, equal to the first
    new = img._dataobj
    plain_img = Img()
    plain_img._dataobj = np.arange(6.0)
    plain = _state(plain_img, get_data(plain_img))
    assert _state(img, cached(img)) == plain
    assert img._data_cache is new and img._dataobj is new
    assert first._data_cache is not new
    assert len(runs) == 1


def test_a_change_in_place_to_the_held_object_runs_every_time(disk_cash):
    plain_img = Img()
    plain = _state(plain_img, get_data_and_scale(plain_img))
    cached, runs = _cached(disk_cash, get_data_and_scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            img = Img()
            assert _state(img, cached(img)) == plain
            assert img._data_cache is img._dataobj
    assert len(runs) == 3
    why = cached.explain(Img()).details["why"]
    assert "also changed something else in its arguments in place" in why, why


def test_an_object_held_under_two_attributes_is_bound_again_by_identity(disk_cash):
    """``img.other`` holds the array ``img._dataobj`` holds: a hit binds
    ``img._data_cache`` to that one object, which both attributes of the
    image it is handed still hold."""

    def make():
        img = Img()
        img.other = img._dataobj
        return img

    plain_img = make()
    plain = _state(plain_img, get_data(plain_img))
    cached, runs = _cached(disk_cash, get_data)
    for _ in range(3):
        img = make()
        held = img._dataobj
        assert _state(img, cached(img)) == plain
        assert img._data_cache is img._dataobj is img.other is held
    assert len(runs) == 1


def test_attributes_that_held_one_object_and_hold_different_ones_now_are_not_resolved():
    img = Img()
    img.other = img._dataobj
    relink = Relink("attributes", (("img", "_dataobj"), ("img", "other")))
    assert attribute_replay.find(relink, {"img": img}) is img._dataobj
    img.other = img._dataobj.copy()
    with pytest.raises(attribute_replay.Unresolved, match=r"img\._dataobj and img\.other held .* different objects"):
        attribute_replay.find(relink, {"img": img})


def test_a_path_that_no_longer_holds_an_attribute_is_not_resolved():
    img = Img()
    relink = Relink("attribute", ("img", "_dataobj"))
    assert attribute_replay.find(relink, {"img": img}) is img._dataobj
    del img._dataobj
    with pytest.raises(attribute_replay.Unresolved, match=r"img\._dataobj.*not an attribute of img"):
        attribute_replay.find(relink, {"img": img})


@pytest.mark.timeout(120)
def test_a_second_process_binds_the_attribute_it_already_held(tmp_path):
    script = tmp_path / "get_data.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import numpy as np
            import cash

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
            LOG = sys.argv[2]

            class Img:
                def __init__(self):
                    self._dataobj = np.arange(6.0)
                    self._data_cache = None

            @c.cache
            def get_data(img):
                fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, b"ran\\n")
                os.close(fd)
                if img._data_cache is None:
                    img._data_cache = np.asanyarray(img._dataobj)
                return float(img._data_cache.sum())

            for _ in range(2):
                img = Img()
                held = img._dataobj
                assert get_data(img) == 15.0
                assert list(vars(img)) == ["_dataobj", "_data_cache"], vars(img)
                assert img._data_cache is held and img._dataobj is held
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "runs.log"
    for _ in range(2):
        out = run_python(script, tmp_path / ".cash", log, cwd=tmp_path)
        assert out.stdout.strip() == "ok", out.stderr
    assert log.read_text(encoding="utf-8").count("ran") == 1
