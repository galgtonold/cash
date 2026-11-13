"""A Windows extension rebuilt from the same code keeps its digest.

MSVC stamps the link time into a ``.pyd``: the COFF header's TimeDateStamp,
each debug-directory entry's, and with ``/DEBUG`` a new GUID and age in the
CodeView record. Hashing the whole file made every rebuild of an unchanged
Cython module a miss. Those fields (and the CheckSum that covers them) are
hashed as zeros; every other byte still counts, and an ELF or a damaged
image is hashed as it is.
"""

from __future__ import annotations

import hashlib
import struct
import time

from cash.pe_digest import extension_bytes_digest, pe_volatile_ranges

_PE_AT = 0x80
_FILE_ALIGN = 0x200
_TEXT_RVA = 0x1000
_RDATA_RVA = 0x2000


def _pe32_plus(
    *,
    stamp: int = 0x5F000000,
    checksum: int = 0x1234,
    debug_stamp: int = 0x5F000000,
    guid: bytes = b"\x11" * 16,
    age: int = 1,
    code: bytes = b"\x48\x31\xc0\xc3",
    text_size: int = _FILE_ALIGN,
) -> bytes:
    """A minimal PE32+ DLL: ``.text`` with *code*, and ``.rdata`` holding a
    debug directory with one CodeView entry and its RSDS record."""
    optional_size = 112 + 16 * 8
    headers = _PE_AT + 4 + 20 + optional_size + 2 * 40
    text_at = _FILE_ALIGN
    rdata_at = text_at + text_size
    image = bytearray(rdata_at + _FILE_ALIGN)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, _PE_AT)
    image[_PE_AT : _PE_AT + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", image, _PE_AT + 4, 0x8664, 2, stamp, 0, 0, optional_size, 0x2022)
    opt = _PE_AT + 24
    struct.pack_into("<H", image, opt, 0x20B)
    struct.pack_into("<I", image, opt + 64, checksum)
    struct.pack_into("<I", image, opt + 60, headers)
    struct.pack_into("<I", image, opt + 108, 16)
    debug_rva = _RDATA_RVA
    struct.pack_into("<II", image, opt + 112 + 6 * 8, debug_rva, 28)
    table = opt + optional_size
    struct.pack_into("<8sIIII", image, table, b".text", text_size, _TEXT_RVA, text_size, text_at)
    struct.pack_into("<8sIIII", image, table + 40, b".rdata", 0x100, _RDATA_RVA, _FILE_ALIGN, rdata_at)
    image[text_at : text_at + len(code)] = code
    rsds_at = rdata_at + 28
    record = b"RSDS" + guid + struct.pack("<I", age) + b"C:\\build\\fastops.pdb\0"
    struct.pack_into("<IIHHIIII", image, rdata_at, 0, debug_stamp, 0, 0, 2, len(record), _RDATA_RVA + 28, rsds_at)
    image[rsds_at : rsds_at + len(record)] = record
    return bytes(image)


def test_two_links_of_the_same_code_share_a_digest():
    first = _pe32_plus()
    second = _pe32_plus(stamp=0x6A000000, checksum=0xBEEF, debug_stamp=0x6A000000, guid=b"\x22" * 16, age=3)
    assert first != second
    # COFF stamp, CheckSum, the debug entry's stamp, and the RSDS GUID with its age.
    assert pe_volatile_ranges(first) == [
        (_PE_AT + 8, 4),
        (_PE_AT + 24 + 64, 4),
        (2 * _FILE_ALIGN + 4, 4),
        (2 * _FILE_ALIGN + 32, 20),
    ]
    assert extension_bytes_digest(first) == extension_bytes_digest(second)


def test_one_changed_code_byte_moves_the_digest():
    """A .pxd change, a compiler flag or a new Cython emits different section bytes."""
    assert extension_bytes_digest(_pe32_plus()) != extension_bytes_digest(_pe32_plus(code=b"\x48\x31\xc0\x90\xc3"))
    image = bytearray(_pe32_plus())
    image[_FILE_ALIGN + 0x100] ^= 1
    assert extension_bytes_digest(bytes(image)) != extension_bytes_digest(_pe32_plus())


def test_the_debug_path_still_counts():
    image = _pe32_plus()
    other = image.replace(b"fastops.pdb", b"slowops.pdb")
    assert extension_bytes_digest(image) != extension_bytes_digest(other)


def test_a_damaged_image_is_hashed_as_it_is():
    image = _pe32_plus()
    for broken in (
        image[:0x50],
        image[:0x100],
        image[: _FILE_ALIGN + 10],
        b"MZ" + b"\xff" * 200,
        b"MZ",
        image[:_PE_AT] + b"PX\0\0" + image[_PE_AT + 4 :],
    ):
        assert pe_volatile_ranges(broken) is None
        assert extension_bytes_digest(broken) == hashlib.sha256(broken).hexdigest()


def test_a_debug_directory_outside_every_section_falls_back():
    image = bytearray(_pe32_plus())
    struct.pack_into("<I", image, _PE_AT + 24 + 112 + 6 * 8, 0x9000)
    assert pe_volatile_ranges(bytes(image)) is None
    assert extension_bytes_digest(bytes(image)) == hashlib.sha256(bytes(image)).hexdigest()


def test_elf_and_mach_o_keep_the_plain_hash():
    elf = b"\x7fELF\x02\x01\x01" + bytes(57) + b"code" * 100
    mach_o = b"\xcf\xfa\xed\xfe" + bytes(60) + b"code" * 100
    for built in (elf, mach_o):
        assert extension_bytes_digest(built) == hashlib.sha256(built).hexdigest()


def test_a_big_image_costs_about_a_plain_hash():
    image = _pe32_plus(text_size=5 * 1024 * 1024)

    def best(fn):
        times = []
        for _ in range(5):
            start = time.perf_counter()
            fn(image)
            times.append(time.perf_counter() - start)
        return min(times)

    plain = best(lambda data: hashlib.sha256(data).hexdigest())
    stable = best(extension_bytes_digest)
    assert stable < plain * 1.5 + 0.002
