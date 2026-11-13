"""A content digest of a built extension that a relink of the same code keeps.

MSVC writes the link time into every ``.pyd`` and ``.dll``: the COFF file
header's ``TimeDateStamp``, each debug-directory entry's ``TimeDateStamp``,
and with ``/DEBUG`` a fresh GUID and age in the CodeView (``RSDS``) record
that ties the image to its ``.pdb``. The optional header's ``CheckSum`` sums
the whole file, those fields included. Rebuilding an unchanged Cython
module therefore gave a new file digest and missed every key that reached
it. `extension_bytes_digest` hashes a PE image with exactly those fields
read as zeros; every code and data byte is still hashed, so a different
compiler, flag, ``.pxd`` or Cython version still moves it. Anything that is
not a well-formed PE image (an ELF ``.so``, a Mach-O, a truncated file) is
hashed as it is.
"""

from __future__ import annotations

import hashlib
import struct

__all__ = ["extension_bytes_digest", "pe_volatile_ranges"]

_COFF_HEADER_SIZE = 20
_SECTION_HEADER_SIZE = 40
_DEBUG_ENTRY_SIZE = 28
_DEBUG_DIRECTORY_INDEX = 6
_DEBUG_TYPE_CODEVIEW = 2
#: Far above any real image's debug entries; a corrupt size is refused, not walked.
_MAX_DEBUG_ENTRIES = 4096


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _rva_to_offset(rva: int, size: int, sections: list[tuple[int, int, int, int]], length: int) -> int | None:
    """The file offset of *size* bytes at *rva*, when one section's raw data holds them all."""
    for virtual_address, virtual_size, raw_size, raw_pointer in sections:
        span = max(virtual_size, raw_size)
        if virtual_address <= rva and rva + size <= virtual_address + span:
            delta = rva - virtual_address
            if delta + size > raw_size:
                return None
            offset = raw_pointer + delta
            return offset if offset + size <= length else None
    return None


def pe_volatile_ranges(data: bytes) -> list[tuple[int, int]] | None:
    """The ``(offset, length)`` spans of *data* that only a relink changes, or
    None when *data* is not a PE image this can parse completely."""
    try:
        return _volatile_ranges(data)
    except (struct.error, ValueError, IndexError, OverflowError):
        return None


def _volatile_ranges(data: bytes) -> list[tuple[int, int]] | None:
    length = len(data)
    if length < 0x40 or data[:2] != b"MZ":
        return None
    pe = _u32(data, 0x3C)
    coff = pe + 4
    if coff + _COFF_HEADER_SIZE > length or data[pe:coff] != b"PE\0\0":
        return None
    n_sections = _u16(data, coff + 2)
    optional_size = _u16(data, coff + 16)
    optional = coff + _COFF_HEADER_SIZE
    if optional + optional_size > length:
        return None
    magic = _u16(data, optional)
    if magic == 0x10B:
        count_at, directories_at = 92, 96
    elif magic == 0x20B:
        count_at, directories_at = 108, 112
    else:
        return None
    if directories_at > optional_size:
        return None
    ranges = [(coff + 4, 4), (optional + 64, 4)]
    n_directories = min(_u32(data, optional + count_at), (optional_size - directories_at) // 8)
    if n_directories <= _DEBUG_DIRECTORY_INDEX:
        return ranges
    debug_rva, debug_size = struct.unpack_from("<II", data, optional + directories_at + 8 * _DEBUG_DIRECTORY_INDEX)
    if debug_rva == 0 or debug_size == 0:
        return ranges
    table = optional + optional_size
    if table + n_sections * _SECTION_HEADER_SIZE > length:
        return None
    sections = [struct.unpack_from("<IIII", data, table + i * _SECTION_HEADER_SIZE + 8) for i in range(n_sections)]
    sections = [(va, vsize, raw_size, raw_ptr) for vsize, va, raw_size, raw_ptr in sections]
    n_entries = debug_size // _DEBUG_ENTRY_SIZE
    if debug_size % _DEBUG_ENTRY_SIZE or n_entries > _MAX_DEBUG_ENTRIES:
        return None
    directory = _rva_to_offset(debug_rva, debug_size, sections, length)
    if directory is None:
        return None
    for i in range(n_entries):
        entry = directory + i * _DEBUG_ENTRY_SIZE
        ranges.append((entry + 4, 4))
        entry_type, data_size, _rva, pointer = struct.unpack_from("<IIII", data, entry + 12)
        if entry_type != _DEBUG_TYPE_CODEVIEW or data_size < 24 or pointer == 0:
            continue
        if pointer + 24 > length:
            return None
        if data[pointer : pointer + 4] == b"RSDS":
            ranges.append((pointer + 4, 20))  # the GUID and the age
    return ranges


def extension_bytes_digest(data: bytes) -> str:
    """The sha256 of a built extension's bytes, with a PE image's link stamps
    (`pe_volatile_ranges`) hashed as zeros; any other file's plain sha256."""
    ranges = pe_volatile_ranges(data) if data[:2] == b"MZ" else None
    if not ranges:
        return hashlib.sha256(data).hexdigest()
    view = memoryview(data)
    hasher = hashlib.sha256(b"cash-pe-v1\0")
    at = 0
    for start, size in sorted(ranges):
        start = max(start, at)
        end = max(start, min(start + size, len(data)))
        hasher.update(view[at:start])
        hasher.update(bytes(end - start))
        at = end
    hasher.update(view[at:])
    return hasher.hexdigest()
