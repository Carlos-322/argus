"""A ZIP has two indexes, its central directory and each member's local header: damage to one never costs a member
whose bytes and other index are intact."""
import io
import os
import struct
import zipfile

import pytest

from prepare import formats

PAYLOADS = {f'rec/file{i}.bin': os.urandom(3000 + 1000 * i) + b'x' * 20000 for i in range(4)}


class Unseekable(io.RawIOBase):
    """A write-only stream: zipfile then writes every member with a data descriptor, as streaming writers do."""

    def __init__(self):
        self.buf = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.buf += b
        return len(b)


def build(method, *, streaming=False, zip64=False):
    sink = Unseekable() if streaming else io.BytesIO()
    with zipfile.ZipFile(sink, 'w', method) as z:
        for name, data in PAYLOADS.items():
            with z.open(name, 'w', force_zip64=zip64) as w:
                w.write(data)
    return bytes(sink.buf) if streaming else sink.getvalue()


def unpacked(tmp_path, raw):
    path = tmp_path / 'up.zip'
    path.write_bytes(raw)
    members, skipped, handle, bad = formats._archive_members(path)
    got = {}
    with handle:
        for parts, size, opener in members:
            with opener() as reader:
                got['/'.join(parts)] = reader.read()
    return got, bad


@pytest.mark.parametrize('method,streaming,zip64', [
    (zipfile.ZIP_DEFLATED, True, False), (zipfile.ZIP_STORED, True, False), (zipfile.ZIP_DEFLATED, False, True),
    (zipfile.ZIP_DEFLATED, False, False)])
def test_archive_cut_short_keeps_every_member_before_the_cut(tmp_path, method, streaming, zip64):
    raw = build(method, streaming=streaming, zip64=zip64)
    directory = raw.rfind(b'PK\x01\x02', 0, raw.rfind(b'PK\x05\x06'))
    first_directory = raw.find(b'PK\x01\x02')
    cut = raw[:first_directory - 10]           # inside the last member's data or descriptor
    got, bad = unpacked(tmp_path, cut)
    names = list(PAYLOADS)
    assert {n: got[n] for n in names[:3]} == {n: PAYLOADS[n] for n in names[:3]}
    assert any('cut short' in note or 'ends early' in note for note in bad)
    assert directory > 0


def corrupt_directory(raw, what):
    raw = bytearray(raw)
    entry = raw.find(b'PK\x01\x02')
    if what == 'name':
        raw[entry + 46] ^= 0x20                 # first byte of the file name
    elif what == 'offset':
        struct.pack_into('<I', raw, entry + 42, 7)
    elif what == 'size':
        struct.pack_into('<I', raw, entry + 20, 5)
    elif what == 'signature':
        raw[entry:entry + 4] = b'XXXX'
    return bytes(raw)


@pytest.mark.parametrize('what', ['name', 'offset', 'size', 'signature'])
@pytest.mark.parametrize('streaming', [False, True])
def test_damaged_directory_entry_is_replaced_by_the_members_own_header(tmp_path, what, streaming):
    raw = corrupt_directory(build(zipfile.ZIP_DEFLATED, streaming=streaming), what)
    got, bad = unpacked(tmp_path, raw)
    assert got == PAYLOADS
    assert any('own headers' in note for note in bad)


def test_intact_archive_reads_through_its_directory_without_notes(tmp_path):
    got, bad = unpacked(tmp_path, build(zipfile.ZIP_DEFLATED, streaming=True))
    assert got == PAYLOADS and bad == []
