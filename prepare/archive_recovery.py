"""Bounded recovery of ZIP member bytes, with checksum damage kept explicit."""
from contextlib import contextmanager
import struct
import zlib


class ChecksumReader:
    def __init__(self, source):
        self.source, self.crc, self.complete = source, 0, False

    def read(self, size=-1):
        data = self.source.read(size)
        self.crc = zlib.crc32(data, self.crc)
        if not data:
            self.complete = True
        return data


@contextmanager
def zip_member(archive, info, damage):
    with archive.open(info) as source:
        # ZipExtFile checks CRC before returning its final buffer. Recover bytes first and report CRC separately.
        source._expected_crc = None
        reader = ChecksumReader(source)
        yield reader
        if reader.complete and reader.crc != info.CRC:
            damage.append(f'retained: {info.filename} (checksum mismatch; bytes retained for format validation)')


class SliceReader:
    def __init__(self, path, start, length):
        self.source = path.open('rb')
        self.source.seek(start)
        self.remaining = length

    def read(self, size=-1):
        data = self.source.read(self.remaining if size < 0 else min(size, self.remaining))
        self.remaining -= len(data)
        return data

    def seekable(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.source.close()


def local_stored_members(path, parts_of, max_files, max_bytes):
    """Recover complete members with explicit local sizes. Never scan inside payloads for headers."""
    members, damage, skipped = [], [f'{path.name} is cut short; its ZIP directory is unavailable'], 0
    length, position, total = path.stat().st_size, 0, 0
    with path.open('rb') as source:
        while position + 30 <= length and len(members) < max_files:
            source.seek(position)
            header = source.read(30)
            if header[:4] != b'PK\x03\x04':
                break
            _, flags, method, _, _, crc, compressed, size, namesize, extrasize = struct.unpack('<HHHHHIIIHH', header[4:])
            raw_name = source.read(namesize)
            if len(raw_name) != namesize:
                break
            name = raw_name.decode('utf-8' if flags & 0x800 else 'cp437', 'replace')
            start = position + 30 + namesize + extrasize
            stop = start + compressed
            if flags & 8:
                damage.append(f'{name} (local ZIP recovery requires explicit sizes; streaming descriptors are unsupported)')
                break
            if stop > length:
                damage.append(f'{name} (member is cut short)')
                break
            if flags & 1 or method not in (0, 8) or method == 0 and compressed != size:
                damage.append(f'{name} (encrypted or unsupported local ZIP compression {method}; member was skipped)')
                position = stop
                continue
            if total + size > max_bytes:
                raise ValueError('local ZIP members exceed the upload byte limit')
            parts = parts_of(name)
            if parts is not None and not name.endswith('/'):
                @contextmanager
                def opener(start=start, size=size, compressed=compressed, method=method, crc=crc, name=name):
                    import zipfile
                    info = zipfile.ZipInfo(name)
                    info.file_size, info.compress_size, info.compress_type, info.CRC = size, compressed, method, crc
                    with SliceReader(path, start, compressed) as stream, zipfile.ZipExtFile(stream, 'r', info) as decoded:
                        decoded._expected_crc = None
                        reader = ChecksumReader(decoded)
                        yield reader
                        if reader.complete and reader.crc != crc:
                            damage.append(f'retained: {name} (checksum mismatch; bytes retained for format validation)')
                members.append((parts, size, opener))
                total += size
            else:
                skipped += 1
            position = stop
    if len(members) == max_files and position < length:
        raise ValueError('local ZIP members exceed the upload file limit')
    return members, skipped, damage
