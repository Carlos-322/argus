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


LOCAL = b'PK\x03\x04'
DESCRIPTOR = b'PK\x07\x08'
CHUNK = 1 << 20


def _zip64_sizes(extra: bytes, size: int, compressed: int) -> tuple[int, int]:
    """A local header's real sizes when it holds the ZIP64 placeholder 0xFFFFFFFF: the ZIP64 extra field lists, in
    order, the uncompressed and the compressed size, each only when its header field is the placeholder."""
    while len(extra) >= 4:
        tag, length = struct.unpack('<HH', extra[:4])
        body = extra[4:4 + length]
        if tag == 1:
            k = 0
            if size == 0xFFFFFFFF and len(body) >= k + 8:
                size = struct.unpack('<Q', body[k:k + 8])[0]
                k += 8
            if compressed == 0xFFFFFFFF and len(body) >= k + 8:
                compressed = struct.unpack('<Q', body[k:k + 8])[0]
            return size, compressed
        extra = extra[4 + length:]
    return size, compressed


def read_local_header(source, offset: int, length: int) -> dict | None:
    """The local header at offset, or None when there is none there. Sizes are None for a member written with a
    data descriptor, which records them after the data instead."""
    if offset < 0 or offset + 30 > length:
        return None
    source.seek(offset)
    header = source.read(30)
    if header[:4] != LOCAL:
        return None
    _, flags, method, _, _, crc, compressed, size, namesize, extrasize = struct.unpack('<HHHHHIIIHH', header[4:])
    raw_name, extra = source.read(namesize), source.read(extrasize)
    if len(raw_name) != namesize or len(extra) != extrasize:
        return None
    if 0xFFFFFFFF in (size, compressed):
        size, compressed = _zip64_sizes(extra, size, compressed)
    entry = {'offset': offset, 'flags': flags, 'method': method, 'raw_name': raw_name,
             'name': raw_name.decode('utf-8' if flags & 0x800 else 'cp437', 'replace'),
             'start': offset + 30 + namesize + extrasize, 'crc': crc, 'compressed': compressed, 'size': size}
    if flags & 8 and not (compressed and size):
        entry.update(crc=None, compressed=None, size=None)
    return entry


def _deflate_end(source, start: int, budget: int):
    """(compressed length, size, crc) of the raw deflate stream at start, found by inflating it: deflate marks its own
    end, so a member written with a data descriptor needs no size in its header. None when it is cut short or
    damaged. Output is counted in bounded steps, never held, and limited to budget bytes."""
    d = zlib.decompressobj(-15)
    source.seek(start)
    fed = size = crc = 0
    while not d.eof:
        chunk = source.read(CHUNK)
        if not chunk:
            return None
        fed += len(chunk)
        data = chunk
        try:
            while data and not d.eof:
                out = d.decompress(data, CHUNK)
                size += len(out)
                crc = zlib.crc32(out, crc)
                if size > budget:
                    raise ValueError('local ZIP members exceed the upload byte limit')
                data = d.unconsumed_tail
        except zlib.error:
            return None
    return fed - len(d.unused_data), size, crc


def _stored_end(source, start: int, length: int):
    """(length, length, crc) of a stored member written with a data descriptor: the first descriptor signature after
    start whose checksum and sizes both match the bytes before it. The running checksum makes this one pass."""
    source.seek(start)
    buf, base, done, crc = b'', start, start, 0
    while True:
        chunk = source.read(CHUNK)
        buf += chunk
        i = buf.find(DESCRIPTOR, done - base)
        while i != -1:
            fields = buf[i + 4:i + 16]
            if len(fields) < 12:
                break                                  # the rest of this descriptor is in the next chunk
            crc = zlib.crc32(buf[done - base:i], crc)
            done = base + i
            stored_crc, compressed, size = struct.unpack('<III', fields)
            if stored_crc == crc and compressed == size == done - start:
                return done - start, done - start, crc
            i = buf.find(DESCRIPTOR, i + 1)
        if not chunk:
            return None
        keep = max(done - base, len(buf) - 15)
        crc = zlib.crc32(buf[done - base:keep], crc)
        done = base + keep
        buf, base = buf[keep:], base + keep


def _content(source, start: int, compressed: int, method: int, budget: int):
    """(size, crc, whole) of the data at start as its compressed length says: inflated (deflate) or as it is (stored),
    read in bounded steps. whole is False when a deflate stream does not end exactly there. None when it is not data."""
    source.seek(start)
    remaining, size, crc = compressed, 0, 0
    d = zlib.decompressobj(-15) if method == 8 else None
    try:
        while remaining:
            chunk = source.read(min(CHUNK, remaining))
            if not chunk:
                return None
            remaining -= len(chunk)
            data = chunk
            while data:
                out = d.decompress(data, CHUNK) if d is not None else data
                size += len(out)
                crc = zlib.crc32(out, crc)
                if size > budget:
                    raise ValueError('local ZIP members exceed the upload byte limit')
                data = d.unconsumed_tail if d is not None else b''
                if d is not None and d.eof:
                    break
    except zlib.error:
        return None
    return size, crc, d is None or (d.eof and not d.unused_data and not remaining)


def _next_header(source, start: int, length: int):
    """The first offset after start holding a local header that parses as an unpackable member, or None."""
    position = start
    while position < length:
        source.seek(position)
        block = source.read(CHUNK + 3)
        i = block.find(LOCAL)
        while i != -1:
            entry = read_local_header(source, position + i, length)
            if entry is not None and entry['method'] in (0, 8) and entry['raw_name'] and not entry['flags'] & 1:
                return position + i
            i = block.find(LOCAL, i + 1)
        if len(block) < CHUNK + 3:
            return None
        position += CHUNK


def _measure(source, entry: dict, length: int, budget: int):
    """(compressed, size, crc) of the member's data as its bytes are, or None. The header's sizes are used only when
    the data they bound inflates to the header's checksum; otherwise the data is measured from itself: a deflate
    stream marks its own end, and a stored member runs to the next header (or its descriptor). A header whose sizes
    lie (forged, or damaged) then costs nothing, and another member's bytes are never returned for it."""
    start, method = entry['start'], entry['method']
    if entry['compressed'] is not None and start + entry['compressed'] <= length:
        got = _content(source, start, entry['compressed'], method, budget)
        if got is not None and got[2] and got[0] == entry['size'] and got[1] == entry['crc']:
            return entry['compressed'], entry['size'], entry['crc']
    if method == 8:
        return _deflate_end(source, start, budget)
    if method == 0:
        if entry['flags'] & 8:
            found = _stored_end(source, start, length)
            if found is not None:
                return found
        end = _next_header(source, start, length)
        end = length if end is None else end
        if end >= start + 16 and _stored_descriptor_at(source, end - 16, start):
            end -= 16
        got = _content(source, start, end - start, 0, budget)
        return None if got is None else (end - start, got[0], got[1])
    return None


def _stored_descriptor_at(source, at: int, start: int) -> bool:
    source.seek(at)
    fields = source.read(16)
    return fields[:4] == DESCRIPTOR and struct.unpack('<II', fields[8:16]) == (at - start, at - start)


def scan_local(path, max_files: int, max_bytes: int) -> tuple[list[dict], list[str]]:
    """Every member the local headers describe, in file order, and what was skipped. The local headers are a ZIP's
    second index, independent of its central directory, so they recover an archive cut short or one whose directory
    is damaged. Each member's sizes are checked against its own checksum (_measure); a member written with a data
    descriptor is measured from its deflate stream or found by its descriptor's checksum; ZIP64 sizes come from the
    header's extra field. A header that cannot be read is skipped to the next one that can, and the bytes between
    are reported, so one damaged header never hides the members after it."""
    entries, damage, total = [], [], 0
    length = path.stat().st_size
    with path.open('rb') as source:
        position = 0
        while position + 30 <= length:
            if len(entries) >= max_files:
                raise ValueError('local ZIP members exceed the upload file limit')
            entry = read_local_header(source, position, length)
            if entry is None:
                source.seek(position)
                if source.read(4) in (b'PK\x01\x02', b'PK\x05\x06', b'PK\x06\x06', b'PK\x06\x07'):
                    break                        # the central directory: no member follows it
                nxt = _next_header(source, position + 1, length)
                if nxt is None:
                    if position:
                        damage.append(f'{length - position:,} bytes after the last readable member could not be read')
                    break
                damage.append(f'{nxt - position:,} damaged bytes at byte {position:,} were skipped to the next member')
                position = nxt
                continue
            measured = _measure(source, entry, length, max_bytes - total) if entry['method'] in (0, 8) else (
                (entry['compressed'], entry['size'], entry['crc']) if entry['compressed'] is not None else None)
            if measured is None:
                damage.append(f"{entry['name']} (its data ends early or cannot be measured)")
                nxt = _next_header(source, entry['start'], length)
                if nxt is None:
                    break
                position = nxt
                continue
            entry['compressed'], entry['size'], measured_crc = measured
            if entry['crc'] is None:
                entry['crc'] = measured_crc
            stop = entry['start'] + entry['compressed']
            tail = 0
            if entry['flags'] & 8:
                source.seek(stop)
                descriptor = source.read(24)
                tail = 4 if descriptor[:4] == DESCRIPTOR else 0
                fields = descriptor[tail:]
                tail += 20 if len(fields) >= 20 and struct.unpack('<IQQ', fields[:20])[1:] == (
                    entry['compressed'], entry['size']) else 12
                if entry['crc'] != measured_crc and len(fields) >= 4:
                    entry['crc'] = struct.unpack('<I', fields[:4])[0]
            total += entry['size']
            if total > max_bytes:
                raise ValueError('local ZIP members exceed the upload byte limit')
            entries.append(entry)
            position = stop + tail
    return entries, damage


def member_opener(path, entry: dict, damage: list):
    """Open one member from its local entry, its bytes kept and a checksum mismatch recorded, as zip_member does."""
    @contextmanager
    def opener():
        import zipfile
        info = zipfile.ZipInfo(entry['name'])
        info.file_size, info.compress_size, info.compress_type = entry['size'], entry['compressed'], entry['method']
        info.CRC = entry['crc']
        with SliceReader(path, entry['start'], entry['compressed']) as stream, \
                zipfile.ZipExtFile(stream, 'r', info) as decoded:
            decoded._expected_crc = None
            reader = ChecksumReader(decoded)
            yield reader
            if reader.complete and reader.crc != entry['crc']:
                damage.append(f"retained: {entry['name']} (checksum mismatch; bytes retained for format validation)")
    return opener


def usable(entry: dict) -> str | None:
    """Why a local entry cannot be unpacked, or None."""
    if entry['flags'] & 1:
        return 'it is encrypted'
    if entry['method'] not in (0, 8):
        return f"it uses ZIP compression method {entry['method']}"
    if entry['method'] == 0 and entry['compressed'] != entry['size']:
        return 'its stored sizes disagree'
    return None


def local_members(path, parts_of, max_files, max_bytes, why):
    """[(parts, size, open)] of every member the local headers recover, the members left out, and the damage notes,
    the first of which says why the local headers were read (why)."""
    entries, stopped = scan_local(path, max_files, max_bytes)
    members, damage, skipped = [], [why] + stopped, 0
    for entry in entries:
        reason = usable(entry)
        if reason:
            damage.append(f"{entry['name']} ({reason}; member was skipped)")
            continue
        parts = parts_of(entry['name'])
        if parts is None or entry['name'].endswith('/'):
            skipped += 1
            continue
        members.append((parts, entry['size'], member_opener(path, entry, damage)))
    return members, skipped, damage


def local_stored_members(path, parts_of, max_files, max_bytes):
    """Members of a ZIP cut short before its directory, recovered from their local headers."""
    return local_members(path, parts_of, max_files, max_bytes,
                         f'{path.name} is cut short; its ZIP directory is unavailable')


def checked_directory(path, infos, max_files, max_bytes):
    """Check every central directory entry against the local header it points at. Returns (the entries that agree,
    {info: the local entry recovered for it} for those that do not, the local entries no directory entry names,
    the names nothing could recover). A directory entry that disagrees with its member (a damaged name, offset, size
    or method) is replaced by the member's own header, found by a scan of the local headers, so a damaged directory
    never costs a member whose bytes are intact."""
    length = path.stat().st_size

    def described(source, entry, info):
        # a member written with a data descriptor states its checksum and sizes once more, right after its data
        source.seek(entry['start'] + info.compress_size)
        tail = source.read(24)
        fields = tail[4:] if tail[:4] == DESCRIPTOR else tail
        return (len(fields) >= 12 and struct.unpack('<III', fields[:12]) == (
                    info.CRC, info.compress_size & 0xFFFFFFFF, info.file_size & 0xFFFFFFFF)) or (
                len(fields) >= 20 and struct.unpack('<IQQ', fields[:20]) == (info.CRC, info.compress_size, info.file_size))

    agree, disagree = [], []
    with path.open('rb') as source:
        for info in infos:
            entry = read_local_header(source, info.header_offset, length)
            name = info.orig_filename.encode('utf-8' if info.flag_bits & 0x800 else 'cp437', 'replace')
            if (entry is not None and entry['raw_name'] == name and entry['method'] == info.compress_type
                    and entry['start'] + info.compress_size <= length
                    and (entry['compressed'] == info.compress_size if entry['compressed'] is not None
                         else described(source, entry, info))):
                agree.append(info)
            else:
                disagree.append(info)
    if not disagree:
        return agree, {}, [], []
    replaced = {}
    with path.open('rb') as source:
        for info in list(disagree):
            entry = _directory_entry(source, info, length)
            if entry is not None:
                replaced[info] = entry
                disagree.remove(info)
    if not disagree:
        return agree, replaced, [], []
    entries, _ = scan_local(path, max_files, max_bytes)
    by_offset = {entry['offset']: entry for entry in entries}
    by_name = {}
    for entry in entries:
        by_name.setdefault(entry['name'], entry)
    claimed = {info.header_offset for info in agree} | {entry['offset'] for entry in replaced.values()}
    lost = []
    for info in disagree:
        entry = by_offset.get(info.header_offset)
        if entry is None or entry['offset'] in claimed:
            entry = by_name.get(info.filename)
        if entry is None or entry['offset'] in claimed:
            lost.append(info.filename)
            continue
        claimed.add(entry['offset'])
        replaced[info] = entry
    extra = [entry for entry in entries if entry['offset'] not in claimed]
    return agree, replaced, extra, lost


def _directory_entry(source, info, length):
    """The member a central directory entry describes, read by the directory's own sizes and checksum, when its local
    header is damaged: its data starts after the header's name and extra fields, whose lengths the damaged header
    may no longer hold, so both its own and the directory's lengths are tried, and only a start whose data inflates
    to the directory's size and checksum is accepted."""
    starts = []
    header = read_local_header(source, info.header_offset, length)
    if header is not None:
        starts.append(header['start'])
    source.seek(info.header_offset + 26)
    lengths = source.read(4)
    if len(lengths) == 4:
        starts.append(info.header_offset + 30 + sum(struct.unpack('<HH', lengths)))
    starts.append(info.header_offset + 30 + len(info.orig_filename.encode('utf-8' if info.flag_bits & 0x800 else 'cp437',
                                                                           'replace')) + len(info.extra))
    for start in dict.fromkeys(starts):
        if start + info.compress_size > length or info.compress_type not in (0, 8):
            continue
        got = _content(source, start, info.compress_size, info.compress_type, info.file_size + 1)
        if got is not None and got[2] and got[0] == info.file_size and got[1] == info.CRC:
            entry = {'offset': info.header_offset, 'flags': info.flag_bits, 'method': info.compress_type,
                     'raw_name': info.orig_filename.encode('utf-8' if info.flag_bits & 0x800 else 'cp437', 'replace'),
                     'name': info.filename, 'start': start, 'crc': info.CRC, 'compressed': info.compress_size,
                     'size': info.file_size}
            if header is not None and start == header['start']:
                # the member's own header located its data, so its name is the one written with that data; the
                # directory's copy is what disagreed
                entry.update(flags=header['flags'], raw_name=header['raw_name'], name=header['name'])
            return entry
    return None
