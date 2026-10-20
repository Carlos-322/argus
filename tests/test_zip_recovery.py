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
    members, skipped, handle, bad, unverified = formats._archive_members(path)
    got = {}
    with handle:
        for parts, size, opener in members:
            with opener() as reader:
                got['/'.join(parts)] = reader.read()
        for label, filename, size, opener in unverified:
            with opener() as reader:
                got[f'apart:{label}/{filename}'] = reader.read()
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
    if what == 'name':
        unverified_name_kept_apart(got, bad)
    else:
        assert got == PAYLOADS
    assert any('own headers' in note for note in bad)


def unverified_name_kept_apart(got, bad):
    # no checksum covers a name, so a member the two indexes name differently keeps its bytes but is returned apart
    # from the archive's tree, never under either name
    assert sorted(got.values()) == sorted(PAYLOADS.values())
    apart = [name for name in got if name not in PAYLOADS]
    assert len(apart) == 1 and apart[0].startswith('apart:'), list(got)
    assert any('neither name can be verified' in note for note in bad)


def test_intact_archive_reads_through_its_directory_without_notes(tmp_path):
    got, bad = unpacked(tmp_path, build(zipfile.ZIP_DEFLATED, streaming=True))
    assert got == PAYLOADS and bad == []


def local_header(raw, index):
    at = -1
    for _ in range(index + 1):
        at = raw.find(b'PK\x03\x04', at + 1)
    return at


@pytest.mark.parametrize('method', [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
@pytest.mark.parametrize('streaming', [False, True])
def test_a_damaged_extra_field_length_is_found_by_where_the_data_ends(tmp_path, method, streaming):
    # the local header's extra-field length decides where the member's data starts, and no checksum covers it
    raw = bytearray(build(method, streaming=streaming))
    struct.pack_into('<H', raw, local_header(raw, 1) + 28, 7)
    got, bad = unpacked(tmp_path, bytes(raw))
    assert got == PAYLOADS


def test_a_wrong_size_in_the_directory_never_refuses_the_archive(tmp_path):
    raw = bytearray(build(zipfile.ZIP_DEFLATED))
    entry = raw.find(b'PK\x01\x02')
    struct.pack_into('<I', raw, entry + 20, 0x08000000)      # compressed size of 134 MB, past every limit
    got, bad = unpacked(tmp_path, bytes(raw))
    assert got == PAYLOADS


def test_a_member_the_damaged_directory_lost_is_found_by_its_own_header(tmp_path):
    raw = bytearray(build(zipfile.ZIP_DEFLATED))
    first = raw.find(b'PK\x01\x02')
    struct.pack_into('<I', raw, first + 42, 7)                 # one entry points at the wrong offset
    end = raw.rfind(b'PK\x05\x06')
    count = struct.unpack_from('<H', raw, end + 10)[0]
    last = raw.rfind(b'PK\x01\x02', 0, end)
    del raw[last:end]                                          # and the directory lost the last entry
    end = raw.rfind(b'PK\x05\x06')
    struct.pack_into('<HH', raw, end + 8, count - 1, count - 1)
    struct.pack_into('<I', raw, end + 12, end - first)
    got, bad = unpacked(tmp_path, bytes(raw))
    assert got == PAYLOADS


@pytest.mark.parametrize('flip', [0, 4])
def test_a_name_damaged_in_the_member_header_is_never_given_to_another_recording(tmp_path, flip):
    # 'rec/file2.bin' becomes 'sec/file2.bin' (another folder) or 'rec/gile2.bin' in the member header: either name
    # could be the damaged one, and choosing by the siblings could hand the data to another recording
    raw = bytearray(build(zipfile.ZIP_DEFLATED))
    raw[local_header(raw, 2) + 30 + flip] ^= 0x01
    got, bad = unpacked(tmp_path, bytes(raw))
    unverified_name_kept_apart(got, bad)


@pytest.mark.parametrize('what', ['name', 'offset', 'size', 'signature'])
def test_a_note_about_the_archive_is_never_counted_as_a_member_left_out(tmp_path, what):
    # a damaged directory whose members are all recovered leaves nothing out, whatever the notes say about it
    path = tmp_path / 'up.zip'
    path.write_bytes(corrupt_directory(build(zipfile.ZIP_DEFLATED), what))
    _, notes, _ = formats.open_archives(path, tmp_path / 'unpacked')
    opened = next(note for note in notes if note.startswith('Opened up.zip'))
    assert 'left out' not in opened, opened
    assert any(note.startswith('Archive damage: ') and 'ZIP directory' in note for note in notes)


def movie(level):
    import av
    import numpy as np
    out = io.BytesIO()
    with av.open(out, 'w', format='mp4') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for _ in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), level, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


RECORDINGS = {'episode1/front.mp4': movie(40), 'episode2/wrist.mp4': movie(120), 'episode3/top.mp4': movie(200)}
TASKS = {'episode1/task.txt': 'task of episode one', 'episode2/task.txt': 'task of episode two',
         'episode3/task.txt': 'task of episode three'}


def name_damage(raw, how):
    raw = bytearray(raw)
    local = raw.find(b'PK\x03\x04')                     # the first member, episode1/front.mp4
    central = raw.find(b'PK\x01\x02')
    if how in ('local_name', 'local_name_and_extra'):
        raw[local + 30 + len('episode')] = ord('2')     # its own header says episode2/front.mp4
    if how in ('directory_name', 'directory_name_and_extra'):
        raw[central + 46 + len('episode')] = ord('2')   # its directory entry says episode2/front.mp4
    if how.endswith('_and_extra'):
        struct.pack_into('<H', raw, local + 28, 7)      # and its header's extra-field length is damaged too
    return bytes(raw)


@pytest.mark.parametrize('how', ['local_name', 'directory_name', 'local_name_and_extra', 'directory_name_and_extra'])
@pytest.mark.parametrize('planted', [None, 'name_unverified/byte_0/task.txt', 'up-byte_0/task.txt',
                                     '000-up-byte_0/task.txt'])
def test_a_member_of_unproven_name_is_its_own_upload_and_no_file_describes_it(tmp_path, how, planted):
    import json
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, 'w', zipfile.ZIP_STORED) as z:
        for name, data in {**RECORDINGS, **TASKS}.items():
            z.writestr(name, data)
        if planted:
            # a file of the upload at a path a recovery folder could take, describing nothing of the damaged member
            z.writestr(planted, 'a task no recording of this archive owns')
    path = tmp_path / 'up.zip'
    path.write_bytes(name_damage(sink.getvalue(), how))
    out = tmp_path / 'out'
    report = formats.convert(path, 'teleop_arms', out, 'ownership', 900)
    contexts = {e['episode_id']: json.loads((out / e['episode_id'] / 'context.json').read_text())
                for e in report['episodes']}
    tasks = {eid: str(ctx.get('instruction') or '') for eid, ctx in contexts.items()}
    # every recording is read: the two intact ones with their own tasks, the damaged one apart with none
    assert len(contexts) == 3, report
    apart = [eid for eid in contexts if 'name_unverified' in eid]
    assert len(apart) == 1 and tasks[apart[0]] == '', tasks
    assert sorted(t for eid, t in tasks.items() if eid not in apart) == ['task of episode three', 'task of episode two']
    assert not any('no recording of this archive owns' in t for t in tasks.values())


@pytest.mark.parametrize('what', ['name', 'offset', 'signature'])
def test_a_member_named_like_a_note_is_still_a_member(tmp_path, what):
    # what kind a note is travels with it as data, so a member whose name reads like a note is counted as a member
    global PAYLOADS
    saved = PAYLOADS
    try:
        PAYLOADS = {'archive: damaged/file0.bin': b'x' * 5000, 'retained: file1.bin': b'y' * 6000}
        raw = corrupt_directory(build(zipfile.ZIP_STORED), what)
    finally:
        PAYLOADS = saved
    path = tmp_path / 'up.zip'
    path.write_bytes(raw)
    members, skipped, handle, bad, unverified = formats._archive_members(path)
    with handle:
        names = {'/'.join(parts) for parts, _, _ in members} | {f'apart:{label}' for label, *_ in unverified}
    assert len(names) == 2, names
    assert all(getattr(note, 'kind', None) in ('archive', 'retained', 'cut', None) for note in bad)


def nested(levels, payload_name, payload):
    data, name = payload, payload_name
    for level in range(levels):
        sink = io.BytesIO()
        with zipfile.ZipFile(sink, 'w', zipfile.ZIP_STORED) as z:
            z.writestr(name, data)
        data, name = sink.getvalue(), f'level{level}.zip'
    return data


@pytest.mark.parametrize('levels', [1, 3, 5, 8])
def test_a_recording_nested_at_any_depth_is_read(tmp_path, levels):
    path = tmp_path / 'up.zip'
    path.write_bytes(nested(levels, 'take/front.mp4', movie(90)))
    report = formats.convert(path, 'ego_head', tmp_path / 'out', 'nested', 900)
    assert len(report['episodes']) == 1 and not report['failed'], report


def test_an_archive_nested_without_end_stops_at_the_budget(tmp_path, monkeypatch):
    # what bounds an archive built to unpack forever is the file and byte budget every level spends, not a depth
    monkeypatch.setattr(formats, 'UNPACK_MAX_FILES', 12)
    path = tmp_path / 'up.zip'
    path.write_bytes(nested(40, 'note.txt', b'x'))
    with pytest.raises(formats.UnpackLimit):
        formats.open_archives(path, tmp_path / 'unpacked')


@pytest.mark.parametrize('beside', ['a different file', 'a folder', 'an earlier run of the archive'])
def test_a_member_whose_path_is_taken_by_another_file_is_never_lost(tmp_path, beside):
    # a zip copy of take1/cam.mp4 beside a folder copy that differs had been skipped with no word. Only a file of the
    # member's size is this member, unpacked by an earlier run; any other keeps its path, and the member is read as
    # an upload of its own, which no file of the upload describes
    up = tmp_path / 'upload'
    (up / 'take1').mkdir(parents=True)
    member = b'ZIP COPY, DIFFERENT BYTES'
    if beside == 'a different file':
        (up / 'take1' / 'cam.mp4').write_bytes(b'FOLDER COPY')
    elif beside == 'a folder':
        (up / 'take1' / 'cam.mp4').mkdir()
    else:
        (up / 'take1' / 'cam.mp4').write_bytes(member)
    with zipfile.ZipFile(up / 'take1.zip', 'w') as z:
        z.writestr('take1/cam.mp4', member)
        z.writestr('take1/notes.txt', b'hello')
    root, notes, apart = formats.open_archives(up, tmp_path / 'dest')
    assert (up / 'take1' / 'notes.txt').read_bytes() == b'hello'
    if beside == 'an earlier run of the archive':
        assert apart == [] and (up / 'take1' / 'cam.mp4').read_bytes() == member, notes
        return
    assert (up / 'take1' / 'cam.mp4').read_bytes() == b'FOLDER COPY' if beside == 'a different file' \
        else (up / 'take1' / 'cam.mp4').is_dir()
    assert len(apart) == 1 and [p.name for p in apart[0].iterdir()] == ['cam.mp4'], apart
    assert (apart[0] / 'cam.mp4').read_bytes() == member
    said = [n for n in notes if 'take1/cam.mp4' in str(n)]
    assert said == [f'take1.zip holds take1/cam.mp4, which differs from the take1/cam.mp4 already in the upload, so '
                    f"the archive's copy was read as an upload of its own, {apart[0].name}, which no other file "
                    'describes.'], notes
    # a rerun finds the member apart from the archive's done marker, never unpacked a second time
    assert formats.open_archives(up, tmp_path / 'dest')[2] == apart


def write_tar(path, names, *, folders=False):
    import tarfile
    with tarfile.open(path, 'w:gz') as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            data = b'' if folders else name.encode()
            if folders:
                info.type = tarfile.DIRTYPE
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def headers_read(monkeypatch):
    """How many tar headers the reader reads, by archive, as tarfile reads them (opening a tar reads its first header,
    which its first next() then hands back, so a header is counted once by where it is)."""
    import tarfile
    read, seen, original = {}, set(), tarfile.TarFile.next

    def counted(self):
        member = original(self)
        if member is not None and (self.name, member.offset) not in seen:
            seen.add((self.name, member.offset))
            read[os.path.basename(self.name)] = read.get(os.path.basename(self.name), 0) + 1
        return member
    monkeypatch.setattr(tarfile.TarFile, 'next', counted)
    return read


@pytest.mark.parametrize('folders', [False, True])
def test_a_tar_stops_reading_headers_when_it_passes_the_file_budget(tmp_path, monkeypatch, folders):
    # every tar header is held as it is read, so the budget is spent while reading, never after: a .tgz of 1.2 million
    # empty members had held every header (1.26 GB) before it was refused
    monkeypatch.setattr(formats, 'UNPACK_MAX_FILES', 50)
    path = tmp_path / 'up.tgz'
    write_tar(path, [f'd/{i}' if folders else f'd/{i}.txt' for i in range(5000)], folders=folders)
    read = headers_read(monkeypatch)
    with pytest.raises(formats.UnpackLimit):
        formats.open_archives(path, tmp_path / 'unpacked')
    assert read == {'up.tgz': 51}, read


@pytest.mark.parametrize('budget', [60, 50])
def test_a_tar_is_read_against_what_earlier_archives_left_of_the_budget(tmp_path, monkeypatch, budget):
    up = tmp_path / 'up'
    up.mkdir()
    for name in ('a', 'b'):
        write_tar(up / f'{name}.tgz', [f'{name}/{i}.txt' for i in range(30)])
    monkeypatch.setattr(formats, 'UNPACK_MAX_FILES', budget)
    read = headers_read(monkeypatch)
    if budget == 60:
        root, notes, apart = formats.open_archives(up, tmp_path / 'unpacked')
        assert [len(list((up / name).iterdir())) for name in ('a', 'b')] == [30, 30], notes
        return
    with pytest.raises(formats.UnpackLimit):
        formats.open_archives(up, tmp_path / 'unpacked')
    assert read == {'a.tgz': 30, 'b.tgz': 21}, read
