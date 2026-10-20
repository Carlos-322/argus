"""What a reader leaves out of an episode is named on it.

Each test feeds a failure in where every instance of a part passes through (every data file, every text array, every
message, every signal) and checks the part left out is named on the episode or in the upload's report, never only
skipped. These were try/except sites that skipped the part with no record."""
import json

import numpy as np
import pandas as pd

from prepare import formats


def _converted(root, out):
    rep = formats.convert(root, "teleop_arms", out, "test", 900)
    assert not rep["failed"], rep
    return rep, [json.loads((out / e["episode_id"] / "context.json").read_text()) for e in rep["episodes"]]


def _kinds(ctx):
    return [i["kind"] for i in ctx.get("reader_issues") or []]


def test_a_lerobot_v3_data_file_that_does_not_open_is_named_in_the_report(tmp_path, monkeypatch):
    from test_formats import _lerobot_v3
    root = tmp_path / "upload"
    _lerobot_v3(root, {"observation.images.top": 30}, {
        "episode_index": np.repeat([0, 1], 15), "frame_index": np.tile(np.arange(15), 2),
        "timestamp": np.tile(np.arange(15) / 30, 2), "observation.state": list(np.zeros((30, 3)))})
    import pyarrow.parquet as pq
    real = pq.ParquetFile

    def damaged(source, *a, **k):
        if "/data/" in str(source):
            raise OSError("the file is cut short")
        return real(source, *a, **k)
    monkeypatch.setattr(pq, "ParquetFile", damaged)
    rep = formats.convert(root, "teleop_arms", tmp_path / "out", "test", 900)
    named = [m for m in rep["missing"] if "data/chunk-000/file-000.parquet" in m]
    assert named and "its episodes could not be found" in named[0], rep["missing"]


def test_a_lerobot_v3_camera_or_data_file_whose_place_is_missing_is_named(tmp_path):
    from test_formats import _lerobot_v3
    top, wrist = "observation.images.top", "observation.images.wrist_left"
    root = tmp_path / "upload"
    _lerobot_v3(root, {top: 20, wrist: 20}, {
        "episode_index": np.repeat([0, 1], 10), "frame_index": np.tile(np.arange(10), 2),
        "timestamp": np.tile(np.arange(10) / 30, 2), "observation.state": list(np.zeros((20, 3)))})
    p = root / "meta/episodes/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True)
    # every episode row lacks where the wrist camera's video is and where its data file is
    pd.DataFrame([{"episode_index": e, "length": 10, "tasks": ["pick"],
                   f"videos/{top}/chunk_index": 0, f"videos/{top}/file_index": 0,
                   f"videos/{top}/from_timestamp": e / 3, f"videos/{top}/to_timestamp": (e + 1) / 3}
                  for e in (0, 1)]).to_parquet(p)
    rep, ctxs = _converted(root, tmp_path / "out")
    assert len(ctxs) == 2
    for ctx in ctxs:
        aligned = [i for i in ctx["reader_issues"] if i["kind"] == "camera_not_aligned"]
        assert [i["camera"] for i in aligned] == [wrist], ctx["reader_issues"]
        assert "its place in meta/episodes could not be read" in aligned[0]["what"]
        assert any(u.startswith(wrist) and "meta/episodes" in u for u in ctx["source"]["unused_cameras"])
    named = [m for m in rep["missing"] if "could not be located" in m]
    assert named and "episode 0" in named[0] and "episode 1" in named[0], rep["missing"]


def test_an_hdf5_text_array_that_does_not_read_is_named():
    class Unreadable:
        def __getitem__(self, key):
            raise OSError("damaged chunk")
    texts = [{"path": "task", "name": "task"}, {"path": "notes/operator", "name": "notes/operator"}]
    issues = []
    instruction, notes = formats.h5_text({"task": Unreadable(), "notes/operator": Unreadable()}, "", texts,
                                         issues=issues)
    assert instruction is None and not set(notes) & {"task", "notes/operator"}
    named = [i for i in issues if i["kind"] == "metadata_unreadable"]
    assert [i["signal"] for i in named] == ["task", "notes/operator"], issues
    assert all("could not be read (OSError)" in i["what"] for i in named)


def _json_channel_mcap(path, n=5):
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        w.add_metadata("recording", {"task_name": "pick"})
        sid = w.register_schema(name="Reading", encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic="/force", message_encoding="json", schema_id=sid)
        for k in range(n):
            w.add_message(ch, log_time=1_000_000_000 + k * 10**8, publish_time=1_000_000_000 + k * 10**8,
                          data=json.dumps({"value": float(k)}).encode())
        w.finish()


def test_mcap_messages_that_do_not_decode_are_named_in_the_retained_fields(tmp_path, monkeypatch):
    path = tmp_path / "rec.mcap"
    _json_channel_mcap(path)

    def broken(data):
        raise ValueError("bad payload")
    monkeypatch.setattr(formats, "_decoder_for", lambda *a, **k: broken)
    ctx = {}
    formats.retain_mcap_fields(path, tmp_path, ctx)
    entry = next(e for e in ctx["mcap_field_inventory"] if e["topic"] == "/force")
    assert [e["message_index"] for e in entry["capture_errors"]] == list(range(5))
    assert all("could not be decoded (ValueError)" in e["what"] for e in entry["capture_errors"])


def test_mcap_metadata_records_that_do_not_read_are_named(tmp_path, monkeypatch):
    from mcap.reader import SeekingReader
    path = tmp_path / "rec.mcap"
    _json_channel_mcap(path)

    def damaged(self):
        raise ValueError("metadata index points past the file")
        yield
    monkeypatch.setattr(SeekingReader, "iter_metadata", damaged)
    ctx = {}
    formats.retain_mcap_fields(path, tmp_path, ctx)
    named = [i for i in ctx["reader_issues"] if i["kind"] == "metadata_unreadable"]
    assert len(named) == 1 and "MCAP metadata records could not be fully read" in named[0]["what"], ctx
    assert ctx["recorded_metadata"]["mcap_metadata"] == []


def test_lines_of_a_large_json_lines_table_that_are_not_json_are_named(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, "TABLE_MAX_BYTES", 10)
    root = tmp_path / "upload"
    root.mkdir()
    (root / "notes.jsonl").write_text('{"task": "pick the cup"}\n{broken\n{"task": "place it"}\nnot json either\n')
    reads = {"dir": root, "missing": [], "metadata_issues": [], "metadata_read": set()}
    tables = formats.annotation_tables(root, reads)
    assert [r["task"] for r in dict(tables)["notes.jsonl"]] == ["pick the cup", "place it"]
    assert any("lines 2 and 4 were not JSON" in m for m in reads["missing"]), reads["missing"]


def test_a_native_state_column_that_is_not_one_array_is_named(tmp_path):
    df = pd.DataFrame({"observation.state": [[[1.0, 2.0], [3.0]]] * 4, "action": [[[0.0], [1.0, 2.0]]] * 4})
    ctx = {"state_kind": "none"}
    formats.archive_lerobot_state(tmp_path, ctx, df, tmp_path / "data.parquet")
    named = [i for i in ctx["reader_issues"] if i["kind"] == "part_unreadable"]
    assert ["observation.state" in named[0]["what"], "action" in named[1]["what"]] == [True, True], named
    assert "recorded_state_archive" not in ctx
    assert json.loads((tmp_path / "context.json").read_text())["reader_issues"] == ctx["reader_issues"]


def test_an_episode_whose_contacts_cannot_be_measured_says_so(tmp_path, monkeypatch):
    from test_container_camera_clocks import hdf_upload
    from label import contacts

    def broken(*a, **k):
        raise RuntimeError("contact finder failed")
    monkeypatch.setattr(contacts, "find", broken)
    upload = hdf_upload(tmp_path, np.arange(12) / 10, signal=True)
    _, ctxs = _converted(upload, tmp_path / "out")
    assert len(ctxs) == 1 and "contacts" not in ctxs[0]
    named = [i for i in ctxs[0]["reader_issues"] if i["kind"] == "part_unreadable" and "contacts" in i["what"]]
    assert len(named) == 1 and "RuntimeError" in named[0]["what"], ctxs[0].get("reader_issues")


def test_a_signal_whose_rows_cannot_be_made_is_named_in_the_prompt(tmp_path, monkeypatch):
    from test_container_camera_clocks import hdf_upload
    from label import episode, signals
    upload = hdf_upload(tmp_path, np.arange(12) / 10, signal=True)
    rep, ctxs = _converted(upload, tmp_path / "out")
    ep = tmp_path / "out" / rep["episodes"][0]["episode_id"]
    assert "force" in ctxs[0].get("signals", {}) or (ep / "signals.npz").exists()

    def broken(*a, **k):
        raise RuntimeError("rows failed")
    monkeypatch.setattr(signals, "summary_rows", broken)
    prompt = episode.build_request(ep)["prompt"]
    assert "leave out force (RuntimeError), because its rows could not be read" in prompt, prompt[-3000:]


def test_a_camera_left_without_frame_times_names_each_stamp_file_beside_it(tmp_path):
    """A per-frame stamp file that does not load, holds another number of stamps or runs backwards leaves the camera
    on its decode times, and the episode says which file and why; another stream's stamps are not this camera's."""
    import numpy as np
    from prepare import formats as fm
    video = tmp_path / "exo_cam-images-rgb.mp4"
    video.write_bytes(b"")
    stamps = tmp_path / "exo_cam-rgb-timestamp.npy"
    named = lambda ctx: [i["what"] for i in ctx.get("reader_issues", []) if i["kind"] == "camera_timestamp_invalid"]

    np.save(stamps, np.arange(5.0))
    ctx = {}
    assert fm.frame_times(video, 7, ctx=ctx) is None
    assert named(ctx) == ["exo_cam-rgb-timestamp.npy beside exo_cam-images-rgb.mp4 is not used as its frame times: "
                          "it holds 5 stamps for 7 frames."]
    np.save(stamps, np.array([0.0, 2.0, 1.0]))
    ctx = {}
    assert fm.frame_times(video, 3, ctx=ctx) is None and "not all finite and in order" in named(ctx)[0]
    stamps.write_bytes(b"not a numpy file")
    ctx = {}
    assert fm.frame_times(video, 3, ctx=ctx) is None and "it does not load" in named(ctx)[0]

    stamps.unlink()
    np.save(tmp_path / "exo_cam-depth-timestamp.npy", np.arange(5.0))
    ctx = {}
    assert fm.frame_times(video, 7, ctx=ctx) is None and named(ctx) == []      # depth stamps are not the colour clock
    np.save(stamps, np.arange(7.0) / 30)
    ctx = {}
    assert fm.frame_times(video, 7, ctx=ctx) is not None and named(ctx) == []  # a fitting file is used, nothing named
