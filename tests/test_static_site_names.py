"""A static build names files after identifiers it does not choose: episode ids and files, dataset names, camera keys
and other models' keys. board/static.py site_name gives each its one name, used both as the file's name on disk and,
unchanged, as its URL path segment, so every path the build publishes is served, whatever the identifier holds."""
import argparse
import functools
import http.server
import json
import random
import re
import shutil
import socketserver
import subprocess
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from board import serve, static

# identifiers of every kind that broke a static build or one of its URLs: Unicode (composed and decomposed), spaces,
# slashes, percent signs (a literal one, and a sequence that URL decoding turns into another name), dots (leading,
# trailing, only dots), URL delimiters, very long names, and names differing only by case
HOSTILE = ["episode_café", "episode_café", "Kitchen / day 1", "kitchen / day 1", "ep 50%", "ep%C3%A9", "epé",
           "ep#1?x=y&z", ".hidden ep", "ep.", "..", ".", "", "task/ep 1", "../escape", "E" * 150, "e" * 150,
           "x" * 300, "Upper_A", "upper_a", "日本語 エピソード", "データ", "ep\t\n", "a+b=c", "~tilde", "ep_🙂",
           "Modèle B 2%", "nul", "CON", "ep\\back", "ep:colon", "ep\"quote'", "ep<tag>"]
KEPT = ["episode_000001", "episode_habit_005733", "molmo", "exo", "left", "depth_left", "extra1", "unshown2",
        "other", "a.b-c_d", "episode_x.json"]
NAME = re.compile(r"[a-z0-9_~-][a-z0-9._~-]{0,99}")


def _check_names(corpus):
    names = {s: static.site_name(s) for s in corpus}
    for s, n in names.items():
        # one alphabet that no URL encoding changes, no path splits and no file system folds
        assert NAME.fullmatch(n) and n not in (".", ".."), (s, n)
        assert urllib.parse.quote(n, safe="") == n and urllib.parse.unquote(n) == n, (s, n)
    # distinct even on a case-insensitive, Unicode-normalising file system
    folded = {unicodedata.normalize("NFC", n).casefold() for n in names.values()}
    assert len(folded) == len(set(corpus))
    return names


def test_site_names_are_one_safe_name_per_identifier():
    _check_names(HOSTILE + KEPT)
    # names that were already safe keep their names, so published media are not renamed
    for s in KEPT:
        assert static.site_name(s) == s
    assert static.site_file("episode_000001.json") == "episode_000001.json"
    assert static.site_file("episode_café.json").endswith(".json")
    # a random corpus over the same characters, so the property holds for the class and not only the list
    rng = random.Random(7)
    alphabet = "aAbBzZ09 ._-%/\\#?&=+~:'\"<>\t\néeÉ́日🙂"
    corpus = {"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 140))) for _ in range(5000)}
    corpus |= {s.upper() for s in corpus} | {unicodedata.normalize("NFD", s) for s in corpus}
    _check_names(sorted(corpus))


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as err:
        return err.code


def _json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([serve.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=30:d=0.3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True, timeout=30)


# (qa file, episode id, dataset, camera views): every kind of hostile name in each place a build writes one
EPISODES = [
    ("episode_café.json", "episode_café", "Kitchen / day 1", ["exo", "left", "extra1"]),
    ("episode_nfd.json", "episode_café", "kitchen / day 1", ["exo"]),
    ("ep 50%.json", "ep 50%", "100% done", ["exo"]),
    ("ep%C3%A9.json", "ep%C3%A9", "100% done", ["exo"]),
    ("ep#1?x=y&z.json", "ep#1?x=y&z", "Kitchen / day 1", ["exo"]),
    (".hidden ep.json", ".hidden ep", "..", ["exo"]),
    ("ep..json", "ep.", "data.set", ["exo"]),
    ("slash_id.json", "task/ep 1", "../escape", ["exo"]),
    ("E" * 150 + ".json", "E" * 150, "x" * 300, ["exo"]),
    ("Upper_A.json", "Upper_A", "Kitchen", ["exo"]),
    ("lower_b.json", "upper_a", "kitchen", ["exo"]),
    ("日本語 エピソード.json", "日本語 エピソード", "データ", ["exo"]),
    ("no_clip.json", "no_clip", "Kitchen", ["exo"]),
    ("episode_000001.json", "episode_000001", "molmo", ["exo"]),
]
COMPARE_KEY = "Modèle B 2%"


@pytest.mark.skipif(not static.FFMPEG or not static.FFPROBE, reason="ffmpeg is not installed")
def test_every_published_path_of_a_hostile_build_is_served(tmp_path):
    board, clips = tmp_path / "board", tmp_path / "clips"
    qa = board / "qa"
    qa.mkdir(parents=True)
    for file, eid, ds, views in EPISODES:
        doc = {"episode_prompt": "Place cup", "dataset": ds, "_rig": "ego_head", "camera_views": views,
               "_meta": {"episode_id": eid}, "duration_s": 0.3,
               "completion": {"task_completed": "success", "completed_at_s": 0.1}}
        (qa / file).write_text(json.dumps(doc))
        if file != "no_clip.json":
            for view in views:
                _clip(serve.clip_path(clips, eid, view))
    # another model's labels of two episodes, under a key that needs encoding
    cmp = board / "compare"
    (cmp / COMPARE_KEY).mkdir(parents=True)
    for file in ("episode_café.json", "ep 50%.json"):
        shutil.copy(qa / file, cmp / COMPARE_KEY / file)
    (cmp / "index.json").write_text(json.dumps({"models": [{"key": COMPARE_KEY, "name": "B"}],
                                                "episodes": {"episode_café.json": {COMPARE_KEY: "parsed"},
                                                             "ep 50%.json": {COMPARE_KEY: "parsed"}}}))
    (cmp / "metrics.json").write_text(json.dumps({"models": []}))
    # a sensors file and a hand keypoints download, each under an episode file that needs encoding
    (board / "sensors").mkdir()
    (board / "sensors" / "index.json").write_text(json.dumps({"format": "board-sensors/1",
                                                              "files": {"ep#1?x=y&z.json": {"signals": 0}}}))
    (board / "sensors" / "ep#1?x=y&z.json").write_text(json.dumps({"format": "board-sensors/1", "signals": []}))
    (board / "hand_keypoints").mkdir()
    (board / "hand_keypoints" / "index.json").write_text(json.dumps({"files": {"日本語 エピソード.json": {"frames": 1}}}))
    (board / "hand_keypoints" / "日本語 エピソード.json").write_text(json.dumps({"format": "pantheon-hand-keypoints/1"}))

    out = tmp_path / "static"
    a = argparse.Namespace(board=board, qa=qa, clips=clips, compare=None, hands=None, keypoints=None, sensors=None,
                           out=out, build_id="b1", force=False, public_base=None, title="T", header=None, jobs=2,
                           threads=1, limit_per_dataset=0, dataset=None)
    assert static.cmd_media(a) == 0
    # one web copy not made yet: the build must list exactly it as missing
    gone = next(e for e in static.plan(qa, clips, cmp) if e["eid"] == "ep 50%")["media"]["exo"]["rel"]
    (out / "media" / gone).unlink()
    assert static.cmd_site(a) == 0
    # nothing is written outside the build's own folders, whatever a name holds
    assert {p.relative_to(out).parts[0] for p in out.rglob("*")} <= {"b1", "media", "LATEST"}

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(out)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{httpd.server_address[1]}/"
    data, media = root + "b1/data/", root + "media/"
    q = functools.partial(urllib.parse.quote, safe="")     # the page's encodeURIComponent
    try:
        index = _json(data + "index.json")
        site = lambda f: index["site_names"].get(f, f)     # noqa: E731  (the page's siteFile)
        lists = dict(zip(index["datasets"], index["lists"]))
        assert set(lists) == {ds for _, _, ds, _ in EPISODES}
        records = [r for ds in lists for r in _json(data + "lists/" + q(lists[ds]))]
        assert sorted(r["file"] for r in records) == sorted(f for f, _, _, _ in EPISODES)
        for file, eid, _, _ in EPISODES:
            assert _json(data + "ep/" + q(site(file)))["_meta"]["episode_id"] == eid
        cindex = _json(data + "compare/index.json")
        key = cindex["site_names"][COMPARE_KEY]
        assert [r["file"] for r in _json(data + "compare/lists/" + q(key) + ".json")]
        for file in cindex["episodes"]:
            assert _json(data + "compare/" + q(key) + "/" + q(site(file)))["_meta"]["episode_id"]
        assert _json(data + "sensors/" + q(site("ep#1?x=y&z.json")))["format"] == "board-sensors/1"
        (kp,) = _json(data + "keypoints/index.json")["files"].values()
        assert _status(media + kp["path"]) == 200
        # every video and frame path a record carries, requested exactly as the page does (media base + path)
        videos = {rel: _status(media + rel) for r in records for rel in r["_media"].values()}
        frames = {rel: _status(media + rel) for r in records for rel in r["_frames"].values()}
        build = _json(root + "b1/BUILD.json")
    finally:
        httpd.shutdown()
        httpd.server_close()
    no_source = {rel for r in records if r["file"] == "no_clip.json" for rel in r["_media"].values()}
    assert frames and set(frames.values()) == {200}
    assert len(videos) == build["media"]["videos"] and len(no_source) == build["media"]["videos_no_source"]
    # BUILD.json lists exactly the videos that are not served and have a source to make them from
    assert set(build["missing_media"]) == {gone} == {rel for rel, s in videos.items() if s != 200} - no_source
    assert all(s == 200 for rel, s in videos.items() if rel not in no_source | {gone})


def _rename(board: Path, stems: dict, values: dict, compare_keys: dict) -> None:
    """Give a built board's identifiers other names wherever it holds them, in file names and in every JSON key and
    value: each episode stem in stems wherever it occurs, each dataset in values where a whole value or key is it,
    and each model key in compare_keys likewise, in the board's compare/ only."""
    def swap(s: str, compare: bool) -> str:
        whole = {**values, **(compare_keys if compare else {})}
        if s in whole:
            return whole[s]
        for old, new in stems.items():
            s = s.replace(old, new)
        return s

    def walk(v, compare):
        if isinstance(v, dict):
            return {swap(k, compare): walk(x, compare) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, compare) for x in v]
        return swap(v, compare) if isinstance(v, str) else v
    for p in board.rglob("*.json"):
        p.write_text(json.dumps(walk(json.loads(p.read_text()), "compare" in p.relative_to(board).parts)))
    for p in sorted(board.rglob("*"), key=lambda p: -len(p.parts)):    # the deepest first
        new = swap(p.name, "compare" in p.relative_to(board).parts[:-1])
        if new != p.name:
            p.rename(p.with_name(new))


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_static_page_reaches_every_file_of_a_board_with_hostile_names(tmp_path):
    """The page itself (tests/render_smoke.js) over a static build whose datasets, episode files and model key all
    need encoding: every episode, every model under Labels by and the comparison, with no request that fails."""
    from test_board_serve import _built_board, _smoke
    board = _built_board(tmp_path)
    _rename(board, {"episode_000000": "episode_café #1", "episode_000001": "Ep 50% ?x=1",
                    "episode_000002": "日本語 エピソード"},
            {"molmo": "Kitchen / day 1", "openaoe": "kitchen / day 1", "mine": "100% ../x"}, {"other": COMPARE_KEY})
    (tmp_path / "clips").mkdir()
    a = argparse.Namespace(board=board, qa=board / "qa", clips=tmp_path / "clips", compare=None, hands=None,
                           keypoints=None, out=tmp_path / "out", build_id="b1", force=False, public_base=None,
                           title="Data Dashboard")
    assert static.cmd_site(a) == 0
    handler = functools.partial(_Quiet, directory=str(tmp_path / "out"))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        res = _smoke(str(tmp_path / "out" / "b1" / "index.html"), f"http://127.0.0.1:{httpd.server_address[1]}/b1/")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert res["episodes"] == 3 and res["rendered"] >= 5 and res["rendered_comparisons"] >= 2 and res["compare_view"]
    assert res["keypoint_links"] == 1 and res["sensors_panels"] == 1 and res["no_label_pages"] == 1
    (lb,) = res["labellers"]
    assert lb["key"] == COMPARE_KEY and lb["episodes"] == 2
    assert res["failed_requests"] == []
