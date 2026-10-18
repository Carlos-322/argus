"""The board page under navigation that outruns its requests (tests/viewer_nav.js), against a live board.

A read that fails is never remembered as the episode, and a response that arrives after the reader moved on never
takes the view back."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from test_board_serve import _built_board, _get, _serve


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_navigation_outrunning_requests(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        (tmp_path / "page.html").write_bytes(_get(url + "/")[2])
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("viewer_nav.js")),
                            str(tmp_path / "page.html"), url + "/"], capture_output=True, text=True, timeout=120)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert r.returncode == 0, r.stderr or r.stdout
    res = json.loads(r.stdout)
    assert not res["first"]["error"] and res["first"]["drawn"] == res["first"]["active"]
    # a 500 with a JSON error body is an error on the page, and is not kept as the episode
    failed = res["failed_read"]
    assert failed["error"] and failed["drawn"] is None and not failed["cached"]
    # opened again once the document reads, the episode is drawn
    again = res["after_repair"]
    assert not again["error"] and again["drawn"] == again["file"] == again["active"] and again["cached"]
    # Back while the comparison was on its way (its metrics, or its fade): the episode stays, as the URL says
    for case in ("back_while_metrics_load", "back_while_view_fades"):
        back = res[case]
        assert not back["compare_view"] and not back["fading"] and back["search"] == back["back_to"], case
        assert back["drawn"] == back["active"], case
    # the same steps without Back do open the comparison, so the cases above are not passing by doing nothing
    assert res["no_back"]["compare_view"] and res["no_back"]["search"].startswith("?view=compare")
    # random clicks, tabs, labellers, comparisons and Backs, each outrunning randomly late responses: the page always
    # ends showing what its URL says (the page before this fix ended wrong in 8 of these 25)
    assert res["random"]["runs"] == 25 and res["random"]["wrong"] == []
