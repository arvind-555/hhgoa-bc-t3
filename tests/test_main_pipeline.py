"""Piece 6 - orchestration flow of src/main.py.

The heavy pieces (face_encoder, face_search_fallback) are swapped for fakes via
sys.modules; chain_verify is the real module with upload_hash/verify stubbed.
No dlib, no browser, no network. This checks the wiring and the exit codes -
each piece has its own tests for its internals.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_verify as cv  # noqa: E402
import main as pipeline  # noqa: E402  (main.py only imports stdlib + pipeline_log at top)

_MATCH = {"source_url": "https://example.social/@dana/posts/9", "title": "sunset", "rank": 1}


def _encoding_record(path, face_index=None, model="hog"):
    return {
        "encoding_dim": 128,
        "face_index": face_index or 0,
        "faces_detected": 1,
        "face_box": {"top": 10, "right": 60, "bottom": 110, "left": 20},
        "meta": {"source_image": Path(path).name, "source_sha256": "ab" * 32, "model": model},
    }


def _fake_fe(encode=None):
    err = type("FaceEncoderError", (Exception,), {})
    return types.SimpleNamespace(
        FaceEncoderError=err,
        encode_face=encode or _encoding_record,
    )


def _fake_fsf(matches, meta=None):
    err = type("FallbackError", (Exception,), {})
    meta = meta or {"match_strength": "strong",
                    "result_source": "the 'Pages with this image' tab",
                    "bing_note": None}
    return types.SimpleNamespace(
        FallbackError=err,
        bing_visual_search=lambda image_path, headed=False, timeout=45.0: (
            list(matches), None, dict(meta)),
    )


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "me.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    return p


@pytest.fixture
def wire(monkeypatch):
    """Install fake face_encoder / face_search_fallback; return a helper to set
    chain_verify.upload_hash / verify."""
    def _install(fe, fsf, *, upload=None, verify=None):
        monkeypatch.setitem(sys.modules, "face_encoder", fe)
        monkeypatch.setitem(sys.modules, "face_search_fallback", fsf)
        if upload is not None:
            monkeypatch.setattr(cv, "upload_hash", upload)
        if verify is not None:
            monkeypatch.setattr(cv, "verify", verify)
    return _install


def test_missing_image_is_exit_4(tmp_path):
    assert pipeline.run(tmp_path / "nope.jpg") == pipeline.EXIT_BAD_INPUT


def test_unfunded_runs_1_to_3_then_stops_at_4(wire, img, tmp_path):
    verified = []
    wire(
        _fake_fe(), _fake_fsf([_MATCH]),
        upload=lambda post, **kw: {"tx_hash": None, "hash": cv.hash_post(post),
                                   "polygonscan_url": None, "broadcast": False,
                                   "reason": "wallet 0x.. holds 0 POL"},
        verify=lambda *a, **k: verified.append(1) or True,
    )
    out = tmp_path / "trace.json"
    rc = pipeline.run(str(img), out_path=str(out))

    assert rc == pipeline.EXIT_UNFUNDED
    assert verified == []  # step 5 never ran

    t = json.loads(out.read_text(encoding="utf-8"))
    assert t["outcome"] == "stopped-unfunded"
    assert t["matched_post"]["matched_url"] == _MATCH["source_url"]
    assert t["matched_post"]["platform"] == "example.social"
    assert t["notarization"]["broadcast"] is False
    by_step = {s["step"]: s["status"] for s in t["steps"]}
    assert by_step[1] == "ok" and by_step[2] == "ok" and by_step[3] == "ok"
    assert by_step[4] == "stopped" and by_step[5] == "skipped"


def test_no_match_stops_at_step_3(wire, img, tmp_path):
    wire(_fake_fe(), _fake_fsf([], meta={"match_strength": "weak",
                                         "result_source": "default view",
                                         "bing_note": "Unable to find pages with this image"}),
         upload=lambda *a, **k: pytest.fail("upload_hash must not be called"),
         verify=lambda *a, **k: pytest.fail("verify must not be called"))
    out = tmp_path / "trace.json"
    rc = pipeline.run(str(img), out_path=str(out))

    assert rc == pipeline.EXIT_NO_MATCH
    t = json.loads(out.read_text(encoding="utf-8"))
    assert t["outcome"] == "no-match"
    assert t["matched_post"] is None
    by_step = {s["step"]: s["status"] for s in t["steps"]}
    assert by_step[1] == "ok" and by_step[2] == "ok"
    assert by_step[3] == "stopped" and by_step[4] == "skipped"


def test_full_success_round_trip(wire, img, tmp_path):
    tx = "0x" + "cd" * 32
    seen: dict = {}

    def fake_upload(post, **kw):
        seen["uploaded_hash"] = cv.hash_post(post)
        return {"tx_hash": tx, "hash": seen["uploaded_hash"],
                "polygonscan_url": f"https://amoy.polygonscan.com/tx/{tx}", "broadcast": True}

    def fake_verify(post, txh, **kw):
        seen["verified_tx"] = txh
        return cv.hash_post(post) == seen["uploaded_hash"]

    wire(_fake_fe(), _fake_fsf([_MATCH]), upload=fake_upload, verify=fake_verify)
    out = tmp_path / "trace.json"
    rc = pipeline.run(str(img), out_path=str(out))

    assert rc == pipeline.EXIT_OK
    assert seen["verified_tx"] == tx
    t = json.loads(out.read_text(encoding="utf-8"))
    assert t["outcome"] == "complete"
    assert t["verified"] is True
    assert t["notarization"]["tx_hash"] == tx
    assert [s["status"] for s in t["steps"]] == ["ok", "ok", "ok", "ok", "ok"]
    # the hash that was uploaded is the hash of the matched_post in the trace
    assert cv.hash_post(t["matched_post"]) == t["notarization"]["hash"]


def test_face_step_failure_is_exit_5(wire, img, tmp_path):
    fe = _fake_fe()

    def boom(path, face_index=None, model="hog"):
        raise fe.FaceEncoderError("no face detected")

    fe.encode_face = boom
    wire(fe, _fake_fsf([_MATCH]))
    out = tmp_path / "trace.json"
    rc = pipeline.run(str(img), out_path=str(out))

    assert rc == pipeline.EXIT_STEP_FAILED
    t = json.loads(out.read_text(encoding="utf-8"))
    assert t["outcome"] == "failed"
    assert t["steps"][0] == {"step": 1, "name": "face_encoder", "status": "failed",
                             "error": "no face detected"}


def test_verify_mismatch_after_upload_is_exit_5(wire, img, tmp_path):
    tx = "0x" + "ee" * 32
    wire(
        _fake_fe(), _fake_fsf([_MATCH]),
        upload=lambda post, **kw: {"tx_hash": tx, "hash": cv.hash_post(post),
                                   "polygonscan_url": "u", "broadcast": True},
        verify=lambda *a, **k: False,
    )
    out = tmp_path / "trace.json"
    assert pipeline.run(str(img), out_path=str(out)) == pipeline.EXIT_STEP_FAILED
    t = json.loads(out.read_text(encoding="utf-8"))
    assert t["outcome"] == "failed" and t["verified"] is False
