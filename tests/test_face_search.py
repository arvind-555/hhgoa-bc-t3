"""Piece 2 tests.

HTTP is mocked HERE (in the test file) with `responses`. src/face_search.py
itself never mocks - the one real end-to-end check is the manual run in the
README against the live Lenso.ai API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import face_search as fs  # noqa: E402

responses = pytest.importorskip("responses")

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d4944415478da6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082"
)


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "q.png"
    p.write_bytes(PNG_1PX)
    return p


def _ok_body():
    return {
        "results": [
            {
                "urlList": [
                    {"imageUrl": "https://img.example/a.jpg",
                     "sourceUrl": "https://social.example/post/1",
                     "title": "post one"}
                ],
                "base64Image": "iVBOR...",
                "confidenceScore": 91.4,
                "date": "2025-02-01",
            },
            {
                "urlList": [
                    {"imageUrl": "https://img.example/b.jpg",
                     "sourceUrl": "https://social.example/post/2",
                     "title": "post two"}
                ],
                "confidenceScore": 74.0,
                "date": "2024-11-10",
            },
        ],
        "availablePages": 3,
    }


def test_missing_key(monkeypatch):
    monkeypatch.delenv(fs.API_KEY_ENV, raising=False)
    monkeypatch.setattr(fs, "_load_dotenv", lambda *a, **k: None)
    with pytest.raises(fs.MissingApiKey):
        fs.get_api_key()


def test_resolve_requires_exactly_one():
    with pytest.raises(fs.BadInput):
        fs.resolve_input(None, None)
    with pytest.raises(fs.BadInput):
        fs.resolve_input("a.jpg", "b.json")


def test_bad_image_path():
    with pytest.raises(fs.BadInput):
        fs.resolve_input("nope/missing.jpg", None)


def test_parse_and_rank():
    matches = fs.parse_matches(_ok_body())
    assert [m["rank"] for m in matches] == [1, 2]
    assert matches[0]["confidence_score"] == 91.4
    assert matches[0]["source_url"] == "https://social.example/post/1"


@responses.activate
def test_live_call_shape(img):
    responses.add(responses.POST, fs.FACE_SEARCH_URL, json=_ok_body(), status=200)
    raw, prov = fs.resolve_input(str(img), None)
    resp = fs.search_faces(raw, "test-key", timeout=5)
    report = fs.build_report(resp, prov, {"sort_type": "QUALITY_DESCENDING", "page": 1})
    assert report["match_count"] == 2
    assert report["matches"][0]["source_url"] == "https://social.example/post/1"
    sent = json.loads(responses.calls[0].request.body)
    assert "image" in sent and sent["page"] == 1
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"


@responses.activate
def test_zero_matches(img):
    responses.add(responses.POST, fs.FACE_SEARCH_URL, json={"results": []}, status=200)
    raw, _ = fs.resolve_input(str(img), None)
    assert fs.parse_matches(fs.search_faces(raw, "k", timeout=5)) == []


@responses.activate
def test_rate_limited(img):
    responses.add(responses.POST, fs.FACE_SEARCH_URL, status=429,
                  headers={"Retry-After": "12"}, body="slow down")
    raw, _ = fs.resolve_input(str(img), None)
    with pytest.raises(fs.RateLimited) as ei:
        fs.search_faces(raw, "k", timeout=5)
    assert ei.value.retry_after == 12
    assert ei.value.exit_code == fs.EXIT_RATE_LIMITED


@responses.activate
def test_api_error(img):
    responses.add(responses.POST, fs.FACE_SEARCH_URL, status=500, body="boom")
    raw, _ = fs.resolve_input(str(img), None)
    with pytest.raises(fs.ApiError) as ei:
        fs.search_faces(raw, "k", timeout=5)
    assert ei.value.status_code == 500


@responses.activate
def test_timeout(img):
    from requests.exceptions import Timeout

    responses.add(responses.POST, fs.FACE_SEARCH_URL, body=Timeout("slow"))
    raw, _ = fs.resolve_input(str(img), None)
    with pytest.raises(fs.ApiTimeout):
        fs.search_faces(raw, "k", timeout=1)


@responses.activate
def test_401(img):
    responses.add(responses.POST, fs.FACE_SEARCH_URL, status=401, body="bad key")
    raw, _ = fs.resolve_input(str(img), None)
    with pytest.raises(fs.ApiError) as ei:
        fs.search_faces(raw, "k", timeout=5)
    assert ei.value.status_code == 401


def test_from_encoding_sha_mismatch(tmp_path, img):
    rec = {"face_box": {"top": 0, "right": 1, "bottom": 1, "left": 0},
           "meta": {"source_image": img.name, "source_sha256": "0" * 64}}
    rp = tmp_path / "rec.json"
    rp.write_text(json.dumps(rec))
    with pytest.raises(fs.BadInput, match="does not match"):
        fs.resolve_input(None, str(rp), image_dir=str(img.parent), crop=False)
