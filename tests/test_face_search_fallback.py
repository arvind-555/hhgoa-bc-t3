"""Piece 2 fallback (Bing / Playwright) - plumbing tests.

No browser is launched here and no network is touched: these cover the pure
URL-filtering / input-resolution logic only. The real check is the manual
--headed run in the README ("Piece 2 (fallback)").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import face_search_fallback as fsf  # noqa: E402


def test_is_offsite():
    assert fsf.is_offsite("https://instagram.com/p/abc")
    assert fsf.is_offsite("https://news.ycombinator.com/item?id=1")
    assert not fsf.is_offsite("https://www.bing.com/images/search?q=x")
    assert not fsf.is_offsite("https://cn.bing.com/images/search")
    assert not fsf.is_offsite("https://support.microsoft.com/x")
    assert not fsf.is_offsite("not-a-url")
    assert not fsf.is_offsite("")


def test_extract_page_urls_filters_dedupes_ranks():
    anchors = [
        {"href": "https://www.bing.com/images/search?q=a", "text": "bing"},
        {"href": "https://example.com/post/1", "text": "  Post   One "},
        {"href": "https://example.com/post/1/", "text": "dup"},
        {"href": "https://other.org/x", "text": "Other"},
        {"href": "javascript:void(0)", "text": "js"},
        {"href": "https://foo.com/images/search?view=detail", "text": "not-bing image search"},
    ]
    out = fsf.extract_page_urls(anchors)
    assert [m["source_url"] for m in out] == [
        "https://example.com/post/1",
        "https://other.org/x",
    ]
    assert [m["rank"] for m in out] == [1, 2]
    assert out[0]["title"] == "Post One"


def test_extract_page_urls_empty():
    assert fsf.extract_page_urls([]) == []


def test_resolve_input_requires_exactly_one():
    with pytest.raises(fsf.BadInput):
        fsf.resolve_input(None, None)
    with pytest.raises(fsf.BadInput):
        fsf.resolve_input("a.jpg", "b.json")


def test_resolve_input_missing_image():
    with pytest.raises(fsf.BadInput):
        fsf.resolve_input("nope/missing.jpg", None)


def test_resolve_input_image_ok(tmp_path):
    p = tmp_path / "q.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    path, prov, cleanup = fsf.resolve_input(str(p), None)
    assert path == p
    assert prov["via"] == "image"
    assert prov["cropped"] is False
    cleanup()  # no-op, must not raise


def test_resolve_from_encoding_sha_mismatch(tmp_path):
    img = tmp_path / "me.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    rec = {
        "face_box": {"top": 0, "right": 1, "bottom": 1, "left": 0},
        "meta": {"source_image": img.name, "source_sha256": "0" * 64},
    }
    rp = tmp_path / "rec.json"
    rp.write_text(json.dumps(rec))
    with pytest.raises(fsf.BadInput, match="does not match"):
        fsf.resolve_input(None, str(rp), image_dir=str(tmp_path), crop=False)


def test_resolve_from_encoding_missing_source_name(tmp_path):
    rp = tmp_path / "rec.json"
    rp.write_text(json.dumps({"meta": {}}))
    with pytest.raises(fsf.BadInput, match="meta.source_image"):
        fsf.resolve_input(None, str(rp))


def test_build_report_shape():
    rep = fsf.build_report(
        [{"rank": 1, "source_url": "https://x.com/a", "title": None}],
        {"source": "me.jpg", "cropped": False},
        {"headed": False, "result_source": "the 'Pages with this image' tab",
         "match_strength": "strong"},
    )
    assert rep["match_count"] == 1
    assert rep["query"]["engine"] == fsf.ENGINE
    assert rep["query"]["headed"] is False
    assert rep["query"]["match_strength"] == "strong"
    assert rep["query"]["result_source"] == "the 'Pages with this image' tab"
    assert "retrieved_utc" in rep


def test_pages_tab_regex():
    assert fsf._PAGES_TAB_RE.search("Pages with this image")
    assert fsf._PAGES_TAB_RE.search("PAGES THAT INCLUDE MATCHING IMAGES")
    assert fsf._PAGES_TAB_RE.search("Pages Including Matching Images")
    assert not fsf._PAGES_TAB_RE.search("Visual Matches")
    assert not fsf._PAGES_TAB_RE.search("Overview")


def test_exit_codes_distinct():
    codes = [
        fsf.EXIT_OK, fsf.EXIT_ZERO_MATCHES, fsf.EXIT_NO_BROWSER, fsf.EXIT_BAD_INPUT,
        fsf.EXIT_SCRAPE, fsf.EXIT_CAPTCHA, fsf.EXIT_TIMEOUT,
    ]
    assert len(codes) == len(set(codes))
    assert fsf.Captcha("x").exit_code == fsf.EXIT_CAPTCHA
    assert fsf.ScrapeError("x").exit_code == fsf.EXIT_SCRAPE
