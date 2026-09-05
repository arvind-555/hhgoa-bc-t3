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
        [{"rank": 1, "source_url": "https://x.com/a", "title": None,
          "match_type": "exact_page"}],
        {"source": "me.jpg", "cropped": False},
        {"headed": False, "result_source": "1 exact page", "match_strength": "high",
         "match_type": "exact_page", "exact_page_count": 1, "visual_similarity_count": 0},
    )
    assert rep["match_count"] == 1
    assert rep["query"]["engine"] == fsf.ENGINE
    assert rep["query"]["match_strength"] == "high"
    assert rep["query"]["match_type"] == "exact_page"
    assert "retrieved_utc" in rep


def test_pages_tab_regex():
    assert fsf._PAGES_TAB_RE.search("Pages with this image")
    assert fsf._PAGES_TAB_RE.search("PAGES THAT INCLUDE MATCHING IMAGES")
    assert not fsf._PAGES_TAB_RE.search("Visual Matches")
    assert fsf._VISUAL_MATCHES_RE.search("Visual Matches")
    assert not fsf._VISUAL_MATCHES_RE.search("Pages with this image")


def test_unwrap_bing_redirect():
    import base64
    real = "https://in.pinterest.com/pin/12345/"
    b64 = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{b64}&ntb=1"
    assert fsf.unwrap_bing_redirect(wrapped) == real
    # non-wrappers pass through unchanged
    assert fsf.unwrap_bing_redirect("https://example.com/x") == "https://example.com/x"
    assert fsf.unwrap_bing_redirect("https://www.bing.com/images/search?q=x") == \
        "https://www.bing.com/images/search?q=x"
    # garbage u= -> unchanged, no crash
    assert fsf.unwrap_bing_redirect("https://www.bing.com/ck/a?u=a1@@@notb64@@@").startswith(
        "https://www.bing.com/ck/a")


def test_extract_page_urls_unwraps_redirects():
    import base64
    real = "https://linkedin.com/in/arvind"
    b64 = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
    anchors = [{"href": f"https://www.bing.com/ck/a?!&&u=a1{b64}", "text": "Arvind"}]
    out = fsf.extract_page_urls(anchors)
    assert out == [{"source_url": real, "title": "Arvind", "rank": 1}]


_M = [{"source_url": "https://example.social/@dana/1", "title": "t", "rank": 1}]


def test_score_matches_high_when_exact_page_present():
    strength, src = fsf.score_matches(exact_matches=_M, visual_matches=[])
    assert strength == "high"
    assert "exact image" in src


def test_score_matches_moderate_for_visual_only():
    strength, src = fsf.score_matches(exact_matches=[], visual_matches=_M)
    assert strength == "moderate"
    assert "visually-similar" in src

    # exact wins even if both tiers have results
    assert fsf.score_matches(_M, _M)[0] == "high"


def test_score_matches_none_when_both_empty():
    strength, src = fsf.score_matches([], [])
    assert strength == "none"
    assert not (strength in ("high", "moderate"))


def test_score_matches_never_contradicts_tiers():
    for ex, vis in [([], []), (_M, []), ([], _M), (_M, _M)]:
        strength, _ = fsf.score_matches(ex, vis)
        assert (strength == "high") == bool(ex)
        assert (strength == "moderate") == (not ex and bool(vis))
        assert (strength == "none") == (not ex and not vis)


def test_tag_matches_stamps_type_and_reranks():
    ms = [{"source_url": "https://a.com"}, {"source_url": "https://b.com"}]
    fsf.tag_matches(ms, fsf.MATCH_TYPE_VISUAL)
    assert [m["match_type"] for m in ms] == ["visual_similarity", "visual_similarity"]
    assert [m["rank"] for m in ms] == [1, 2]


# ---- the false-positive that shipped: office-design blogs labelled exact_page --
# real URLs from the failing run on a 2-people indoor photo
_FALSE_EXACT = [
    {"source_url": "https://www.bizbash.com/meetings/13-inspiration-sparking-design-ideas-for-meeting-rooms"},
    {"source_url": "https://aia-india.com/15-creative-meeting-room-ideas-that-spark-collaboration/"},
    {"source_url": "https://suite101.com/30-office-meeting-corner-ideas/"},
    {"source_url": "https://blog.armerboard.com/11-best-business-meeting-room-setup-ideas"},
    {"source_url": "https://www.pinterest.com/ideas/office-meeting-setup-inspiration/907114594866/"},
    {"source_url": "https://roomagine.ai/blog/conference-room-setup-ideas/"},
]
_REAL_PAGES = [
    {"source_url": "https://www.linkedin.com/in/arvind-xyz"},
    {"source_url": "https://twitter.com/arvind/status/1839..."},
    {"source_url": "https://someblog.com/2026/03/our-team-offsite"},
]


def test_looks_like_content_marketing_flags_listicles():
    for m in _FALSE_EXACT:
        assert fsf.looks_like_content_marketing(m["source_url"]), m["source_url"]
    for m in _REAL_PAGES:
        assert not fsf.looks_like_content_marketing(m["source_url"]), m["source_url"]
    assert fsf.content_marketing_ratio(_FALSE_EXACT) == 1.0
    assert fsf.content_marketing_ratio(_REAL_PAGES) == 0.0


def test_assign_tiers_bing_no_match_card_yields_no_exact():
    """The reported bug: Bing said 'Unable to find pages with this image' but the
    scraper still returned #b_results web results as exact_page/high."""
    exact, visual, notes = fsf.assign_tiers(
        "serp:no-exact-matches", _FALSE_EXACT,
        bing_zero_msg="Unable to find pages with this image")
    assert exact == []          # <-- must NOT be labelled exact_page
    assert visual == []         # the text-query fallback list is dropped entirely
    assert any("no_exact" in n for n in notes)

    # even if _collect_result_anchors mislabels the mode, the error message alone
    # is enough to suppress exact_page
    exact, _, _ = fsf.assign_tiers("serp:vsa3-weblist", _FALSE_EXACT,
                                   bing_zero_msg="Unable to find pages with this image")
    assert exact == []


def test_assign_tiers_vsa3_weblist_is_not_exact():
    exact, visual, notes = fsf.assign_tiers("serp:vsa3-weblist", _FALSE_EXACT, bing_zero_msg="")
    assert exact == []                      # on vsa=3 but no matching-section -> not exact
    assert [m["source_url"] for m in visual] == [m["source_url"] for m in _FALSE_EXACT]
    assert any("text-query" in n or "NOT labelled" in n for n in notes)


def test_assign_tiers_matching_section_with_real_pages_is_exact():
    exact, visual, notes = fsf.assign_tiers("serp:matching-section", _REAL_PAGES, bing_zero_msg="")
    assert [m["source_url"] for m in exact] == [m["source_url"] for m in _REAL_PAGES]
    assert visual == []


def test_assign_tiers_demotes_content_marketing_even_from_matching_section():
    """Defense in depth: even a 'matching-section' scrape gets demoted if the
    domains are overwhelmingly SEO listicles (misidentified panel)."""
    exact, visual, notes = fsf.assign_tiers("serp:matching-section", _FALSE_EXACT, bing_zero_msg="")
    assert exact == []
    assert len(visual) == len(_FALSE_EXACT)
    assert any("demoted" in n for n in notes)


def test_assign_tiers_candidate_domains_relate_to_claimed_tier():
    """The test the bug report asked for: whatever we label 'exact_page' must not
    look like a generic content-marketing result set."""
    for mode in ("serp:no-exact-matches", "serp:vsa3-weblist", "serp:matching-section",
                 "serp:generic", "document"):
        exact, _, _ = fsf.assign_tiers(mode, _FALSE_EXACT,
                                       bing_zero_msg=("Unable to find pages with this image"
                                                      if mode == "serp:no-exact-matches" else ""))
        assert fsf.content_marketing_ratio(exact) < 0.5, (
            f"mode {mode!r} labelled a content-marketing-heavy set as exact_page")


def test_exit_codes_distinct():
    codes = [
        fsf.EXIT_OK, fsf.EXIT_ZERO_MATCHES, fsf.EXIT_NO_BROWSER, fsf.EXIT_BAD_INPUT,
        fsf.EXIT_SCRAPE, fsf.EXIT_CAPTCHA, fsf.EXIT_TIMEOUT,
    ]
    assert len(codes) == len(set(codes))
    assert fsf.Captcha("x").exit_code == fsf.EXIT_CAPTCHA
    assert fsf.ScrapeError("x").exit_code == fsf.EXIT_SCRAPE
