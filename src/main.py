"""Piece 6 - the whole trace as one command.

    python src/main.py path/to/image.jpg

Runs, in order, printing a step-by-step trace (pipeline_log convention):

  1. face_encoder            - detect + encode the face
  2. face_search_fallback    - Bing reverse-image search for that photo
  3. (extract)               - pick the best matched page + its metadata
  4. chain_verify.upload_hash - notarize the matched post on Polygon Amoy
  5. chain_verify.verify      - confirm it round-trips off-chain

If the wallet can't pay for gas, steps 1-3 still run in full; the pipeline then
stops cleanly at step 4 with the funding message (exit 2) - it does not crash
and does not skip the chain step silently.

Consent-bound: only run on yourself or people who have given explicit permission.

Exit codes:
  0  complete   - face -> match -> notarized -> verified
  2  unfunded   - ran steps 1-3, stopped at step 4 (fund the wallet, re-run)
  3  no match   - ran steps 1-2, the search returned nothing to notarize
  4  bad input  - image missing / a required dependency not installed
  5  step failed - a stage errored out (details in the log + trace file)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
from pathlib import Path

from pipeline_log import fail, get_logger, step

log = get_logger("pipeline")

EXIT_OK = 0
EXIT_UNFUNDED = 2
EXIT_NO_MATCH = 3
EXIT_BAD_INPUT = 4
EXIT_STEP_FAILED = 5


# --------------------------------------------------------------------------
# step 3 - turn search results into a "matched post"
# --------------------------------------------------------------------------
def extract_matched_post(matches: list[dict], encoding_record: dict,
                         search_meta: dict, image_path) -> dict | None:
    """Best candidate page + whatever metadata is available. None only if the
    search produced no usable page link in EITHER tier (exact_page or
    visual_similarity).

    An exact_page hit is preferred; a visual_similarity hit is accepted and
    labelled as such - step 3 no longer stops just because the only results are
    visually-similar.
    """
    usable = [m for m in matches if m.get("source_url")]
    if not usable:
        return None
    # prefer exact_page over visual_similarity, then by rank
    best = min(usable, key=lambda m: (m.get("match_type") != "exact_page",
                                      m.get("rank", 99)))
    url = best["source_url"]
    meta = encoding_record.get("meta", {})
    return {
        "matched_url": url,
        "title": best.get("title"),
        "platform": urllib.parse.urlparse(url).hostname or "",
        "rank": best.get("rank", 1),
        "match_type": best.get("match_type") or search_meta.get("match_type"),
        "candidates_found": len(usable),
        "exact_page_count": search_meta.get("exact_page_count"),
        "visual_similarity_count": search_meta.get("visual_similarity_count"),
        "discovered_via": search_meta.get("result_source") or "bing-images-visual-search",
        "match_strength": search_meta.get("match_strength"),
        "query_image": meta.get("source_image") or Path(image_path).name,
        "query_image_sha256": meta.get("source_sha256"),
        "query_face_index": encoding_record.get("face_index"),
        "traced_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _face_summary(rec: dict) -> dict:
    m = rec.get("meta", {})
    return {
        "encoding_dim": rec.get("encoding_dim"),
        "face_index": rec.get("face_index"),
        "faces_detected": rec.get("faces_detected"),
        "face_box": rec.get("face_box"),
        "source_image": m.get("source_image"),
        "source_sha256": m.get("source_sha256"),
        "model": m.get("model"),
    }


def _write_trace(trace: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    log.info("Trace written to %s", p)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run(
    image_path: str | Path,
    *,
    face_index: int | None = None,
    model: str = "hog",
    headed: bool = False,
    search_timeout: float = 45.0,
    env_file: str = ".env",
    rpc_url: str | None = None,
    out_path: str | None = None,
) -> int:
    image_path = Path(image_path)
    if not image_path.is_file():
        return fail(log, f"input image not found: {image_path}", EXIT_BAD_INPUT)

    try:
        import face_encoder as fe
    except ImportError as exc:
        return fail(log, f"step 1 needs face_recognition/dlib installed ({exc}) - see README",
                    EXIT_BAD_INPUT)
    import chain_verify as cv
    import face_search_fallback as fsf

    out_path = out_path or f"output/trace_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    trace: dict = {
        "input_image": str(image_path),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "steps": [],
        "outcome": "incomplete",
    }

    def note(n, name, status, **detail):
        trace["steps"].append({"step": n, "name": name, "status": status, **detail})

    log.info("=" * 68)
    log.info("HH Goa Task 3 - full trace:  %s", image_path)
    log.info("=" * 68)

    # ---------- 1. face detection + encoding ----------
    log.info("[STEP 1/5] face_encoder - detect + encode the face")
    try:
        with step(log, "face_encoder.encode_face"):
            record = fe.encode_face(str(image_path), face_index=face_index, model=model)
    except fe.FaceEncoderError as exc:
        note(1, "face_encoder", "failed", error=str(exc))
        trace["outcome"] = "failed"
        _write_trace(trace, out_path)
        return fail(log, f"step 1 (face encoding) failed: {exc}", EXIT_STEP_FAILED)
    trace["face"] = _face_summary(record)
    note(1, "face_encoder", "ok",
         faces_detected=record["faces_detected"], face_index=record["face_index"])
    log.info("  -> %d face(s) detected; encoded #%d as a %d-d vector; source sha256 %s..",
             record["faces_detected"], record["face_index"], record["encoding_dim"],
             record["meta"]["source_sha256"][:16])

    # ---------- 2. reverse image search ----------
    log.info("[STEP 2/5] face_search_fallback - Bing reverse-image search")
    try:
        with step(log, "face_search_fallback.bing_visual_search"):
            matches, debug_dump, search_meta = fsf.bing_visual_search(
                image_path, headed=headed, timeout=search_timeout)
    except fsf.FallbackError as exc:
        note(2, "face_search_fallback", "failed", error=str(exc))
        trace["outcome"] = "failed"
        _write_trace(trace, out_path)
        return fail(log, f"step 2 (reverse image search) failed: {exc}", EXIT_STEP_FAILED)
    trace["search"] = {
        "candidates": matches,
        "match_strength": search_meta.get("match_strength"),
        "match_type": search_meta.get("match_type"),
        "exact_page_count": search_meta.get("exact_page_count"),
        "visual_similarity_count": search_meta.get("visual_similarity_count"),
        "result_source": search_meta.get("result_source"),
        "bing_note": search_meta.get("bing_note"),
        "visual_matches_fallback": search_meta.get("visual_matches_fallback"),
        "debug_dump": debug_dump,
    }
    note(2, "face_search_fallback", "ok",
         candidates=len(matches), match_strength=search_meta.get("match_strength"),
         exact_page=search_meta.get("exact_page_count"),
         visual_similarity=search_meta.get("visual_similarity_count"))
    log.info("  -> %d candidate(s): %s exact_page + %s visual_similarity  ->  match_strength=%s",
             len(matches), search_meta.get("exact_page_count"),
             search_meta.get("visual_similarity_count"),
             search_meta.get("match_strength") or "?")
    for m in matches[:8]:
        log.info("       #%-2s [%s]  %s", m.get("rank", "?"),
                 m.get("match_type", "?"), m.get("source_url"))

    # sanity check: an "exact_page" set that looks like SEO / content-marketing
    # sites is the classic "scraped the wrong panel" false positive - do not
    # notarize that as a high-confidence exact match.
    exact = [m for m in matches if m.get("match_type") == "exact_page"]
    if exact:
        cm = fsf.content_marketing_ratio(exact)
        if cm >= 0.5:
            hosts = ", ".join(urllib.parse.urlparse(m["source_url"]).hostname
                              for m in exact[:6])
            log.error("  !! %.0f%% of the exact_page candidates are generic "
                      "content-marketing / listicle sites (%s)", cm * 100, hosts)
            log.error("  !! this is the known DOM-misread false-positive pattern - "
                      "NOT notarizing these as an exact match")
            note(2, "face_search_fallback", "suspect_exact_page",
                 content_marketing_ratio=round(cm, 2), hosts=hosts)
            trace["search"]["exact_page_rejected"] = {
                "reason": "content_marketing_ratio>=0.5", "ratio": round(cm, 2),
                "hosts": hosts, "urls": [m["source_url"] for m in exact],
            }
            # drop the suspect exact_page tier; keep only genuine visual_similarity
            matches = [m for m in matches if m.get("match_type") != "exact_page"]
            for r, m in enumerate(matches, 1):
                m["rank"] = r
            search_meta = {**search_meta, "match_type": "visual_similarity" if matches else None,
                           "exact_page_count": 0,
                           "match_strength": "moderate" if matches else "none",
                           "result_source": "exact_page tier rejected as content-marketing; "
                           + str(search_meta.get("result_source") or "")}
            trace["search"]["match_strength"] = search_meta["match_strength"]
            trace["search"]["candidates"] = matches

    # ---------- 3. extract the matched post ----------
    log.info("[STEP 3/5] extract - pick the best matched post")
    matched_post = extract_matched_post(matches, record, search_meta, image_path)
    if matched_post is None:
        trace["matched_post"] = None
        trace["outcome"] = "no-match"
        note(3, "extract", "stopped", reason="no usable link in either tier "
             "(exact_page / visual_similarity)")
        note(4, "chain_verify.upload_hash", "skipped", reason="nothing to notarize")
        note(5, "chain_verify.verify", "skipped", reason="nothing to notarize")
        _write_trace(trace, out_path)
        log.warning("  -> no matched post: 0 candidates in exact_page AND visual_similarity")
        if search_meta.get("bing_note"):
            log.warning("     Bing said: %s", search_meta["bing_note"])
        if not headed:
            log.warning("     Bing degrades headless results - retry with:  "
                        "python src/main.py %s --headed", image_path)
        return fail(log, "stopped at step 3: nothing to notarize (no match in either tier)",
                    EXIT_NO_MATCH)
    trace["matched_post"] = matched_post
    note(3, "extract", "ok", matched_url=matched_post["matched_url"],
         match_type=matched_post.get("match_type"))
    if matched_post.get("match_type") != "exact_page":
        log.warning("  -> best candidate is %s (not an exact-image page)",
                    matched_post.get("match_type"))
    log.info("  -> matched post:")
    for k, v in matched_post.items():
        log.info("       %-18s %s", k, v)

    post_hash = cv.hash_post(matched_post)
    log.info("  -> matched-post SHA-256: %s", post_hash)

    # ---------- 4. notarize on-chain ----------
    log.info("[STEP 4/5] chain_verify.upload_hash - notarize on Polygon Amoy")
    try:
        result = cv.upload_hash(matched_post, env_file=env_file, rpc_url=rpc_url)
    except cv.cs.ChainSetupError as exc:
        note(4, "chain_verify.upload_hash", "failed", error=str(exc))
        trace["outcome"] = "failed"
        _write_trace(trace, out_path)
        return fail(log, f"step 4 (chain upload) failed: {exc}", EXIT_STEP_FAILED)

    trace["notarization"] = {
        "hash": result["hash"],
        "tx_hash": result["tx_hash"],
        "polygonscan_url": result["polygonscan_url"],
        "broadcast": result["broadcast"],
        "reason": result.get("reason"),
    }

    if not result["broadcast"]:
        trace["outcome"] = "stopped-unfunded"
        note(4, "chain_verify.upload_hash", "stopped", reason=result.get("reason"))
        note(5, "chain_verify.verify", "skipped", reason="nothing was broadcast")
        _write_trace(trace, out_path)
        log.warning("  -> STOP at step 4: %s", result.get("reason"))
        log.warning("     Steps 1-3 completed; the matched post + its hash are in %s", out_path)
        log.warning("     Fund the wallet:   python src/chain_setup.py")
        log.warning("     Then re-run:        python src/main.py %s", image_path)
        return EXIT_UNFUNDED

    note(4, "chain_verify.upload_hash", "ok", tx_hash=result["tx_hash"])
    log.info("  -> notarized. tx %s", result["tx_hash"])
    log.info("     %s", result["polygonscan_url"])

    # ---------- 5. verify the round-trip ----------
    log.info("[STEP 5/5] chain_verify.verify - confirm the on-chain record")
    try:
        matched = cv.verify(matched_post, result["tx_hash"], env_file=env_file, rpc_url=rpc_url)
    except (cv.VerifyError, cv.cs.ChainSetupError) as exc:
        note(5, "chain_verify.verify", "failed", error=str(exc))
        trace["outcome"] = "failed"
        _write_trace(trace, out_path)
        return fail(log, f"step 5 (verify) failed: {exc}", EXIT_STEP_FAILED)

    trace["verified"] = matched
    if not matched:
        note(5, "chain_verify.verify", "failed", reason="hash mismatch right after upload")
        trace["outcome"] = "failed"
        _write_trace(trace, out_path)
        return fail(log, "step 5: the on-chain hash did NOT match the matched post "
                    "(should be impossible immediately after upload)", EXIT_STEP_FAILED)

    note(5, "chain_verify.verify", "ok", matched=True)
    trace["outcome"] = "complete"
    _write_trace(trace, out_path)

    log.info("=" * 68)
    log.info("PIPELINE COMPLETE  -  face -> match -> notarized -> verified")
    log.info("  matched url : %s", matched_post["matched_url"])
    log.info("  post hash   : %s", result["hash"])
    log.info("  tx          : %s", result["tx_hash"])
    log.info("  polygonscan : %s", result["polygonscan_url"])
    log.info("  trace       : %s", out_path)
    log.info("=" * 68)
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main",
        description="HH Goa Task 3 - run the whole trace: face -> reverse image search "
                    "-> notarize on Polygon Amoy -> verify.",
    )
    p.add_argument("image", help="path to the input photo (consented faces only)")
    p.add_argument("--face-index", type=int, default=None, metavar="N",
                   help="which face to use when the photo has more than one")
    p.add_argument("--model", choices=["hog", "cnn"], default="hog",
                   help="face detector (cnn needs a CUDA dlib build)")
    p.add_argument("--headed", action="store_true",
                   help="show the browser during the Bing search (far more reliable results)")
    p.add_argument("--search-timeout", type=float, default=45.0,
                   help="per-step timeout for the browser search (seconds)")
    p.add_argument("--env-file", default=".env", help="dotenv file with the wallet key")
    p.add_argument("--rpc-url", default=None, help="Polygon Amoy RPC override")
    p.add_argument("--out", default=None,
                   help="trace JSON path (default: output/trace_<timestamp>.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        args.image,
        face_index=args.face_index,
        model=args.model,
        headed=args.headed,
        search_timeout=args.search_timeout,
        env_file=args.env_file,
        rpc_url=args.rpc_url,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
