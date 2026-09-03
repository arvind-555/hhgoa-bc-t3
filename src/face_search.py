"""Piece 2 - reverse face search via the Lenso.ai face-search API.

Takes an image (a path, or the encoding record from piece 1 which is used
to locate + integrity-check the original photo), calls the LIVE Lenso.ai
facial-recognition endpoint, and prints the ranked list of matches with
their source URLs.

  Endpoint : POST https://api.eyematch.ai/search   (Lenso face search)
  Auth     : Authorization: Bearer <key>   -- key read from env LENSO_API_KEY
  Docs     : https://github.com/lenso-ai/reverse-image-search-api

There is NO mock, fixture, sample, or hardcoded result anywhere in this
file. Every run performs a real HTTP request; with no key / no network it
fails loudly rather than inventing data.

Handled failure modes:
  * missing API key                 -> exit 3
  * bad image / bad encoding record  -> exit 4
  * API error (4xx/5xx, not 429)     -> exit 5
  * rate limited (HTTP 429)          -> exit 6  (honours Retry-After)
  * timeout / connection failure     -> exit 7
  * zero matches                     -> exit 2  (not an error, just empty)

Usage:
    python src/face_search.py --image data/input/me.jpg
    python src/face_search.py --from-encoding output/me.json
    python src/face_search.py --image data/input/me.jpg --sort QUALITY_DESCENDING --out output/me.matches.json
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import time
from pathlib import Path

import requests

from pipeline_log import fail, get_logger, step

log = get_logger("face_search")

FACE_SEARCH_URL = "https://api.eyematch.ai/search"
API_KEY_ENV = "LENSO_API_KEY"

SORT_TYPES = (
    "QUALITY_DESCENDING",
    "QUALITY_ASCENDING",
    "DATE_DESCENDING",
    "DATE_ASCENDING",
)

EXIT_OK = 0
EXIT_ZERO_MATCHES = 2
EXIT_NO_KEY = 3
EXIT_BAD_INPUT = 4
EXIT_API_ERROR = 5
EXIT_RATE_LIMITED = 6
EXIT_TIMEOUT = 7


class FaceSearchError(Exception):
    exit_code = 1


class MissingApiKey(FaceSearchError):
    exit_code = EXIT_NO_KEY


class BadInput(FaceSearchError):
    exit_code = EXIT_BAD_INPUT


class ApiError(FaceSearchError):
    exit_code = EXIT_API_ERROR

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RateLimited(ApiError):
    exit_code = EXIT_RATE_LIMITED

    def __init__(self, message: str, retry_after: float | None = None, body: str | None = None):
        super().__init__(message, status_code=429, body=body)
        self.retry_after = retry_after


class ApiTimeout(FaceSearchError):
    exit_code = EXIT_TIMEOUT


# --------------------------------------------------------------------------
# env / key
# --------------------------------------------------------------------------
def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal, dependency-free .env loader: KEY=VALUE lines, no override."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def get_api_key() -> str:
    _load_dotenv()
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingApiKey(
            f"no API key: set the {API_KEY_ENV} environment variable "
            f"(or add {API_KEY_ENV}=... to a .env file)"
        )
    log.info("Lenso API key loaded from env %s (%d chars)", API_KEY_ENV, len(key))
    return key


# --------------------------------------------------------------------------
# image input
# --------------------------------------------------------------------------
def _read_image_bytes(image_path: Path) -> bytes:
    if not image_path.is_file():
        raise BadInput(f"image not found: {image_path}")
    data = image_path.read_bytes()
    if not data:
        raise BadInput(f"image is empty: {image_path}")
    return data


def _crop_to_box(image_bytes: bytes, box: dict, pad_frac: float = 0.35) -> bytes:
    """Crop to a piece-1 face_box (top/right/bottom/left) with padding."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dep
        raise BadInput("cropping needs Pillow (pip install Pillow); pass --no-crop") from exc

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    bw = box["right"] - box["left"]
    bh = box["bottom"] - box["top"]
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    left = max(0, box["left"] - px)
    top = max(0, box["top"] - py)
    right = min(w, box["right"] + px)
    bottom = min(h, box["bottom"] + py)
    out = io.BytesIO()
    img.crop((left, top, right, bottom)).save(out, format="JPEG", quality=95)
    return out.getvalue()


def resolve_input(
    image: str | None,
    from_encoding: str | None,
    image_dir: str | Path = "data/input",
    crop: bool = True,
) -> tuple[bytes, dict]:
    """Return (image_bytes, provenance dict) from either --image or --from-encoding."""
    if bool(image) == bool(from_encoding):
        raise BadInput("give exactly one of image / from_encoding")

    if image:
        path = Path(image)
        raw = _read_image_bytes(path)
        log.info("Query image: %s (%.0f KB)", path, len(raw) / 1024)
        return raw, {"source": str(path), "cropped": False, "via": "image"}

    log.info("Resolving query image from encoding record %s", from_encoding)
    rec_path = Path(from_encoding)
    if not rec_path.is_file():
        raise BadInput(f"encoding record not found: {rec_path}")
    try:
        record = json.loads(rec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BadInput(f"encoding record is not valid JSON: {rec_path} ({exc})") from exc

    meta = record.get("meta", {})
    src_name = meta.get("source_image")
    if not src_name:
        raise BadInput(f"encoding record has no meta.source_image: {rec_path}")

    candidates = [Path(image_dir) / src_name, rec_path.parent / src_name, Path(src_name)]
    img_path = next((c for c in candidates if c.is_file()), None)
    if img_path is None:
        raise BadInput(
            f"cannot find source image '{src_name}' from {rec_path}; "
            f"looked in: {', '.join(str(c) for c in candidates)}. "
            f"Pass --image-dir or --image."
        )

    raw = _read_image_bytes(img_path)
    log.info("Found source image: %s (%.0f KB)", img_path, len(raw) / 1024)

    want_sha = meta.get("source_sha256")
    if want_sha:
        import hashlib

        got = hashlib.sha256(raw).hexdigest()
        if got != want_sha:
            raise BadInput(
                f"image {img_path} sha256 {got[:12]}.. does not match the "
                f"encoding record ({want_sha[:12]}..) - wrong or edited file"
            )
        log.info("SHA-256 matches the encoding record (%s)", got[:16] + "..")

    prov: dict = {"source": str(img_path), "cropped": False, "via": "from_encoding",
                  "encoding_record": str(rec_path)}

    box = record.get("face_box")
    if crop and isinstance(box, dict) and {"top", "right", "bottom", "left"} <= box.keys():
        raw = _crop_to_box(raw, box)
        prov["cropped"] = True
        prov["face_index"] = record.get("face_index")
        log.info("Cropped to piece-1 face box #%s -> %.0f KB",
                 record.get("face_index"), len(raw) / 1024)

    return raw, prov


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


# --------------------------------------------------------------------------
# API call
# --------------------------------------------------------------------------
def search_faces(
    image_bytes: bytes,
    api_key: str,
    sort_type: str = "QUALITY_DESCENDING",
    page: int = 1,
    domains_to_include: list[str] | None = None,
    domains_to_exclude: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 0,
    session: requests.Session | None = None,
) -> dict:
    """Perform the live Lenso face search. Returns the parsed JSON response."""
    if sort_type not in SORT_TYPES:
        raise BadInput(f"sort_type must be one of {SORT_TYPES}")

    payload: dict = {"image": image_to_base64(image_bytes), "sortType": sort_type, "page": page}
    if domains_to_include:
        payload["domainsToInclude"] = list(domains_to_include)
    if domains_to_exclude:
        payload["domainsToExclude"] = list(domains_to_exclude)
    if from_date:
        payload["fromDate"] = from_date
    if to_date:
        payload["toDate"] = to_date

    http = session or requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    log.info("POST %s  (sort=%s page=%d payload=%.0f KB timeout=%.0fs)",
             FACE_SEARCH_URL, sort_type, page, len(payload["image"]) / 1024, timeout)

    attempt = 0
    while True:
        attempt += 1
        started = time.monotonic()
        try:
            resp = http.post(
                FACE_SEARCH_URL, json=payload, headers=headers,
                timeout=(10.0, timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise ApiTimeout(
                f"Lenso API did not respond within {timeout}s (POST {FACE_SEARCH_URL})"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ApiTimeout(f"could not connect to Lenso API ({FACE_SEARCH_URL}): {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"request to Lenso API failed: {exc}") from exc

        elapsed = time.monotonic() - started
        log.info("Lenso responded HTTP %d in %.1fs", resp.status_code, elapsed)

        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            if attempt <= max_retries:
                wait = retry_after or min(30.0, 2.0 ** attempt)
                log.warning("Rate limited (HTTP 429); retry %d/%d in %.0fs",
                            attempt, max_retries, wait)
                time.sleep(wait)
                continue
            raise RateLimited(
                "Lenso API rate limit hit (HTTP 429)"
                + (f"; retry after {retry_after:.0f}s" if retry_after else "")
                + " - slow down or check your plan quota",
                retry_after=retry_after,
                body=_short_body(resp),
            )

        if resp.status_code == 401:
            raise ApiError("Lenso API rejected the key (HTTP 401) - check "
                           f"{API_KEY_ENV} and that the subscription is active",
                           status_code=401, body=_short_body(resp))

        if not resp.ok:
            raise ApiError(
                f"Lenso API returned HTTP {resp.status_code}: {_short_body(resp)}",
                status_code=resp.status_code, body=_short_body(resp),
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise ApiError(
                f"Lenso API returned non-JSON body (HTTP {resp.status_code})",
                status_code=resp.status_code, body=_short_body(resp),
            ) from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            when = dt.datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
            return max(0.0, (when - dt.datetime.utcnow()).total_seconds())
        except ValueError:
            return None


def _short_body(resp: requests.Response, limit: int = 500) -> str:
    text = (resp.text or "").strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


# --------------------------------------------------------------------------
# response normalisation
# --------------------------------------------------------------------------
def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_matches(api_response: dict) -> list[dict]:
    """Flatten Lenso 'results' into a ranked list of matches."""
    results = api_response.get("results") or []
    matches: list[dict] = []
    for res in results:
        url_list = res.get("urlList") or []
        sources = [
            {
                "image_url": u.get("imageUrl"),
                "source_url": u.get("sourceUrl"),
                "title": u.get("title"),
            }
            for u in url_list
        ]
        first = sources[0] if sources else {"image_url": None, "source_url": None, "title": None}
        matches.append(
            {
                "confidence_score": _to_float(res.get("confidenceScore")),
                "date": res.get("date"),
                "image_url": first["image_url"],
                "source_url": first["source_url"],
                "title": first["title"],
                "sources": sources,
            }
        )

    matches.sort(key=lambda m: (m["confidence_score"] is not None, m["confidence_score"] or 0.0),
                 reverse=True)
    for rank, m in enumerate(matches, start=1):
        m["rank"] = rank
    top = matches[0]["confidence_score"] if matches else None
    log.info("Lenso returned %d match(es)%s", len(matches),
             f"; top confidence {top:.1f}" if top is not None else "")
    return matches


def build_report(api_response: dict, provenance: dict, request_info: dict) -> dict:
    matches = parse_matches(api_response)
    return {
        "query": {**provenance, **request_info, "endpoint": FACE_SEARCH_URL},
        "match_count": len(matches),
        "available_pages": api_response.get("availablePages"),
        "matches": matches,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_report(report: dict) -> None:
    matches = report["matches"]
    q = report["query"]
    print(f"query source : {q.get('source')}"
          + ("  (cropped to piece-1 face box)" if q.get("cropped") else ""))
    print(f"endpoint     : {q['endpoint']}  sort={q.get('sort_type')} page={q.get('page')}")
    print(f"matches      : {report['match_count']}"
          + (f"  (available pages: {report['available_pages']})"
             if report.get("available_pages") else ""))
    print()

    for m in matches:
        conf = f"{m['confidence_score']:.1f}" if m["confidence_score"] is not None else "  ? "
        print(f"  #{m['rank']:<2} conf={conf:>5}  {m.get('date') or '':<10}  "
              f"{m.get('source_url') or m.get('image_url') or '(no url)'}")
        if m.get("title"):
            print(f"        {m['title']}")

    if not matches:
        return

    top = matches[0]
    print()
    print("=" * 60)
    print("TOP MATCH")
    print("=" * 60)
    conf = f"{top['confidence_score']:.1f}" if top["confidence_score"] is not None else "unknown"
    print(f"  confidence : {conf}")
    print(f"  source     : {top.get('source_url') or '(none)'}")
    print(f"  image      : {top.get('image_url') or '(none)'}")
    print(f"  title      : {top.get('title') or '(none)'}")
    print(f"  date       : {top.get('date') or '(unknown)'}")
    if len(top.get("sources", [])) > 1:
        print(f"  (+{len(top['sources']) - 1} more source page(s) for this face)")
    print("=" * 60)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="face_search",
        description="Live reverse face search via the Lenso.ai API (piece 2).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to the query image")
    src.add_argument("--from-encoding", help="path to a piece-1 encoding JSON (locates the photo)")

    p.add_argument("--image-dir", default="data/input",
                   help="where to look for the photo named in the encoding record")
    p.add_argument("--no-crop", action="store_true",
                   help="with --from-encoding, do NOT crop to the recorded face box")
    p.add_argument("--sort", choices=SORT_TYPES, default="QUALITY_DESCENDING")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--domains-include", default=None,
                   help="comma-separated domains to restrict to")
    p.add_argument("--domains-exclude", default=None,
                   help="comma-separated domains to drop")
    p.add_argument("--from-date", default=None, help="ISO yyyy-MM-dd lower bound")
    p.add_argument("--to-date", default=None, help="ISO yyyy-MM-dd upper bound")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=0,
                   help="retry this many times on HTTP 429 (honours Retry-After)")
    p.add_argument("--out", default=None, help="write the normalised JSON report here")
    p.add_argument("--raw-out", default=None, help="write the raw Lenso response here")
    return p.parse_args(argv)


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.info("face_search starting  (Lenso.ai live API, sort=%s page=%d)",
             args.sort, args.page)
    try:
        with step(log, "Loading API key"):
            api_key = get_api_key()
        with step(log, "Resolving query image"):
            image_bytes, provenance = resolve_input(
                args.image, args.from_encoding,
                image_dir=args.image_dir, crop=not args.no_crop,
            )
        request_info = {"sort_type": args.sort, "page": args.page}
        with step(log, "Calling Lenso face-search API"):
            api_response = search_faces(
                image_bytes, api_key,
                sort_type=args.sort, page=args.page,
                domains_to_include=_split_csv(args.domains_include),
                domains_to_exclude=_split_csv(args.domains_exclude),
                from_date=args.from_date, to_date=args.to_date,
                timeout=args.timeout, max_retries=args.max_retries,
            )
    except FaceSearchError as exc:
        body = getattr(exc, "body", None)
        msg = str(exc) + (f"  [response body: {body}]" if body else "")
        return fail(log, msg, exc.exit_code)

    if args.raw_out:
        Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.raw_out).write_text(json.dumps(api_response, indent=2), encoding="utf-8")
        log.info("Wrote raw Lenso response to %s", args.raw_out)

    report = build_report(api_response, provenance, request_info)
    _print_report(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Wrote ranked report to %s", args.out)

    if report["match_count"] == 0:
        log.warning("No matches found for this face  (exit %d)", EXIT_ZERO_MATCHES)
        return EXIT_ZERO_MATCHES
    log.info("face_search done: %d match(es)  (exit 0)", report["match_count"])
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
