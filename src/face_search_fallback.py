"""Piece 2 (fallback) - reverse image search via Bing Images visual search.

Lenso.ai (src/face_search.py) needs a paid Developer Subscription (their tier
starts at USD 2,400+/month) - out of scope for this demo. This is the no-API
fallback: it drives a REAL Chromium browser with Playwright, uploads the photo
to Bing Images' "search by image" (visual search), opens the "Pages with this
image" tab, and scrapes the URLs of the web pages that use it.

Bing's visual search now renders results in-page under tabs ("Overview /
Visual Matches / Pages with this image / Solve") - it no longer navigates to
/images/search. We wait for that UI, click "Pages with this image" (pages
using the *exact* photo - a strong signal) rather than settling for "Visual
Matches" (visually-similar look-alikes - weak), and log which one we returned.

There is NO mock, fixture, sample, cached HTML or hardcoded result anywhere in
this file. Every run launches a browser and performs a live search; if Bing's
DOM has moved, a CAPTCHA appears, or nothing matches, it says so and exits
non-zero rather than inventing data.

Bing changes its markup often. When a selector goes missing the script writes a
screenshot + the live HTML to output/ so you can see what it hit; --headed lets
you watch the run and re-find selectors visually. See the README section
"Piece 2 (fallback)".

Handled failure modes:
  * Playwright / Chromium not installed   -> exit 3
  * bad image / bad encoding record       -> exit 4
  * Bing UI / selectors not found         -> exit 5  (+ debug dump)
  * CAPTCHA / "unusual traffic" block      -> exit 6  (+ debug dump)
  * navigation / browser timeout           -> exit 7
  * zero matching pages                    -> exit 2  (not an error, just empty)

Usage:
    python src/face_search_fallback.py --image data/input/me.jpg
    python src/face_search_fallback.py --image data/input/me.jpg --headed --keep-open
    python src/face_search_fallback.py --from-encoding output/me.json --crop --out output/me.bing.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
from pathlib import Path

from pipeline_log import fail, get_logger, step

log = get_logger("face_search_fallback")

BING_IMAGES_URL = "https://www.bing.com/images"
ENGINE = "bing-images-visual-search"

# hosts that are Bing / Microsoft chrome, never a genuine "matching page"
_OWN_HOSTS = (
    "bing.com", "microsoft.com", "msn.com", "live.com", "microsoftonline.com",
    "windows.net", "office.com", "microsofttranslator.com", "msedge.net",
    "go.microsoft.com",
)

_CAPTCHA_MARKERS = (
    "verify you are human", "verify you're human", "verify you are a human",
    "unusual traffic", "are you a robot", "prove you're not a robot",
    "to continue, please verify", "solve this puzzle", "captcha",
    "help us keep your account secure",
)

# the results tab that lists pages using the *exact* uploaded image (strong
# match), as opposed to "Visual Matches" (visually-similar look-alikes, weak).
_PAGES_TAB_RE = re.compile(
    r"pages\s+(with\s+this\s+image|that\s+include|including)", re.I
)
# Bing tags the "Pages with this image" results SERP with vsa=3 in the URL
# (vsa=2 is "Visual Matches"). Used to confirm the tab click actually landed.
_PAGES_TAB_URL_RE = re.compile(r"[?&]vsa=3(?:&|$)")

# On the vsa=3 SERP the matching pages render as normal Bing web results; the
# "no pages" state renders an #error-title card. Wait for either.
_SERP_RESULT_SELECTOR = (
    "#b_results li.b_algo a[href^='http'], #b_results h2 a[href^='http'], "
    ".b_algo a[href^='http'], #error-title, .search-error-container"
)

EXIT_OK = 0
EXIT_ZERO_MATCHES = 2
EXIT_NO_BROWSER = 3
EXIT_BAD_INPUT = 4
EXIT_SCRAPE = 5
EXIT_CAPTCHA = 6
EXIT_TIMEOUT = 7


class FallbackError(Exception):
    exit_code = 1


class NoBrowser(FallbackError):
    exit_code = EXIT_NO_BROWSER


class BadInput(FallbackError):
    exit_code = EXIT_BAD_INPUT


class ScrapeError(FallbackError):
    exit_code = EXIT_SCRAPE

    def __init__(self, message: str, debug_dump: str | None = None):
        super().__init__(message)
        self.debug_dump = debug_dump


class Captcha(FallbackError):
    exit_code = EXIT_CAPTCHA

    def __init__(self, message: str, debug_dump: str | None = None):
        super().__init__(message)
        self.debug_dump = debug_dump


class NavTimeout(FallbackError):
    exit_code = EXIT_TIMEOUT


# --------------------------------------------------------------------------
# image input  (mirrors src/face_search.py, but yields a path to upload)
# --------------------------------------------------------------------------
def _check_image_file(p: Path) -> Path:
    if not p.is_file():
        raise BadInput(f"image not found: {p}")
    if p.stat().st_size == 0:
        raise BadInput(f"image is empty: {p}")
    return p


def _crop_to_box(src: Path, box: dict, pad_frac: float = 0.35) -> Path:
    """Crop to a piece-1 face_box (top/right/bottom/left) with padding -> temp jpg."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dep
        raise BadInput("cropping needs Pillow (pip install Pillow); drop --crop") from exc

    img = Image.open(src).convert("RGB")
    w, h = img.size
    bw = box["right"] - box["left"]
    bh = box["bottom"] - box["top"]
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    crop = img.crop((
        max(0, box["left"] - px),
        max(0, box["top"] - py),
        min(w, box["right"] + px),
        min(h, box["bottom"] + py),
    ))
    fd, tmp = tempfile.mkstemp(prefix="facecrop_", suffix=".jpg")
    os.close(fd)
    crop.save(tmp, format="JPEG", quality=95)
    return Path(tmp)


def resolve_input(
    image: str | None,
    from_encoding: str | None,
    image_dir: str | Path = "data/input",
    crop: bool = False,
):
    """Return (upload_path, provenance, cleanup) from either --image or --from-encoding.

    cleanup() removes any temporary crop file; it is a no-op otherwise.
    """
    if bool(image) == bool(from_encoding):
        raise BadInput("give exactly one of --image / --from-encoding")

    if image:
        path = _check_image_file(Path(image))
        log.info("Query image: %s (%.0f KB)", path, path.stat().st_size / 1024)
        return path, {"source": str(path), "cropped": False, "via": "image"}, (lambda: None)

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
            f"cannot find source image '{src_name}' from {rec_path}; looked in: "
            + ", ".join(str(c) for c in candidates)
            + ". Pass --image-dir or --image."
        )
    _check_image_file(img_path)

    log.info("Found source image: %s (%.0f KB)", img_path, img_path.stat().st_size / 1024)

    want_sha = meta.get("source_sha256")
    if want_sha:
        got = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if got != want_sha:
            raise BadInput(
                f"image {img_path} sha256 {got[:12]}.. does not match the encoding "
                f"record ({want_sha[:12]}..) - wrong or edited file"
            )
        log.info("SHA-256 matches the encoding record (%s..)", got[:16])

    prov: dict = {"source": str(img_path), "cropped": False, "via": "from_encoding",
                  "encoding_record": str(rec_path)}

    box = record.get("face_box")
    if crop and isinstance(box, dict) and {"top", "right", "bottom", "left"} <= box.keys():
        tmp = _crop_to_box(img_path, box)
        prov["cropped"] = True
        prov["face_index"] = record.get("face_index")
        log.info("Cropped to piece-1 face box #%s -> %s (%.0f KB)",
                 record.get("face_index"), tmp, tmp.stat().st_size / 1024)
        return tmp, prov, (lambda: tmp.unlink(missing_ok=True))

    return img_path, prov, (lambda: None)


# --------------------------------------------------------------------------
# pure result-parsing helpers  (no browser, no network - unit tested)
# --------------------------------------------------------------------------
def is_offsite(url: str) -> bool:
    """True if url is an http(s) link to a host that is not Bing/Microsoft."""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return not any(host == h or host.endswith("." + h) for h in _OWN_HOSTS)


def extract_page_urls(anchors: list[dict]) -> list[dict]:
    """anchors: [{'href': ..., 'text': ...}] scraped from the results DOM.

    Keep off-site http(s) links, drop Bing's own image-search URLs, dedupe on
    host+path (order preserved), and rank.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        if not is_offsite(href):
            continue
        parsed = urllib.parse.urlparse(href)
        if "/images/search" in parsed.path:
            continue
        key = (parsed.hostname or "").lower() + parsed.path.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        title = " ".join((a.get("text") or "").split())[:200]
        out.append({"source_url": href, "title": title or None})

    for rank, m in enumerate(out, start=1):
        m["rank"] = rank
    return out


def build_report(matches: list[dict], provenance: dict, request_info: dict) -> dict:
    return {
        "query": {**provenance, **request_info, "engine": ENGINE, "search_url": BING_IMAGES_URL},
        "match_count": len(matches),
        "matches": matches,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# browser automation
# --------------------------------------------------------------------------
def _dump_debug(page, out_dir: str) -> str | None:
    """Save a screenshot + the live HTML so a broken selector can be re-found."""
    try:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        png = d / f"bing_debug_{stamp}.png"
        html = d / f"bing_debug_{stamp}.html"
        page.screenshot(path=str(png), full_page=True)
        html.write_text(page.content(), encoding="utf-8")
        log.info("Saved debug screenshot + HTML: %s , %s", png, html)
        return f"{png} , {html}"
    except Exception as exc:
        log.warning("Could not write debug dump: %s: %s", type(exc).__name__, exc)
        return None


def _looks_like_captcha(page) -> bool:
    url = (page.url or "").lower()
    if any(k in url for k in ("/captcha", "challenge", "/block", "sacaptcha")):
        log.warning("CAPTCHA / block URL: %s", page.url)
        return True
    try:
        body = (page.locator("body").inner_text(timeout=3000) or "").lower()
    except Exception:
        return False
    hit = next((m for m in _CAPTCHA_MARKERS if m in body), None)
    if hit:
        log.warning("CAPTCHA / verification marker on page: %r (url=%s)", hit, page.url)
        return True
    return False


def _dismiss_consent(page) -> None:
    """Best-effort click of the EU cookie / consent banner if Bing shows one."""
    for sel in (
        "#bnp_btn_accept", "button#bnp_btn_accept",
        "button:has-text('Accept all')", "button:has-text('Accept')",
        "button:has-text('I agree')",
    ):
        try:
            btn = page.locator(sel).first
            if btn.count() and btn.is_visible():
                btn.click(timeout=2000)
                page.wait_for_timeout(400)
                log.info("Dismissed a cookie/consent banner (%s)", sel)
                return
        except Exception:
            continue


def _find_file_input(page):
    """Locate Bing's image-upload <input type=file>, revealing it if needed.

    Returns a Locator, or None if the control cannot be found (Bing changed its
    markup - update the selector lists below).
    """
    direct = ("input#sb_fileinput", "input[type='file']")
    for sel in direct:
        try:
            loc = page.locator(sel).first
            if loc.count():
                log.info("Found upload input directly (%s)", sel)
                return loc
        except Exception:
            pass

    # reveal it by clicking the visual-search / camera control
    triggers = (
        "#sb_sbip", "#sbi_b", ".camera", "#vsbtn",
        "[aria-label*='visual search' i]",
        "[aria-label*='search using an image' i]",
        "[aria-label*='search by image' i]",
        "[title*='visual search' i]",
    )
    for sel in triggers:
        try:
            btn = page.locator(sel).first
            if not btn.count():
                continue
            btn.click(timeout=3000)
        except Exception:
            continue
        page.wait_for_timeout(600)
        for sel2 in direct:
            try:
                loc = page.locator(sel2).first
                if loc.count():
                    log.info("Revealed upload input via %s, found %s", sel, sel2)
                    return loc
            except Exception:
                pass
    return None


def _collect_result_anchors(page) -> dict:
    """Scrape candidate result links from the "Pages with this image" view.

    That view is a normal Bing SERP: results are `#b_results > li.b_algo`, and an
    explicit "Unable to find pages with this image" card means a definitive zero.
    Falls back to heading-named sections / known containers / the whole document
    if the SERP markup isn't found. Returns {'mode', 'links', 'error'}.
    """
    js = r"""() => {
      const grab = root => [...root.querySelectorAll("a[href]")].map(a => ({
        href: a.href,
        text: (a.innerText || a.getAttribute('aria-label') || a.title || '').trim(),
      }));
      const httpOnly = xs => xs.filter(x => x.href.startsWith('http'));

      // explicit "no pages" card that Bing renders on the vsa=3 SERP
      const errEl = document.querySelector(
        '#error-title, .search-error-container .error-title');
      const error = errEl ? (errEl.innerText || '').trim() : '';

      // 1. the "Pages with this image" SERP: matching pages are normal Bing
      //    web results (li.b_algo), one link per result.
      const algo = [...document.querySelectorAll(
        '#b_results li.b_algo, #b_results .b_algo, ol#b_results > li, .b_algo')];
      if (algo.length) {
        const links = [];
        for (const li of algo) {
          const a = li.querySelector('h2 a[href], a.tilk[href], .b_algoheader a[href]')
                    || li.querySelector('a[href]');
          if (a && a.href.startsWith('http')) {
            links.push({href: a.href, text: (a.innerText || '').trim()});
          }
        }
        if (links.length) return {mode: 'serp:b_algo', links, error};
      }
      if (error) return {mode: 'serp:error-card', links: [], error};

      // 1b. same SERP, less specific: any off-site link in the results column
      for (const sel of ['#b_results', '#b_content', 'main']) {
        const el = document.querySelector(sel);
        if (el) {
          const links = httpOnly(grab(el));
          if (links.length) return {mode: 'serp:' + sel, links, error};
        }
      }

      // 2. a section whose heading names matching / including / "this image" pages
      const heads = [...document.querySelectorAll(
        "h1,h2,h3,h4,[role=heading],.tab-head,.iuscp_hd,.vsi_head")];
      for (const h of heads) {
        const t = (h.innerText || '').toLowerCase();
        if (t.includes('page') &&
            (t.includes('match') || t.includes('includ') || t.includes('this'))) {
          let sec = h.closest('section,div');
          for (let i = 0; i < 4 && sec; i++) {
            const links = httpOnly(grab(sec));
            if (links.length > 2) return {mode: 'section:' + t.slice(0, 40), links, error};
            sec = sec.parentElement;
          }
        }
      }

      // 3. known result containers
      for (const sel of ['.pageResults', '.insights', '#insights', '.richImgArea',
                         '[class*=insights]', '#vsi_results']) {
        const el = document.querySelector(sel);
        if (el) {
          const links = httpOnly(grab(el));
          if (links.length) return {mode: sel, links, error};
        }
      }

      // 4. whole document (weak - lots of Bing chrome will be filtered out later)
      return {mode: 'document', links: httpOnly(grab(document)), error};
    }"""
    data = page.evaluate(js)
    if not isinstance(data, dict):
        log.warning("Result scrape returned no data; treating as empty")
        return {"mode": "document", "links": [], "error": ""}
    data.setdefault("error", "")
    log.info("Scraped %d raw link(s) from the results page (locator: %s)%s",
             len(data.get("links", [])), data.get("mode"),
             f'; Bing card: "{data["error"]}"' if data.get("error") else "")
    return data


def _settle(page, ms: int = 6000) -> None:
    """Short 'network idle' wait. Bing keeps telemetry sockets open, so a
    full-timeout networkidle wait usually just burns the whole budget."""
    try:
        page.wait_for_load_state("networkidle", timeout=ms)
    except Exception:
        pass


def _select_pages_tab(page, timeout_ms: int) -> tuple[str | None, bool]:
    """Wait for and click Bing's "Pages with this image" tab.

    That tab lists pages using the *exact* uploaded photo - the strong signal we
    want (an actual post), not "Visual Matches" / look-alike strangers.

    The tab is an <a> that navigates to a `?...&vsa=3` SERP; we click it and
    wait for that URL (or, failing that, network idle) before returning.

    Returns (tab_label, is_strong):
      * (label, True)  - clicked the tab and landed on the vsa=3 results SERP
      * (label, False) - the tab was found but the click did not land there;
                         the caller scrapes whatever is shown (weak)
      * (None,  False) - no such tab at all (Bing changed its markup)
    """
    tab = (
        page.get_by_role("tab", name=_PAGES_TAB_RE)
        .or_(page.get_by_role("link", name=_PAGES_TAB_RE))
        .or_(page.get_by_role("button", name=_PAGES_TAB_RE))
        .or_(page.get_by_text(_PAGES_TAB_RE))
        .first
    )
    try:
        tab.wait_for(state="visible", timeout=timeout_ms)
    except Exception:
        log.warning("No 'Pages with this image' tab appeared (regex %s)",
                    _PAGES_TAB_RE.pattern)
        return None, False

    try:
        label = " ".join((tab.inner_text(timeout=2000) or "").split())
    except Exception:
        label = ""
    label = label or "Pages with this image"
    log.info("Found the %r tab", label)

    if _PAGES_TAB_URL_RE.search(page.url or ""):
        log.info("Already on the vsa=3 results SERP")
        return label, True

    clicked = False
    for force in (False, True):
        try:
            tab.click(timeout=5000, force=force)
            clicked = True
            break
        except Exception:
            continue
    if not clicked:
        log.warning("Found the %r tab but could not click it", label)
        return label, False

    landed = False
    try:
        page.wait_for_url(_PAGES_TAB_URL_RE, timeout=timeout_ms)
        landed = True
    except Exception:
        landed = bool(_PAGES_TAB_URL_RE.search(page.url or ""))
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    _settle(page)
    log.info("Clicked %r; %s (url=%s)", label,
             "landed on the vsa=3 results SERP" if landed
             else "did NOT reach the vsa=3 SERP", page.url)
    return label, landed


def _wait_for_serp_results(page, timeout_ms: int) -> None:
    """Wait for the vsa=3 SERP to render its results (or its 'no pages' card),
    nudging a lazily-loaded list with a scroll or two. Best-effort - returns
    even if nothing appeared (the caller then decides via _collect_result_anchors).
    """
    slice_ms = min(max(3000, timeout_ms // 3), 12000)
    for attempt in range(4):
        try:
            page.wait_for_selector(_SERP_RESULT_SELECTOR, timeout=slice_ms)
            log.info("Results SERP rendered")
            return
        except Exception:
            pass
        try:
            if attempt == 2:
                # Bing sometimes serves an empty SERP shell to a fresh headless
                # navigation; a reload often fills it in.
                log.info("Results not visible yet - reloading the SERP once")
                page.reload(wait_until="domcontentloaded")
                _settle(page)
            else:
                page.mouse.wheel(0, 2400)
                page.wait_for_timeout(700)
        except Exception:
            pass
    log.warning("Results SERP did not clearly render after 4 tries - scraping anyway")


def bing_visual_search(
    image_path: Path,
    *,
    headed: bool = False,
    slow_mo: int = 0,
    timeout: float = 45.0,
    keep_open: bool = False,
    debug_dir: str = "output",
) -> tuple[list[dict], str | None, dict]:
    """Run a live Bing Images visual search for image_path.

    Returns (matches, debug_dump, meta):
      * debug_dump is set only when the results page loaded but yielded zero
        off-site links (so you can check it was really empty, not a scrape miss).
      * meta = {"result_source": <human string>, "match_strength": "strong"|"weak"}
        - "strong" means the links came from the "Pages with this image" tab
        (pages using the exact photo); "weak" means we fell back to whatever
        Bing showed (visually-similar / look-alike results).
    """
    try:
        from playwright.sync_api import Error as PWError
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise NoBrowser(
            "Playwright is not installed - run `pip install playwright` then "
            "`playwright install chromium`"
        ) from exc

    timeout_ms = int(timeout * 1000)
    log.info("Bing visual search: image=%s  mode=%s  per-step timeout=%.0fs",
             image_path, "HEADED" if headed else "headless", timeout)

    with sync_playwright() as pw:
        with step(log, "Launching Chromium (%s)" % ("headed" if headed else "headless")):
            try:
                browser = pw.chromium.launch(
                    headless=not headed,
                    slow_mo=slow_mo,
                    # Bing serves a stripped-down SERP to obvious automation; these
                    # blunt the most common headless tells. Real results are still
                    # far more reliable with --headed.
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except PWError as exc:
                raise NoBrowser(
                    f"could not launch Chromium ({exc}). Run `playwright install chromium`."
                ) from exc

        ctx = browser.new_context(
            locale="en-US",
            timezone_id="Asia/Kolkata",
            viewport={"width": 1366, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            with step(log, f"Opening {BING_IMAGES_URL}"):
                try:
                    page.goto(BING_IMAGES_URL, wait_until="domcontentloaded",
                              timeout=timeout_ms)
                except PWTimeout as exc:
                    raise NavTimeout(
                        f"Bing Images did not load within {timeout}s ({BING_IMAGES_URL}) - "
                        "check your connection or raise --timeout"
                    ) from exc

            _dismiss_consent(page)
            if _looks_like_captcha(page):
                raise Captcha(
                    "Bing served a CAPTCHA / verification page before the search - you are "
                    "rate-limited or flagged. Try later, a different network, or --headed and "
                    "solve it by hand.",
                    debug_dump=_dump_debug(page, debug_dir),
                )

            with step(log, "Finding Bing's image-upload control"):
                file_input = _find_file_input(page)
                if file_input is None:
                    raise ScrapeError(
                        "could not find Bing's image-upload <input> (the camera / \"search "
                        "by image\" control). Bing's markup has likely changed - re-run with "
                        "--headed to inspect it and update _find_file_input().",
                        debug_dump=_dump_debug(page, debug_dir),
                    )

            with step(log, f"Uploading {image_path.name} to Bing visual search"):
                try:
                    file_input.set_input_files(str(image_path), timeout=timeout_ms)
                except PWError as exc:
                    raise ScrapeError(
                        f"failed to hand the image to Bing's upload input: {exc}",
                        debug_dump=_dump_debug(page, debug_dir),
                    ) from exc
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                except PWTimeout:
                    pass

            # Bing's visual search now renders results in-page under tabs
            # ("Overview / Visual Matches / Pages with this image / Solve") and
            # no longer navigates to /images/search. Open the "Pages with this
            # image" tab (pages using the *exact* photo - the strong signal),
            # not "Visual Matches" (look-alike strangers).
            with step(log, "Opening the 'Pages with this image' tab"):
                tab_label, is_strong = _select_pages_tab(page, timeout_ms)
                if is_strong:
                    _wait_for_serp_results(page, timeout_ms)
                _settle(page)

            if _looks_like_captcha(page):
                raise Captcha(
                    "Bing served a CAPTCHA after the image upload",
                    debug_dump=_dump_debug(page, debug_dir),
                )

            with step(log, "Scraping result links"):
                scraped = _collect_result_anchors(page)
                matches = extract_page_urls(scraped.get("links", []))
                log.info("Kept %d off-site page link(s) after filtering Bing/dupes "
                         "(from %d raw)", len(matches), len(scraped.get("links", [])))
            bing_zero_msg = (scraped.get("error") or "").strip()
            on_pages_serp = str(scraped.get("mode", "")).startswith("serp:")

            # real breakage: no tab, no results, and no explicit "zero" card
            if tab_label is None and not matches and not bing_zero_msg:
                raise ScrapeError(
                    "uploaded the image but found neither the 'Pages with this image' tab "
                    "nor any results. Bing's visual-search UI has likely changed - re-run "
                    "with --headed and update _select_pages_tab() / _collect_result_anchors().",
                    debug_dump=_dump_debug(page, debug_dir),
                )

            if tab_label is not None and (is_strong or on_pages_serp):
                is_strong = True
                result_source = f"the {tab_label!r} tab"
            elif tab_label is not None:
                result_source = (f"default view ({tab_label!r} tab found but the click "
                                 "did not land on the results SERP)")
            else:
                result_source = "default view (no 'Pages with this image' tab found)"
                is_strong = False

            match_strength = "strong" if is_strong else "weak"
            if is_strong:
                log.info("Returning %d page(s) from %s  ->  STRONG match "
                         "(pages that use this exact image)", len(matches), result_source)
            else:
                log.warning("Returning %d page(s) from %s  ->  WEAK match "
                            "(visually-similar / look-alike results, not this exact photo)",
                            len(matches), result_source)
            if bing_zero_msg:
                log.info("Bing card on the page: %r", bing_zero_msg)

            if keep_open:
                log.info("--keep-open: leaving the browser open; press Enter here to close it")
                try:
                    input()
                except EOFError:
                    page.wait_for_timeout(15000)

            # dump the page only when we got nothing AND Bing didn't explicitly
            # say "no pages" (i.e. it might be a scrape miss worth inspecting)
            debug_dump = (_dump_debug(page, debug_dir)
                          if (not matches and not bing_zero_msg) else None)
            meta = {
                "result_source": result_source,
                "match_strength": match_strength,
                "bing_note": bing_zero_msg or None,
            }
            return matches, debug_dump, meta
        finally:
            ctx.close()
            browser.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_report(report: dict) -> None:
    q = report["query"]
    print(f"query image : {q.get('source')}"
          + ("  (cropped to piece-1 face box)" if q.get("cropped") else ""))
    print(f"engine      : Bing Images visual search  ({q['search_url']})")
    strength = q.get("match_strength")
    if strength:
        note = ("pages using this exact image" if strength == "strong"
                else "visually-similar / look-alike results only")
        print(f"match type  : {strength.upper()}  ({note})")
        print(f"scraped from: {q.get('result_source')}")
    print(f"results     : {report['match_count']} page(s)")
    if q.get("bing_note"):
        print(f"bing note   : {q['bing_note']}")
    print()

    for m in report["matches"]:
        print(f"  #{m['rank']:<2} {m['source_url']}")
        if m.get("title"):
            print(f"       {m['title']}")

    if not report["matches"]:
        return

    top = report["matches"][0]
    print()
    print("=" * 60)
    print("TOP MATCH")
    print("=" * 60)
    print(f"  page  : {top['source_url']}")
    print(f"  title : {top.get('title') or '(none)'}")
    print("=" * 60)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="face_search_fallback",
        description="Reverse image search via Bing Images visual search, driven by a real "
                    "Playwright/Chromium browser (piece 2 fallback for when Lenso's paid tier "
                    "is out of reach).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to the query image")
    src.add_argument("--from-encoding",
                     help="path to a piece-1 encoding JSON (locates + sha256-checks the photo)")

    p.add_argument("--image-dir", default="data/input",
                   help="where to look for the photo named in the encoding record")
    p.add_argument("--crop", action="store_true",
                   help="with --from-encoding, upload a crop of the recorded face box (needs Pillow)")
    p.add_argument("--headed", action="store_true",
                   help="show the browser window so you can watch the run / debug selectors")
    p.add_argument("--keep-open", action="store_true",
                   help="after scraping, keep the browser open until you press Enter")
    p.add_argument("--slow-mo", type=int, default=0, metavar="MS",
                   help="delay each browser action by MS milliseconds (use with --headed)")
    p.add_argument("--timeout", type=float, default=45.0,
                   help="per-step navigation timeout in seconds (default 45)")
    p.add_argument("--debug-dir", default="output",
                   help="where screenshot + HTML dumps land on failure (default: output/)")
    p.add_argument("--out", default=None, help="write the JSON report here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.info("face_search_fallback starting  (Bing visual search via Playwright, %s)",
             "headed" if args.headed else "headless")
    try:
        with step(log, "Resolving query image"):
            upload_path, provenance, cleanup = resolve_input(
                args.image, args.from_encoding, image_dir=args.image_dir, crop=args.crop,
            )
        try:
            matches, debug_dump, meta = bing_visual_search(
                upload_path,
                headed=args.headed,
                slow_mo=args.slow_mo,
                timeout=args.timeout,
                keep_open=args.keep_open,
                debug_dir=args.debug_dir,
            )
        finally:
            cleanup()
    except FallbackError as exc:
        msg = str(exc)
        dump = getattr(exc, "debug_dump", None)
        if dump:
            msg += f"  [debug dump: {dump}]"
        return fail(log, msg, exc.exit_code)

    report = build_report(matches, provenance, {"headed": args.headed, **meta})
    _print_report(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Wrote report to %s", args.out)

    if report["match_count"] == 0:
        if meta.get("bing_note"):
            log.warning('Bing found no pages with this image ("%s")  (exit %d)',
                        meta["bing_note"], EXIT_ZERO_MATCHES)
        else:
            log.warning("No pages with matching images found  (exit %d)", EXIT_ZERO_MATCHES)
            if not args.headed:
                log.warning("Re-run with --headed - Bing serves degraded results to "
                            "headless browsers.")
        if debug_dump:
            log.info("Saved the page for inspection: %s", debug_dump)
        return EXIT_ZERO_MATCHES

    log.info("face_search_fallback done: %d page(s), %s match  (exit 0)",
             report["match_count"], meta.get("match_strength", "?"))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
