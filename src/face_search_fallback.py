"""Piece 2 (fallback) - reverse image search via Bing Images visual search.

Lenso.ai (src/face_search.py) needs a paid Developer Subscription (their tier
starts at USD 2,400+/month) - out of scope for this demo. This is the no-API
fallback: it drives a REAL Chromium browser with Playwright, uploads the photo
to Bing Images' "search by image" (visual search), opens the "Pages with this
image" tab, and scrapes the URLs of the web pages that use it.

Two tiers of candidate, each labelled honestly:
  * match_type "exact_page"        - from the "Pages with this image" tab
  * match_type "visual_similarity" - from the "Visual Matches" tab, or (when
                                     Bing shows neither tab) its default
                                     post-upload results

meta["match_strength"] reflects whichever tier found results:
  "high"     - >= 1 exact_page candidate
  "moderate" - 0 exact_page, >= 1 visual_similarity candidate
  "none"     - both tiers empty

All of Bing's outbound links are `https://www.bing.com/ck/a?...&u=a1<base64>`
redirect wrappers; `unwrap_bing_redirect()` resolves them to the real
destination before filtering, otherwise every candidate looks like a bing.com
link and gets dropped.

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
import base64
import binascii
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

# "Visual Matches" tab - the fallback when "Pages with this image" has nothing.
# It lists visually-similar images (look-alikes), so it is always a WEAK match.
_VISUAL_MATCHES_RE = re.compile(r"visual\s+matches", re.I)
_VISUAL_MATCHES_URL_RE = re.compile(r"[?&]vsa=2(?:&|$)")

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


def unwrap_bing_redirect(url: str) -> str:
    """Bing wraps every outbound result link as
    `https://www.bing.com/ck/a?...&u=a1<base64url>` - which our off-site filter
    would otherwise throw away as "a bing.com link". Return the real destination
    (or the url unchanged if it isn't a wrapper).
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return url
    if "bing.com" not in (parsed.hostname or "") or not parsed.path.startswith("/ck/"):
        return url
    u = urllib.parse.parse_qs(parsed.query).get("u", [""])[0]
    if not u:
        return url
    if u[:2] in ("a1", "a2", "a3"):  # Bing's format marker
        u = u[2:]
    u += "=" * (-len(u) % 4)
    try:
        decoded = base64.urlsafe_b64decode(u).decode("utf-8", "replace")
    except (binascii.Error, ValueError):
        return url
    return decoded if decoded.startswith(("http://", "https://")) else url


def extract_page_urls(anchors: list[dict]) -> list[dict]:
    """anchors: [{'href': ..., 'text': ...}] scraped from the results DOM.

    Unwrap Bing's `/ck/a` redirect links, keep the off-site http(s) ones, drop
    Bing's own image-search URLs, dedupe on host+path (order preserved), rank.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for a in anchors:
        href = unwrap_bing_redirect((a.get("href") or "").strip())
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
        title = " ".join((a.get("text") or "").split())
        # Bing's `a.tilk` anchors read "Site Name https://site.com > a > b" -
        # keep just the readable name.
        title = re.split(r"\s+https?://", title, maxsplit=1)[0][:200]
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
    """Scrape the "Pages with this image" (vsa=3) SERP - carefully.

    When Bing has NO exact-image match it (a) renders an
    "Unable to find pages with this image" card AND (b) fills #b_results with
    ordinary web results for the text query it derived from the photo. Those
    web results are NOT image matches - scraping them and calling them
    "exact_page" is a false positive. So:

      mode 'serp:no-exact-matches'  - the "unable to find" card is present.
                                      links = []  (the #b_results list is a
                                      text-query fallback, deliberately ignored).
      mode 'serp:matching-section'  - a section whose heading actually names
                                      matching / including pages. links = that
                                      section's anchors  (a real exact_page set).
      mode 'serp:vsa3-weblist'      - on a vsa=3 URL, #b_results has b_algo rows,
                                      but no "unable to find" card and no
                                      labelled matching section. AMBIGUOUS - the
                                      caller must NOT treat these as exact_page
                                      without corroboration.
      mode 'serp:generic'/'document'- b_algo / anchors, not on a vsa=3 URL.

    Returns {'mode', 'links', 'error'}.
    """
    js = r"""() => {
      const T = s => (s || '').replace(/\s+/g, ' ').trim();
      const grab = root => [...root.querySelectorAll('a[href]')]
        .map(a => ({href: a.href, text: T(a.getAttribute('aria-label') || a.innerText || a.title)}))
        .filter(x => x.href.startsWith('http'));

      // (1) the "no exact matches" card WINS. Everything in #b_results under it
      //     is Bing's derived-text-query web list, not image matches.
      const errEl = document.querySelector('#error-title, .search-error-container .error-title');
      const errText = errEl ? T(errEl.innerText) : '';
      const noMatch = /unable to find pages with this image|no matching pages|couldn.?t find pages/i;
      if (noMatch.test(errText) || noMatch.test(document.body.innerText.slice(0, 20000))) {
        return {mode: 'serp:no-exact-matches', links: [],
                error: errText || 'Unable to find pages with this image'};
      }

      // (2) a section whose heading genuinely names matching / including pages
      const heads = [...document.querySelectorAll('h1,h2,h3,h4,[role=heading],.b_focusLabel,.iuscp_hd')];
      for (const h of heads) {
        const t = T(h.innerText).toLowerCase();
        const names_matches = t.includes('page') &&
          (t.includes('matching image') || t.includes('include this image') ||
           t.includes('that include') || t.includes('with this image'));
        if (!names_matches) continue;
        let sec = h.closest('section,ol,ul,div');
        for (let i = 0; i < 3 && sec; i++) {
          const links = grab(sec);
          if (links.length) return {mode: 'serp:matching-section', links, error: ''};
          sec = sec.parentElement;
        }
      }

      // (3) the #b_results web list
      const onVsa3 = /[?&]vsa=3(&|$)/.test(location.href);
      const algo = [...document.querySelectorAll('#b_results li.b_algo, #b_results .b_algo')];
      if (algo.length) {
        const links = [];
        for (const li of algo) {
          const a = li.querySelector('h2 a[href^="http"], a.tilk[href^="http"], '
                                     + '.b_algoheader a[href^="http"]')
                    || li.querySelector('a[href^="http"]');
          if (a) links.push({href: a.href, text: T(a.getAttribute('aria-label') || a.innerText)});
        }
        if (links.length)
          return {mode: onVsa3 ? 'serp:vsa3-weblist' : 'serp:generic', links, error: ''};
      }

      // (4) nothing structured
      return {mode: 'document', links: grab(document), error: ''};
    }"""
    data = page.evaluate(js)
    if not isinstance(data, dict):
        log.warning("Result scrape returned no data; treating as empty")
        return {"mode": "document", "links": [], "error": ""}
    data.setdefault("error", "")
    log.info("Scraped %d raw link(s) (mode=%s)%s",
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


def _open_result_tab(page, name_re, url_re, timeout_ms: int, what: str) -> tuple[str | None, bool]:
    """Find + click one of Bing's visual-search result tabs and wait for its
    SERP URL (`?...&vsa=N`).

    Returns (tab_label, landed):
      * (label, True)  - clicked and landed on that tab's SERP
      * (label, False) - tab found but the click didn't reach its SERP
      * (None,  False) - no such tab (Bing changed its markup)
    """
    tab = (
        page.get_by_role("tab", name=name_re)
        .or_(page.get_by_role("link", name=name_re))
        .or_(page.get_by_role("button", name=name_re))
        .or_(page.get_by_text(name_re))
        .first
    )
    # The tab bar, when Bing shows it, renders within a few seconds - don't burn
    # the whole per-step budget waiting for a tab that isn't there.
    try:
        tab.wait_for(state="visible", timeout=min(timeout_ms, 12000))
    except Exception:
        log.warning("No %r tab appeared (regex %s) - Bing didn't render its "
                    "visual-search tab bar for this image", what, name_re.pattern)
        return None, False

    try:
        label = " ".join((tab.inner_text(timeout=2000) or "").split())
    except Exception:
        label = ""
    label = label or what
    log.info("Found the %r tab", label)

    if url_re.search(page.url or ""):
        log.info("Already on the %s SERP", what)
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
        page.wait_for_url(url_re, timeout=timeout_ms)
        landed = True
    except Exception:
        landed = bool(url_re.search(page.url or ""))
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    _settle(page)
    log.info("Clicked %r; %s (url=%s)", label,
             f"landed on the {what} SERP" if landed else f"did NOT reach the {what} SERP",
             page.url)
    return label, landed


def _select_pages_tab(page, timeout_ms: int) -> tuple[str | None, bool]:
    """Open the "Pages with this image" tab (pages using the *exact* photo -
    the strong signal), not "Visual Matches" / look-alike strangers."""
    return _open_result_tab(page, _PAGES_TAB_RE, _PAGES_TAB_URL_RE, timeout_ms,
                            "Pages with this image")


def _collect_visual_matches(page) -> dict:
    """Scrape source-page links off Bing's visually-similar results.

    Covers three DOM shapes Bing has shipped for this:
      * old image grid  - `a.iusc[m]` whose `m` JSON `purl` is the source page
      * newer "similar images" panel - `.richImgLnk` / `.iuscp a`
      * the plain SERP Bing now returns for many photos - each web result's
        source-site link (`a.tilk`) and title link (`#b_results li.b_algo h2 a`)

    Every href is passed through `unwrap_bing_redirect` later, so the
    `https://www.bing.com/ck/a?...` wrappers Bing puts on all outbound links are
    resolved to the real destination. Returns {'mode', 'links'}.
    """
    js = r"""() => {
      const seen = new Set();
      const push = (arr, href, text) => {
        if (href && !seen.has(href)) { seen.add(href); arr.push({href, text: (text||'').trim()}); }
      };

      // 1. image cards with a JSON `m` blob (purl = the page the image is on)
      const cards = [];
      for (const el of document.querySelectorAll('[m]')) {
        const raw = el.getAttribute('m');
        if (!raw || raw.indexOf('purl') === -1) continue;
        try {
          const j = JSON.parse(raw);
          if (j && j.purl && /^https?:/i.test(j.purl)) push(cards, j.purl, j.t || j.desc);
        } catch (e) {}
      }
      if (cards.length) return {mode: 'vm:image-cards', links: cards};

      // 2. "similar images" / insights panel anchors
      const sim = [];
      for (const a of document.querySelectorAll(
          '.richImgLnk[href], .iuscp a[href], .insights a[href^="http"], '
          + '.similarView a[href], .imgpt a[href]')) {
        push(sim, a.href, a.innerText || a.getAttribute('aria-label') || a.title);
      }
      if (sim.length) return {mode: 'vm:similar-panel', links: sim};

      // 3. the plain SERP Bing returns for many photos: per-result source link
      //    (a.tilk) + the result title link
      const serp = [];
      for (const a of document.querySelectorAll(
          '#b_results li.b_algo a.tilk[href], #b_results li.b_algo h2 a[href], '
          + '#b_results .b_algo cite ~ a[href], #b_results a.tilk[href]')) {
        push(serp, a.href, a.getAttribute('aria-label') || a.innerText || a.title);
      }
      if (serp.length) return {mode: 'vm:serp-results', links: serp};

      // 4. last resort: every http anchor in the main content column
      const main = document.querySelector('#b_content, main, body');
      const any = [];
      if (main) for (const a of main.querySelectorAll('a[href^="http"]'))
        push(any, a.href, a.innerText || a.title);
      return {mode: any.length ? 'vm:any-anchor' : 'vm:empty', links: any};
    }"""
    data = page.evaluate(js)
    if not isinstance(data, dict):
        return {"mode": "vm:empty", "links": []}
    log.info("Visual Matches scrape: %d raw link(s) (locator: %s)",
             len(data.get("links", [])), data.get("mode"))
    return data


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


MATCH_TYPE_EXACT = "exact_page"          # pages that use the EXACT uploaded image
MATCH_TYPE_VISUAL = "visual_similarity"  # visually-similar / Bing-derived results


def score_matches(exact_matches: list[dict], visual_matches: list[dict]) -> tuple[str, str]:
    """Combine the two tiers into (match_strength, result_source).

    match_strength:
      * "high"     - >= 1 exact_page match ("Pages with this image")
      * "moderate" - 0 exact_page, >= 1 visual_similarity match ("Visual
                     Matches", or Bing's derived-SERP results)
      * "none"     - both tiers empty

    Never contradicts the candidate lists: "high"/"moderate" only when that
    tier actually has links.
    """
    if exact_matches:
        return "high", (f"{len(exact_matches)} page(s) using this exact image "
                        "('Pages with this image')")
    if visual_matches:
        return "moderate", (f"{len(visual_matches)} visually-similar result(s) "
                            "('Visual Matches' / Bing's image interpretation) - "
                            "no exact-image pages found")
    return "none", "no results in either tier ('Pages with this image' or 'Visual Matches')"


def tag_matches(matches: list[dict], match_type: str) -> list[dict]:
    """Stamp each candidate with its match_type and re-rank 1..N (in place)."""
    for rank, m in enumerate(matches, start=1):
        m["match_type"] = match_type
        m["rank"] = rank
    return matches


# Domains / URL shapes that are SEO listicles / content-marketing, i.e. very
# unlikely to host a specific personal photo. If an "exact_page" set is mostly
# these, we almost certainly scraped a related-articles / text-query panel.
_LISTICLE_PATH_RE = re.compile(
    r"/\d{1,3}[-_][a-z][a-z-]*\b(best|top|creative|smart|inspiring|powerful|amazing|"
    r"stunning|essential|genius|clever|inspiration|ideas?|ways|tips|design|setup|"
    r"trends?|guide|hacks?|examples?)\b", re.I,
)
_CONTENT_MARKETING_HINTS = ("/blog/", "/blogs/", "/ideas/", "/inspiration/", "/guide",
                            "/tips", "-ideas", "-guide", "/how-to", "/trends", "-setup-",
                            "-design-", "-inspiration")


def looks_like_content_marketing(url: str) -> bool:
    """Heuristic: does this URL look like an SEO listicle / content-marketing
    article (e.g. "13-design-ideas-for-meeting-rooms") rather than a page that
    would actually host a specific user's photo? Used to catch the "we scraped
    a related-articles / text-query panel and called it exact_page" bug."""
    try:
        p = urllib.parse.urlparse(url)
    except ValueError:
        return False
    host, path = (p.hostname or "").lower(), (p.path or "").lower()
    if host in ("pinterest.com", "www.pinterest.com") and "/ideas/" in path:
        return True
    if _LISTICLE_PATH_RE.search(path):
        return True
    return sum(1 for h in _CONTENT_MARKETING_HINTS if h in path) >= 2


def content_marketing_ratio(matches: list[dict]) -> float:
    if not matches:
        return 0.0
    n = sum(1 for m in matches if looks_like_content_marketing(m.get("source_url", "")))
    return n / len(matches)


def assign_tiers(scrape_mode: str, raw_matches: list[dict], *,
                 bing_zero_msg: str) -> tuple[list[dict], list[dict], list[str]]:
    """Pure: from the scrape, decide (exact_matches, visual_candidates, notes).

    This is the guard against "scraped the wrong DOM panel and called it
    exact_page". Every branch is deliberately conservative:

      * Bing said "unable to find pages with this image"  -> exact = [] (hard
        zero; the accompanying #b_results is a text-query fallback, dropped).
      * a genuinely labelled matching-images section       -> exact = raw.
      * a vsa=3 #b_results web list with no such label      -> NOT exact; the
        links become visual_similarity candidates instead.
      * anything else                                       -> visual_similarity.
      * an "exact" set that is mostly SEO listicles         -> demoted to
        visual_similarity (known false-positive pattern).
    """
    notes: list[str] = []
    if bing_zero_msg or scrape_mode == "serp:no-exact-matches":
        return [], [], ["bing_reports_no_exact_matches"]

    if scrape_mode == "serp:matching-section" and raw_matches:
        exact = list(raw_matches)
        ratio = content_marketing_ratio(exact)
        if ratio >= 0.6 and len(exact) >= 3:
            notes.append(f"exact_page demoted: {ratio:.0%} look like content-marketing "
                         "sites (misidentified DOM panel pattern)")
            return [], exact, notes
        return exact, [], notes

    if scrape_mode == "serp:vsa3-weblist":
        notes.append("on vsa=3 but results are a derived-text-query web list "
                     "(no matching-images section) - NOT labelled exact_page")
        return [], list(raw_matches), notes

    notes.append(f"results from Bing's default page (mode={scrape_mode})")
    return [], list(raw_matches), notes


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
      * matches - each dict has `source_url`, `title`, `rank`, and
        `match_type` = "exact_page" | "visual_similarity".
      * debug_dump - set only when we got nothing and Bing gave no explicit card.
      * meta = {
          "match_strength": "high"|"moderate"|"none",  # high = >=1 exact_page,
                                                       # moderate = visual only
          "match_type": "exact_page"|"visual_similarity"|None,  # of the returned tier
          "exact_page_count": int, "visual_similarity_count": int,
          "result_source": <human string>, "bing_note": <str|None>,
          "visual_matches_fallback": {attempted,tab_found,landed,usable_links}|None,
        }
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

            pages_tab_present = tab_label is not None
            on_vsa3 = bool(_PAGES_TAB_URL_RE.search(page.url or ""))

            # ---- Tier 1: exact_page ("Pages with this image") ----------------
            with step(log, "Scraping the 'Pages with this image' SERP"):
                scraped = _collect_result_anchors(page)
                raw = extract_page_urls(scraped.get("links", []))
                log.info("Kept %d off-site link(s) (from %d raw, scrape mode %s, on_vsa3=%s)",
                         len(raw), len(scraped.get("links", [])), scraped.get("mode"), on_vsa3)
            scrape_mode = str(scraped.get("mode", ""))
            bing_zero_msg = (scraped.get("error") or "").strip()

            exact_matches, derived_serp_matches, tier_notes = assign_tiers(
                scrape_mode, raw, bing_zero_msg=bing_zero_msg)
            for n in tier_notes:
                log.warning("tier check: %s", n)
            if exact_matches:
                exact_matches = tag_matches(exact_matches, MATCH_TYPE_EXACT)
                log.info("%d exact_page candidate(s) from a labelled matching-images section",
                         len(exact_matches))
            elif bing_zero_msg:
                log.info("Bing: %r  ->  0 exact_page matches (the #b_results web list here "
                         "is a text-query fallback, not image matches)", bing_zero_msg)
            elif derived_serp_matches:
                log.warning("%d link(s) are NOT confirmed exact-image pages - routing to the "
                            "visual_similarity tier instead", len(derived_serp_matches))

            # ---- Tier 2: visual_similarity ("Visual Matches" tab) -----------
            visual_matches: list[dict] = []
            vm_fallback: dict | None = None
            if not exact_matches:
                log.info("No exact-image pages - trying the 'Visual Matches' tab...")
                vm_fallback = {"attempted": True, "tab_found": False,
                               "landed": False, "usable_links": 0}
                fb_timeout_ms = min(timeout_ms, 15000)  # don't double the wall time
                with step(log, "Opening the 'Visual Matches' tab"):
                    vm_label, vm_landed = _open_result_tab(
                        page, _VISUAL_MATCHES_RE, _VISUAL_MATCHES_URL_RE, fb_timeout_ms,
                        "Visual Matches")
                    if vm_label is not None:
                        _wait_for_serp_results(page, fb_timeout_ms)
                        _settle(page)
                if vm_label is None:
                    log.warning("No 'Visual Matches' tab on this page (Bing's markup, "
                                "or a headless shell, or a face photo Bing won't reverse-search)")
                else:
                    vm_fallback["tab_found"] = True
                    vm_fallback["landed"] = vm_landed
                    vm_scraped = _collect_visual_matches(page)
                    vm_matches = extract_page_urls(vm_scraped.get("links", []))
                    vm_fallback["usable_links"] = len(vm_matches)
                    if vm_matches:
                        log.info("'Visual Matches' tab supplied %d candidate(s)", len(vm_matches))
                        visual_matches = vm_matches
                    else:
                        log.warning("'Visual Matches' tab returned 0 usable links for this image")

            # If neither tab produced anything but Bing's default page had links,
            # use those as the visual tier (honestly labelled).
            if not exact_matches and not visual_matches and derived_serp_matches:
                log.warning("Falling back to Bing's default-page results as "
                            "visual_similarity candidates (%d link(s))", len(derived_serp_matches))
                visual_matches = derived_serp_matches
            if visual_matches:
                visual_matches = tag_matches(visual_matches, MATCH_TYPE_VISUAL)

            # ---- real breakage: nothing anywhere, no tab, no "no pages" card -
            vm_tab_found = bool(vm_fallback and vm_fallback["tab_found"])
            if (not pages_tab_present and not vm_tab_found
                    and not exact_matches and not visual_matches and not bing_zero_msg):
                raise ScrapeError(
                    "uploaded the image but found no results tabs ('Pages with this image' / "
                    "'Visual Matches') and no result links anywhere. Bing's visual-search UI "
                    "has likely changed - re-run with --headed and update _open_result_tab() / "
                    "_collect_result_anchors() / _collect_visual_matches().",
                    debug_dump=_dump_debug(page, debug_dir),
                )

            matches = exact_matches or visual_matches
            match_strength, result_source = score_matches(exact_matches, visual_matches)

            # ---- DEBUG: exactly what we're about to return, and from where ----
            log.info("Candidates: %d exact_page, %d visual_similarity  "
                     "(scrape_mode=%s, on_vsa3=%s, pages_tab=%r, vm_tab=%s, bing_card=%r)  "
                     "->  match_strength=%s",
                     len(exact_matches), len(visual_matches), scrape_mode, on_vsa3,
                     tab_label, (vm_fallback or "not attempted"),
                     bing_zero_msg or None, match_strength)
            for i, m in enumerate(matches, 1):
                log.info("    candidate[%d] (%s)  %s", i, m.get("match_type"),
                         m.get("source_url"))
            if not matches:
                log.info("    (no candidates in either tier)")

            # extra guard: an exact_page set dominated by content-marketing
            # domains is the classic "scraped the wrong panel" tell - warn loudly
            # even though assign_tiers should already have demoted it.
            if exact_matches:
                cm = content_marketing_ratio(exact_matches)
                if cm >= 0.5:
                    log.warning("!! %.0f%% of the exact_page candidates look like generic "
                                "content-marketing / listicle sites - treat this "
                                "match_strength=high with suspicion (possible DOM misread): %s",
                                cm * 100, ", ".join(
                                    urllib.parse.urlparse(m["source_url"]).hostname
                                    for m in exact_matches[:6]))

            if match_strength == "high":
                log.info("Returning %d exact_page candidate(s) from %s  ->  match_strength=high",
                         len(matches), result_source)
            elif match_strength == "moderate":
                log.warning("Returning %d visual_similarity candidate(s) from %s  ->  "
                            "match_strength=moderate (no confirmed exact-image pages)",
                            len(matches), result_source)
            else:
                log.warning("No candidates in either tier  ->  match_strength=none (%s)",
                            result_source)
            if bing_zero_msg:
                log.info("Bing card on the page: %r", bing_zero_msg)

            if keep_open:
                log.info("--keep-open: leaving the browser open; press Enter here to close it")
                try:
                    input()
                except EOFError:
                    page.wait_for_timeout(15000)

            debug_dump = (_dump_debug(page, debug_dir)
                          if (not matches and not bing_zero_msg) else None)
            meta = {
                "result_source": result_source,
                "match_strength": match_strength,
                "match_type": (MATCH_TYPE_EXACT if exact_matches
                               else MATCH_TYPE_VISUAL if visual_matches else None),
                "exact_page_count": len(exact_matches),
                "visual_similarity_count": len(visual_matches),
                "bing_note": bing_zero_msg or None,
                "visual_matches_fallback": vm_fallback,
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
        note = {
            "high": "exact_page - pages using this exact image",
            "moderate": "visual_similarity only - no exact-image pages",
            "none": "no usable results in either tier",
        }.get(strength, strength)
        print(f"match       : {strength.upper()}  ({note})")
        print(f"tiers       : {q.get('exact_page_count', 0)} exact_page, "
              f"{q.get('visual_similarity_count', 0)} visual_similarity")
        print(f"source      : {q.get('result_source')}")
    print(f"results     : {report['match_count']} candidate(s)")
    if q.get("bing_note"):
        print(f"bing note   : {q['bing_note']}")
    print()

    for m in report["matches"]:
        print(f"  #{m['rank']:<2} [{m.get('match_type', '?')}]  {m['source_url']}")
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
            log.warning('No candidates - Bing card: "%s"  (exit %d)',
                        meta["bing_note"], EXIT_ZERO_MATCHES)
        else:
            log.warning("No candidates in either tier (exact_page / visual_similarity)  "
                        "(exit %d)", EXIT_ZERO_MATCHES)
            if not args.headed:
                log.warning("Re-run with --headed - Bing serves degraded results to "
                            "headless browsers.")
        if debug_dump:
            log.info("Saved the page for inspection: %s", debug_dump)
        return EXIT_ZERO_MATCHES

    log.info("face_search_fallback done: %d candidate(s), match_strength=%s (%s)  (exit 0)",
             report["match_count"], meta.get("match_strength", "?"),
             meta.get("match_type", "?"))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
