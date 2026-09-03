# HH Goa 2026 - Task 3

Face scan -> web/social search for a matching post -> blockchain upload +
re-verification of the discovered data.

**Status:** building in pieces.
- [x] Piece 1 - detect + encode a face from an input image
- [x] Piece 2 - reverse image search (Bing visual search via Playwright; Lenso.ai
      API path exists but its tier is out of budget)
- [x] Piece 3 - Polygon Amoy access (`chain_setup.py`)
- [x] Piece 5 - hash notarize + tamper-verify (`chain_verify.py`)
- [x] Piece 6 - one-command pipeline (`src/main.py`)

Live broadcast (steps 4-5 of the pipeline) is pending a funded wallet -
everything up to it is written and tested.

## Architecture (locked)

| Stage | Tool |
|-------|------|
| Face detection / encoding | `face_recognition` (Python / dlib), 128-d ResNet embedding |
| Reverse face search | Lenso.ai API (free tier); fallback: scripted Bing/Google Images via Playwright |
| Blockchain | Polygon Amoy testnet (free faucet, EVM, Polygonscan for the demo) |

No hosting - pipeline only.

## Ethics / scope

Only run on **yourself or people who have given explicit permission**. No
public figures, no scraped strangers. This is a consent-bound demo.

## Terminal logging

Every script logs its progress to **stderr** via `src/pipeline_log.py` so you
can watch a run and see each step start / finish (with timing), what it found
(`Detected 1 face`, `Lenso returned 12 match(es)`, `Clicked 'Pages with this
image'; landed on the vsa=3 results SERP`), and - on failure - one line naming
the step and the reason (no raw traceback). **stdout** stays clean for the
machine-readable output (the JSON records, the ranked report).

```
14:22:01 INFO    face_encoder starting  (mode=encode, image=data/input/me.jpg)
14:22:01 INFO    START  Detecting faces (detector=hog)
14:22:02 INFO    DONE   Detecting faces (detector=hog)  (0.8s)
14:22:02 INFO    Detected 1 face(s): [0] 214x214px
...
14:22:03 ERROR   ABORTED (exit 2): no face detected in landscape.jpg with the 'hog' detector - try --model cnn
```

`HHGOA_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` (default `INFO`) tunes verbosity.
New pipeline scripts must use `pipeline_log` the same way.

---

## Piece 1 - face encoding

`src/face_encoder.py` takes a photo, detects a face with `face_recognition`,
and prints a JSON record with the 128-d encoding plus metadata (source
SHA-256, model, timestamp). No network, no chain. Later pieces read this JSON.

It refuses to guess: **no face -> error, multiple faces -> error** telling
you to pass `--face-index N` (faces ordered left-to-right, top-to-bottom;
`--list-faces` shows them), **missing file -> error**.

### Install (Windows)

dlib is the only hard part. Use **Python 3.10** (prebuilt dlib wheels
exist; 3.13 would force a source build).

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# no-compiler path: dlib-bin ships a prebuilt wheel
pip install "numpy<2" Pillow click dlib-bin face_recognition_models pytest
pip install face_recognition==1.3.0 --no-deps
```

`--no-deps` on the last line stops pip from trying to compile `dlib` from
source (face_recognition pins the name `dlib`, not `dlib-bin`).

<details>
<summary>If <code>dlib-bin</code> has no wheel for your Python</summary>

Install "Visual Studio Build Tools" with the *Desktop development with C++*
workload and CMake, then `pip install dlib==19.24.*` (compiles, ~5-10 min).
</details>

Verify:

```powershell
python -c "import face_recognition, dlib; print('dlib', dlib.__version__)"
```

### Run (manual test)

```powershell
# 1. put a consented photo at data\input\me.jpg, then:
python src\face_encoder.py data\input\me.jpg
#    -> prints the JSON record (128-d encoding + metadata) to stdout

# 2. save it for later pieces:
python src\face_encoder.py data\input\me.jpg --out output\me.json

# 3. no face -> exit 2
python src\face_encoder.py data\input\landscape.jpg

# 4. group photo -> exit 3, lists the faces:
python src\face_encoder.py data\input\group.jpg
python src\face_encoder.py data\input\group.jpg --list-faces
python src\face_encoder.py data\input\group.jpg --face-index 1   # pick one

# 5. missing file -> exit 4
python src\face_encoder.py data\input\does_not_exist.jpg

# check exit code in PowerShell after any run:
$LASTEXITCODE
```

Flags: `--face-index N`, `--list-faces`, `--model {hog,cnn}` (cnn needs
dlib+CUDA - skip on a 4GB GTX 1650), `--jitters N` (use `10` for a sturdier
enrollment vector), `--out PATH`.

Exit codes: `0` ok, `2` no face, `3` multiple faces (need `--face-index`),
`4` bad input, `5` `--face-index` out of range.

### Automated tests

```powershell
pytest -q
```

Plumbing tests always run (incl. `tests\test_pipeline_log.py` for the shared
logging helper). Add `tests\fixtures\one_face.jpg`, `two_faces.jpg`,
`no_face.jpg` (all gitignored) to also exercise real detection.

### Known limitations (piece 1)

- `hog` detector misses profile / small / low-light faces; `cnn` is better
  but needs a CUDA dlib build.
- `--face-index` ordering is geometric (left-to-right, top-to-bottom), so
  it can shift if a face detection flickers between runs on a hard photo.
- Encoding quality depends on the photo; `--jitters 10` helps.
- The 128-d vector **is** biometric data - `output/` and `data/input/` are
  gitignored on purpose.

---

## Piece 2 - reverse face search (Lenso.ai)

`src/face_search.py` sends the photo to the **live** Lenso.ai facial-recognition
endpoint (`POST https://api.eyematch.ai/search`) and prints the ranked matches
with their source URLs. No mock / fixture / hardcoded result exists in that
file - with no key or no network it fails loudly.

Docs: <https://github.com/lenso-ai/reverse-image-search-api>

### Setup

1. Register at lenso.ai and start a Developer Subscription -> you get an
   Authorization token (or email contact@lenso.ai).
2. `copy .env.example .env` and put the token in `LENSO_API_KEY`.
   `.env` is gitignored.

```powershell
pip install requests
# dev-only, for the mocked tests: pip install responses
```

### Run (manual test - real API call)

```powershell
# from a raw photo:
python src\face_search.py --image data\input\me.jpg

# or reuse piece 1's output (finds + sha256-checks the photo, crops to the
# recorded face box so Lenso searches the right face):
python src\face_search.py --from-encoding output\me.json

# save the ranked report:
python src\face_search.py --image data\input\me.jpg --out output\me.matches.json

$LASTEXITCODE   # 0 = >=1 match, 2 = zero matches
```

Output: a ranked list (`#rank conf=NN.N  date  source_url`) then a boxed
`TOP MATCH` block.

Flags: `--from-encoding` / `--image` (one required), `--image-dir`,
`--no-crop`, `--sort {QUALITY_DESCENDING,QUALITY_ASCENDING,DATE_DESCENDING,DATE_ASCENDING}`,
`--page N`, `--domains-include a.com,b.com`, `--domains-exclude`,
`--from-date`/`--to-date` (yyyy-MM-dd), `--timeout S`, `--max-retries N`
(429 backoff, honours `Retry-After`), `--out`, `--raw-out`.

Exit codes: `0` matches, `2` zero matches, `3` no API key, `4` bad input,
`5` API error, `6` rate limited (429), `7` timeout / connection failure.

### Tests

```powershell
pytest -q tests\test_face_search.py
```

HTTP is mocked in the **test** file only (`responses`). `src/face_search.py`
is never mocked; the real check is the manual run above.

### Known limitations (piece 2)

- Needs a paid Lenso Developer Subscription; free tier / trial limits apply.
- 20 results per page; use `--page` / re-run for more.
- Lenso runs its own face detection on the image - piece 1's 128-d encoding
  can't be sent (not invertible), so we send the photo (optionally cropped
  to piece 1's face box).
- Match quality/coverage is whatever Lenso's index has; a real person can
  legitimately return zero matches.
- The request base64-encodes the whole image; large photos increase latency.

---

## Piece 2 (fallback) - Bing Images visual search (Playwright)

Lenso.ai's Developer Subscription starts at **USD 2,400+/month** - ruled out.
`src/face_search_fallback.py` is the no-API path: it launches a **real Chromium
browser** with Playwright, uploads the photo to Bing Images' *search by image*
(visual search), then opens the **"Pages with this image"** tab and scrapes the
result links from it.

Bing's visual search renders results in-page under tabs
(*Overview / Visual Matches / Pages with this image / Solve*) - it no longer
navigates to `/images/search`. The script clicks **"Pages with this image"**
(the tab whose SERP is tagged `vsa=3`) because those are pages using the
*exact* uploaded photo - the strong signal. It logs whether the links it
returns came from that tab (`match type: STRONG`) or from a weaker fallback
view (`WEAK` - visually-similar / look-alike results), and records the same in
the JSON report (`query.match_strength`, `query.result_source`). An explicit
"Unable to find pages with this image" card is reported as a clean zero.

Every run is a live browser session - no mock, fixture, cached HTML or
hardcoded result exists in that file. If Bing's DOM moved, a CAPTCHA appears, or
nothing matches, it fails loudly.

> **Run this `--headed`.** Bing serves a stripped-down, non-rendering SERP shell
> to obvious headless automation, so headless runs frequently come back with
> zero results even when matches exist. `--headed` (a visible browser window)
> gets the real "Pages with this image" results.

### Setup

```powershell
pip install playwright
playwright install chromium      # one-time: downloads the browser binary
```

### Run (manual test - real browser + live search)

```powershell
# from a raw photo (use --headed - see the note above):
python src\face_search_fallback.py --image data\input\me.jpg --headed

# reuse piece 1's output (locates + sha256-checks the photo; --crop uploads
# just the recorded face box):
python src\face_search_fallback.py --from-encoding output\me.json --crop --headed --out output\me.bing.json

# watch each step slowly / debug selectors:
python src\face_search_fallback.py --image data\input\me.jpg --headed --keep-open --slow-mo 300

$LASTEXITCODE   # 0 = >=1 page, 2 = zero
```

Output: a `match type: STRONG|WEAK` line, the ranked `#rank  source_url` list,
then a boxed `TOP MATCH`. The same detail is in the `--out` JSON
(`query.match_strength`, `query.result_source`, `query.bing_note`).

Flags: `--image` / `--from-encoding` (one required), `--image-dir`, `--crop`,
`--headed`, `--keep-open`, `--slow-mo MS`, `--timeout S`, `--debug-dir DIR`,
`--out`.

Exit codes: `0` pages found, `2` zero pages (incl. Bing's "unable to find pages
with this image" card), `3` Playwright / Chromium not installed, `4` bad input,
`5` Bing UI / selectors not found, `6` CAPTCHA / blocked, `7` navigation timeout.

### When a selector breaks

On exit `5` / `6` (and on a zero-result run that Bing did *not* explicitly flag)
the script writes `output\bing_debug_<timestamp>.png` + `.html` - the screenshot
and live HTML of whatever page it was on. Open the HTML and update the selector
lists in the relevant helper:

- `_find_file_input()` - the camera / "search by image" upload `<input>`
- `_select_pages_tab()` / `_PAGES_TAB_RE` / `_PAGES_TAB_URL_RE` - the
  "Pages with this image" tab and the `vsa=3` URL it lands on
- `_collect_result_anchors()` / `_SERP_RESULT_SELECTOR` - the result rows on
  that tab's SERP (`#b_results li.b_algo`) and the "no pages" error card

`--headed --slow-mo 500` lets you watch every step in a visible window.

### Tests

```powershell
pytest -q tests\test_face_search_fallback.py
```

Pure URL-filtering / input-resolution logic only - no browser is launched, no
network touched (the Playwright import is lazy). The real check is the manual
`--headed` run above.

### Known limitations (piece 2 fallback)

- **Headless is degraded.** Bing serves an empty / plain-web-search SERP shell
  to obvious headless automation instead of the real "Pages with this image"
  results. The script reloads once to coax it and drops most anti-headless
  tells, but `--headed` is the reliable path. CAPTCHA (exit `6`) is the harder
  version of the same block - retry later, from a different IP, or `--headed`
  and solve it by hand.
- Bing has **no official visual-search API**; the markup can change without
  notice and break the selectors - that is expected, hence the debug dumps.
- Visual search matches the *whole image*, not identity - it is weaker than
  Lenso's face search and returns page-level links, not ranked face confidences.
  The "Pages with this image" tab (`match type: STRONG`) is the useful one; a
  `WEAK` result means the script only saw visually-similar / look-alike images.
- Scrapes only the public result links Bing renders; login-walled or
  JS-deferred results below the fold may be missed.

---

## Piece 3 - Polygon Amoy setup

`src/chain_setup.py` gets you onto the **Polygon Amoy testnet** (chain id
**80002**): it generates a throwaway wallet once, connects over web3.py, and
reports the wallet's POL balance so you can fund it.

| | |
|---|---|
| Network | Polygon Amoy testnet |
| Chain ID | 80002 |
| RPC | `https://rpc-amoy.polygon.technology` (override with `AMOY_RPC_URL`) |
| Explorer | <https://amoy.polygonscan.com> |
| Faucet | <https://bwarelabs.com/faucets/polygon-testnet> |

### Secrets

The private key is generated locally and written **only to `.env`** (gitignored
- confirmed by `git check-ignore .env`). It is never hard-coded and never
printed in full; the terminal shows only the last 4 chars (`...e824`) for
confirmation. The wallet **address** is public - safe to paste into a faucet or
explorer.

### Setup

```powershell
pip install web3
copy .env.example .env          # if you don't have a .env yet
python src\chain_setup.py
```

What it does:

1. **Wallet** - reuses `WALLET_PRIVATE_KEY` from `.env` if present (never
   regenerates over funds); otherwise creates one and writes
   `WALLET_PRIVATE_KEY` + `WALLET_ADDRESS` to `.env`. `--force-new` replaces it.
2. **Connect** - web3.py to the Amoy RPC; verifies the RPC really is on chain
   80002.
3. **Address** - printed plainly on stdout (safe to share).
4. **Balance** - prints your POL balance.
5. If balance is 0, prints:
   `Get free testnet POL: paste this address into https://bwarelabs.com/faucets/polygon-testnet`

```powershell
python src\chain_setup.py --rpc-url https://polygon-amoy.g.alchemy.com/v2/KEY   # private RPC
python src\chain_setup.py --force-new                                           # new wallet
$LASTEXITCODE   # 0 = funded, 2 = zero balance (go to the faucet)
```

If the default public RPC is unreachable, try `AMOY_RPC_URL` =
`https://polygon-amoy-bor-rpc.publicnode.com` or `https://polygon-amoy.drpc.org`.

Exit codes: `0` funded, `2` zero balance, `3` web3 not installed, `4` bad
config (bad key in `.env`), `5` RPC unreachable, `6` RPC on the wrong chain.

### Tests

```powershell
pytest -q tests\test_chain_setup.py
```

`.env` read/write helpers, secret masking, and the local wallet lifecycle
(reuse / `--force-new`). No RPC is contacted.

---

## Piece 5 - hash notarization + verification

`src/chain_verify.py` puts a discovered post (piece 2's output) on-chain and
proves later that a copy is untampered.

- **`upload_hash(post_data: dict) -> dict`** - SHA-256 the post over a
  deterministic serialization (JSON, keys sorted, no whitespace, UTF-8), put the
  digest in the **calldata** of a plain self-transaction on Amoy (no smart
  contract), sign + broadcast with the `.env` wallet, and return
  `{tx_hash, hash, polygonscan_url, broadcast}`. If the wallet can't cover gas
  it **stops before building the transaction** and returns `broadcast: False`
  with a reason - it never fails confusingly on "insufficient funds".
- **`verify(post_data: dict, tx_hash: str) -> bool`** - re-hash the post the
  same way, pull the transaction back off-chain, read the digest out of its
  calldata, compare. Logs which hash came from where and whether they matched.

Calldata layout: `b"HHGOAv1:"` (8-byte marker) + `sha256(post_data)` (32 bytes).

```powershell
python src\chain_verify.py upload  match.json                 # -> {tx_hash, hash, polygonscan_url}
python src\chain_verify.py verify  match.json  0x<tx_hash>     # -> MATCH / MISMATCH
```

Exit codes: `0` ok, `2` wallet can't cover gas, `3` web3 missing, `4` bad input,
`5` RPC error, `6` hash mismatch, `7` couldn't fetch / parse the transaction.

### Demo: is the blockchain step real?

```powershell
python demo_tamper_verify.py
```

Notarizes a sample post, `verify()`s it (passes), changes one field, `verify()`s
again against the same tx (**fails**). That failing second check is the proof
the chain step is load-bearing, not decorative. With a 0-balance wallet the demo
still shows the hash-level guarantee (one field changed -> completely different
SHA-256) and tells you to fund + re-run for the on-chain round-trip.

### Tests

```powershell
pytest -q tests\test_chain_verify.py
```

Hashing determinism, calldata encode/decode, and `verify()` match/mismatch -
with the wallet, RPC and broadcast all mocked, so no funds or network needed.
The live broadcast is exercised by hand once the wallet is funded.

### Status

Wallet `0x670D925B49F2188749FE390CE37654Ef69c51f05` is **unfunded** - everything
up to the broadcast call is written and tested; the live broadcast + on-chain
`verify()` round-trip get tested once POL lands from the faucet.

---

## Piece 6 - the whole trace, one command

```powershell
python src\main.py path\to\image.jpg
```

`src/main.py` runs the full chain and prints a step-by-step trace:

| Step | Module | Does |
|---|---|---|
| 1 | `face_encoder` | detect + encode the face |
| 2 | `face_search_fallback` | Bing reverse-image search for that photo |
| 3 | *(extract)* | pick the best matched page + its metadata -> a `matched_post` dict |
| 4 | `chain_verify.upload_hash` | notarize `matched_post` on Polygon Amoy |
| 5 | `chain_verify.verify` | confirm it round-trips off-chain |

Every run writes a full JSON trace to `output/trace_<timestamp>.json` (the
`matched_post`, its hash, the tx, per-step status).

**Unfunded wallet**: steps 1-3 run in full, then step 4 detects the empty
wallet, logs the faucet message, and **stops cleanly** (exit 2) - no crash, no
silent skip. Fund the wallet (`python src/chain_setup.py`) and re-run.

Flags: `--face-index N`, `--model {hog,cnn}`, `--headed` (far better search
results), `--search-timeout S`, `--env-file`, `--rpc-url`, `--out`.

Exit codes: `0` complete (face -> match -> notarized -> verified), `2` unfunded
(stopped at step 4), `3` no match (search found nothing to notarize), `4` bad
input / missing dependency, `5` a step failed.

### Tests

```powershell
pytest -q tests\test_main_pipeline.py
```

Orchestration flow only - the heavy pieces are faked, `chain_verify` is real
with `upload_hash`/`verify` stubbed. Checks the wiring and every exit code; no
dlib, browser, or network.
