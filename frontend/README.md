# Trace viewer

`viewer.html` — a single static HTML file that renders the `trace_*.json` files
the pipeline writes to `output/`. No backend, no server, no network calls; the
file you pick is read in the browser with `FileReader`. Nothing is uploaded.

## How to open it

**Option A — just open the file (simplest):**

Double-click `frontend/viewer.html`, or drag it into a browser tab. The address
bar will show `file:///.../frontend/viewer.html`. That's fine — everything runs
client-side.

**Option B — local static server (if your browser is fussy about `file://`):**

```powershell
cd D:\hhgoa-bc-t3\frontend
python -m http.server 8000
```

Then open <http://localhost:8000/viewer.html>. Stop the server with `Ctrl+C`.

## How to use it

1. Click **Choose trace_\*.json** (or drag a file onto the page).
2. Pick a file from `D:\hhgoa-bc-t3\output\`, e.g. `trace_20260904-172459.json`.
3. Read the trace:
   - **Outcome pill** — `Complete` / `No match found` / `Stopped — wallet
     unfunded` / `Failed` / `Incomplete`. A `no-match` is shown as a valid
     result, not an error.
   - **Step rail** — the 5 pipeline steps with ✓ / ■ / – / ✕ status.
   - **1 · Face detection** — faces found, which one was used, the face box
     (with a small schematic), model, source SHA-256.
   - **2 · Reverse image search** — `match_strength` (`HIGH` / `MODERATE` /
     `NONE`), the tier counts, and every candidate with a colour-coded
     `EXACT PAGE` (green) or `VISUAL SIMILARITY` (amber) badge and a link.
     If the `exact_page` tier was rejected as content-marketing, that warning
     shows here.
   - **3 · Matched post** — the chosen candidate, or a plain "no matched post"
     state.
   - **4–5 · Blockchain** — the post hash, the `tx_hash`, a **View transaction
     on Polygonscan** button (the `polygonscan_url`), and the on-chain
     verification result. If the wallet was unfunded it says so and shows the
     hash that *would* be notarised.
4. **raw trace JSON** at the bottom expands the full file.

To view a different trace, just pick another file — no reload needed.
