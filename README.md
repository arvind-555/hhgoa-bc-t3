# HH Goa 2026 - Task 3

Face scan -> web/social search for a matching post -> blockchain upload +
re-verification of the discovered data.

**Status:** building in pieces.
- [x] Piece 1 - detect + encode a face from an input image
- [ ] Piece 2 - reverse face search on the web (Lenso.ai, Playwright fallback)
- [ ] Piece 3 - push the found post to Polygon Amoy + re-verify on-chain

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

Plumbing tests always run. Add `tests\fixtures\one_face.jpg`,
`two_faces.jpg`, `no_face.jpg` (all gitignored) to also exercise real
detection.

### Known limitations (piece 1)

- `hog` detector misses profile / small / low-light faces; `cnn` is better
  but needs a CUDA dlib build.
- `--face-index` ordering is geometric (left-to-right, top-to-bottom), so
  it can shift if a face detection flickers between runs on a hard photo.
- Encoding quality depends on the photo; `--jitters 10` helps.
- The 128-d vector **is** biometric data - `output/` and `data/input/` are
  gitignored on purpose.

## Blockchain

Polygon Amoy testnet (chain id 80002). Details land with piece 3.
