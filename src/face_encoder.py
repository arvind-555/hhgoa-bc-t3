"""Piece 1 - face detection + encoding (HH Goa Task 3).

Take an image path, detect a face with `face_recognition`, print a 128-d
encoding as JSON. No network, no blockchain - later pieces consume the JSON.

Behaviour:
  * missing / unreadable file        -> error, exit 4
  * no face found                    -> error, exit 2
  * exactly one face                 -> encode it
  * more than one face               -> error listing every face, exit 3;
                                        caller must pass --face-index N
  * --face-index out of range        -> error, exit 5

Faces are indexed left-to-right, then top-to-bottom, so the index is
stable across runs. Use --list-faces to see the boxes before choosing.

Usage:
    python src/face_encoder.py data/input/me.jpg
    python src/face_encoder.py data/input/me.jpg --out output/me.json
    python src/face_encoder.py data/input/group.jpg --list-faces
    python src/face_encoder.py data/input/group.jpg --face-index 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import face_recognition
import numpy as np

from pipeline_log import fail, get_logger, step

log = get_logger("face_encoder")

ENCODER_NAME = "face_recognition / dlib ResNet-29, 128-d"

EXIT_OK = 0
EXIT_NO_FACE = 2
EXIT_MULTIPLE_FACES = 3
EXIT_BAD_INPUT = 4
EXIT_INDEX_RANGE = 5


class FaceEncoderError(Exception):
    """Base class - carries a process exit code."""

    exit_code = 1


class BadInput(FaceEncoderError):
    exit_code = EXIT_BAD_INPUT


class NoFaceFound(FaceEncoderError):
    exit_code = EXIT_NO_FACE


class MultipleFacesFound(FaceEncoderError):
    exit_code = EXIT_MULTIPLE_FACES


class FaceIndexOutOfRange(FaceEncoderError):
    exit_code = EXIT_INDEX_RANGE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _box_dict(box: tuple[int, int, int, int]) -> dict:
    top, right, bottom, left = box
    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
        "width": right - left,
        "height": bottom - top,
    }


def load_image(image_path: str | Path) -> tuple[Path, "np.ndarray"]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise BadInput(f"file not found: {image_path}")
    if not image_path.is_file():
        raise BadInput(f"not a file: {image_path}")
    with step(log, f"Loading image {image_path}"):
        try:
            image = face_recognition.load_image_file(str(image_path))
        except Exception as exc:  # PIL throws assorted errors on junk input
            raise BadInput(f"could not read image {image_path.name}: {exc}") from exc
    h, w = image.shape[0], image.shape[1]
    log.info("Image loaded: %d x %d px, %.0f KB on disk",
             w, h, image_path.stat().st_size / 1024)
    return image_path, image


def detect_faces(image: "np.ndarray", model: str = "hog") -> list[tuple[int, int, int, int]]:
    """Return face boxes ordered left-to-right, then top-to-bottom."""
    with step(log, f"Detecting faces (detector={model})"):
        boxes = face_recognition.face_locations(image, model=model)
    # box = (top, right, bottom, left); sort by left, then top
    boxes = sorted(boxes, key=lambda b: (b[3], b[0]))
    if boxes:
        shapes = ", ".join(
            f"[{i}] {r - l}x{b - t}px" for i, (t, r, b, l) in enumerate(boxes)
        )
        log.info("Detected %d face(s): %s", len(boxes), shapes)
    else:
        log.info("Detected 0 faces")
    return boxes


def encode_face(
    image_path: str | Path,
    face_index: int | None = None,
    model: str = "hog",
    num_jitters: int = 1,
) -> dict:
    """Detect faces and return an encoding record for the chosen face.

    face_index is required (and validated) when more than one face is found;
    ignored-if-0 / validated when exactly one face is found.
    """
    log.info("encode_face: image=%s face_index=%s model=%s jitters=%d",
             image_path, face_index, model, num_jitters)
    path, image = load_image(image_path)
    boxes = detect_faces(image, model=model)

    if not boxes:
        raise NoFaceFound(
            f"no face detected in {path.name} with the '{model}' detector - "
            f"try --model cnn, or use a clearer / better-lit photo"
        )

    if len(boxes) > 1 and face_index is None:
        listing = "\n".join(
            f"  [{i}] {_box_dict(b)}" for i, b in enumerate(boxes)
        )
        raise MultipleFacesFound(
            f"{len(boxes)} faces detected in {path.name}; "
            f"re-run with --face-index (0..{len(boxes) - 1}):\n{listing}"
        )

    idx = 0 if face_index is None else face_index
    if not (0 <= idx < len(boxes)):
        raise FaceIndexOutOfRange(
            f"--face-index {idx} out of range; {len(boxes)} face(s) detected "
            f"(valid 0..{len(boxes) - 1})"
        )

    chosen = boxes[idx]
    log.info("Encoding face #%d  box=%s", idx, _box_dict(chosen))
    with step(log, f"Computing 128-d encoding (num_jitters={num_jitters})"):
        encoding = face_recognition.face_encodings(
            image, known_face_locations=[chosen], num_jitters=num_jitters
        )[0]

    sha = _sha256(path)
    log.info("Encoded face #%d -> %d-d vector; source SHA-256 %s",
             idx, int(encoding.shape[0]), sha)
    return {
        "encoding": encoding.tolist(),
        "encoding_dim": int(encoding.shape[0]),
        "face_index": idx,
        "face_box": _box_dict(chosen),
        "faces_detected": len(boxes),
        "all_face_boxes": [_box_dict(b) for b in boxes],
        "meta": {
            "source_image": path.name,
            "source_sha256": sha,
            "model": model,
            "num_jitters": num_jitters,
            "encoder": ENCODER_NAME,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }


def _list_faces(image_path: str | Path, model: str) -> dict:
    log.info("list-faces: image=%s model=%s", image_path, model)
    path, image = load_image(image_path)
    boxes = detect_faces(image, model=model)
    return {
        "source_image": path.name,
        "model": model,
        "faces_detected": len(boxes),
        "faces": [{"index": i, "box": _box_dict(b)} for i, b in enumerate(boxes)],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="face_encoder",
        description="Detect a face and emit its 128-d encoding as JSON.",
    )
    p.add_argument("image", help="path to input image (jpg/png)")
    p.add_argument(
        "--face-index",
        type=int,
        default=None,
        metavar="N",
        help="which face to encode (required when >1 face is found); "
        "faces are ordered left-to-right, top-to-bottom",
    )
    p.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="detector: hog=CPU (default), cnn=needs dlib built with CUDA",
    )
    p.add_argument(
        "--jitters",
        type=int,
        default=1,
        metavar="N",
        help="re-sample count for the encoding; 10 = sturdier enrollment vector",
    )
    p.add_argument(
        "--list-faces",
        action="store_true",
        help="just detect and print the face boxes with their indices, then exit",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="also write the JSON to this path (default: stdout only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "list-faces" if args.list_faces else "encode"
    log.info("face_encoder starting  (mode=%s, image=%s)", mode, args.image)
    try:
        if args.list_faces:
            result = _list_faces(args.image, model=args.model)
        else:
            result = encode_face(
                args.image,
                face_index=args.face_index,
                model=args.model,
                num_jitters=args.jitters,
            )
    except FaceEncoderError as exc:
        return fail(log, str(exc), exc.exit_code)

    text = json.dumps(result, indent=2)
    print(text)  # stdout: the JSON record for later pieces
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        log.info("Wrote encoding record to %s", out_path)
    log.info("face_encoder done  (exit 0)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
