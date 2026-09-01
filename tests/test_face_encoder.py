"""Piece 1 plumbing tests. Real verification is the manual run in the README.

Optional fixtures (gitignored), exercised only if present:
  tests/fixtures/one_face.jpg    - a single clear face
  tests/fixtures/two_faces.jpg   - two faces
  tests/fixtures/no_face.jpg     - a photo with no face
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import face_encoder as fe  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
ONE = FIX / "one_face.jpg"
TWO = FIX / "two_faces.jpg"
NONE_ = FIX / "no_face.jpg"


def test_deps_import():
    import face_recognition  # noqa: F401

    assert hasattr(fe, "encode_face")


def test_missing_file():
    with pytest.raises(fe.BadInput):
        fe.encode_face("nope/missing.jpg")


def test_box_dict_geometry():
    d = fe._box_dict((10, 60, 110, 20))  # top,right,bottom,left
    assert d == {"top": 10, "right": 60, "bottom": 110, "left": 20, "width": 40, "height": 100}


@pytest.mark.skipif(not ONE.is_file(), reason="no fixtures/one_face.jpg")
def test_single_face():
    rec = fe.encode_face(ONE)
    assert np.asarray(rec["encoding"]).shape == (128,)
    assert rec["encoding_dim"] == 128
    assert rec["faces_detected"] == 1
    assert rec["face_index"] == 0


@pytest.mark.skipif(not TWO.is_file(), reason="no fixtures/two_faces.jpg")
def test_multiple_faces_requires_index():
    with pytest.raises(fe.MultipleFacesFound):
        fe.encode_face(TWO)  # no --face-index -> refuse
    rec = fe.encode_face(TWO, face_index=1)
    assert rec["face_index"] == 1
    assert rec["faces_detected"] == 2
    with pytest.raises(fe.FaceIndexOutOfRange):
        fe.encode_face(TWO, face_index=9)


@pytest.mark.skipif(not NONE_.is_file(), reason="no fixtures/no_face.jpg")
def test_no_face():
    with pytest.raises(fe.NoFaceFound):
        fe.encode_face(NONE_)
