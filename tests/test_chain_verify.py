"""Piece 5 - hashing + verification logic, with the broadcast step mocked.

No funded wallet and no live network: `chain_setup.connect` / `load_or_create_wallet`
are replaced with fakes, so these run anywhere. The live broadcast is exercised
by hand once the wallet is funded (see demo_tamper_verify.py).
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_verify as cv  # noqa: E402

SAMPLE = {
    "platform": "example.social",
    "post_url": "https://example.social/@dana/posts/8842",
    "author_handle": "dana",
    "caption": "Sunset at Anjuna beach \U0001f305 #goa2026",
    "posted_utc": "2026-01-14T17:32:00Z",
    "image_sha256": "0x" + "ab" * 32,
    "match_confidence": 0.94,
}

WALLET_ADDR = "0x" + "00" * 19 + "2a"
WALLET_KEY = b"\x11" * 32


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class _FakeWallet:
    address = WALLET_ADDR
    key = WALLET_KEY


class _Signed:
    raw_transaction = b"\x02\xf8-signed-raw-tx-"


class _TxHash:
    def __init__(self, h: str):
        self._h = h

    def hex(self) -> str:
        return self._h


class _FakeAccountModule:
    def __init__(self):
        self.calls = []

    def sign_transaction(self, tx, private_key):
        self.calls.append((tx, private_key))
        return _Signed()


class _FakeEth:
    def __init__(self, *, balance, tx=None, tx_error=False):
        self.balance = balance
        self._tx = tx
        self._tx_error = tx_error
        self.account = _FakeAccountModule()
        self.sent = []

    def get_balance(self, addr):
        return self.balance

    def get_transaction_count(self, addr):
        return 3

    @property
    def max_priority_fee(self):
        return 25_000_000_000

    def get_block(self, which):
        return {"baseFeePerGas": 20_000_000_000}

    def estimate_gas(self, tx):
        return 21_600

    def send_raw_transaction(self, raw):
        self.sent.append(raw)
        return _TxHash("0x" + "cd" * 32)

    def wait_for_transaction_receipt(self, txh, timeout=0):
        return {"status": 1, "blockNumber": 4242}

    def get_transaction(self, txh):
        if self._tx_error:
            raise RuntimeError("transaction not found")
        return self._tx


class FakeW3:
    def __init__(self, **kw):
        self.eth = _FakeEth(**kw)

    def from_wei(self, n, unit):
        return Decimal(n) / Decimal(10 ** 18)

    def to_wei(self, n, unit):
        assert unit == "gwei"
        return int(Decimal(str(n)) * (10 ** 9))


@pytest.fixture
def patched(monkeypatch):
    holder = types.SimpleNamespace(w3=None, wallet=_FakeWallet())
    monkeypatch.setattr(cv, "_rpc_url", lambda env: "http://fake-rpc")
    monkeypatch.setattr(cv.cs, "connect", lambda url, timeout=20.0: holder.w3)
    monkeypatch.setattr(cv.cs, "load_wallet", lambda env: holder.wallet)
    return holder


def _tx_carrying(post: dict, block=99) -> dict:
    return {"input": cv.encode_calldata(cv.hash_post(post)), "blockNumber": block}


# --------------------------------------------------------------------------
# hashing  (pure)
# --------------------------------------------------------------------------
def test_canonical_bytes_is_key_order_independent():
    a = {"b": 2, "a": 1, "z": {"y": 1, "x": 2}}
    b = {"a": 1, "z": {"x": 2, "y": 1}, "b": 2}
    assert cv.canonical_bytes(a) == cv.canonical_bytes(b)
    assert cv.hash_post(a) == cv.hash_post(b)


def test_hash_post_shape_and_known_vector():
    h = cv.hash_post({"x": 1})
    assert h.startswith("0x") and len(h) == 66
    assert h == "0x" + hashlib.sha256(b'{"x":1}').hexdigest()
    assert h == cv.hash_post({"x": 1})  # stable


def test_hash_changes_on_any_single_field_edit():
    h0 = cv.hash_post(SAMPLE)
    for key, val in SAMPLE.items():
        edited = dict(SAMPLE)
        edited[key] = val + "!" if isinstance(val, str) else "CHANGED"
        assert cv.hash_post(edited) != h0, f"editing {key} did not change the hash"


def test_hash_post_rejects_non_json():
    with pytest.raises(cv.VerifyError):
        cv.hash_post({"when": object()})


def test_calldata_roundtrip():
    h = cv.hash_post(SAMPLE)
    assert cv.decode_calldata(cv.encode_calldata(h)) == h


def test_encode_calldata_rejects_wrong_length():
    with pytest.raises(ValueError):
        cv.encode_calldata("0x1234")


def test_decode_calldata_rejects_foreign_and_truncated():
    with pytest.raises(cv.VerifyError):
        cv.decode_calldata(b"NOTOURS:" + b"\x00" * 32)
    with pytest.raises(cv.VerifyError):
        cv.decode_calldata(cv.CALLDATA_MAGIC + b"\x00" * 10)


def test_hashes_match_is_case_and_prefix_insensitive():
    assert cv.hashes_match("0xABCD", "abcd")
    assert not cv.hashes_match("0xabcd", "0xabce")


# --------------------------------------------------------------------------
# upload_hash  (broadcast mocked)
# --------------------------------------------------------------------------
def test_upload_hash_stops_when_wallet_empty(patched):
    patched.w3 = FakeW3(balance=0)
    res = cv.upload_hash(SAMPLE, wait=False)
    assert res["broadcast"] is False
    assert res["tx_hash"] is None
    assert res["polygonscan_url"] is None
    assert res["hash"] == cv.hash_post(SAMPLE)
    assert "reason" in res
    assert patched.w3.eth.sent == []            # never broadcast
    assert patched.w3.eth.account.calls == []   # never signed


def test_upload_hash_stops_when_balance_too_low_for_gas(patched):
    patched.w3 = FakeW3(balance=1000)  # 1000 wei
    res = cv.upload_hash(SAMPLE, wait=False)
    assert res["broadcast"] is False
    assert patched.w3.eth.sent == []


def test_upload_hash_broadcasts_when_funded(patched):
    patched.w3 = FakeW3(balance=10 ** 18)  # 1 POL
    res = cv.upload_hash(SAMPLE, wait=True)
    assert res["broadcast"] is True
    assert res["tx_hash"] == "0x" + "cd" * 32
    assert res["hash"] == cv.hash_post(SAMPLE)
    assert res["polygonscan_url"].endswith("/tx/0x" + "cd" * 32)
    assert len(patched.w3.eth.sent) == 1
    assert len(patched.w3.eth.account.calls) == 1
    tx, key = patched.w3.eth.account.calls[0]
    assert tx["data"] == "0x" + cv.encode_calldata(res["hash"]).hex()
    assert tx["to"] == WALLET_ADDR
    assert "from" not in tx  # eth-account rejects a `from` field
    assert key == WALLET_KEY


# --------------------------------------------------------------------------
# verify  (get_transaction mocked)
# --------------------------------------------------------------------------
def test_verify_true_when_post_matches_chain(patched):
    patched.w3 = FakeW3(balance=0, tx=_tx_carrying(SAMPLE))
    assert cv.verify(SAMPLE, "0xabc") is True


def test_verify_false_when_post_was_tampered(patched):
    patched.w3 = FakeW3(balance=0, tx=_tx_carrying(SAMPLE))
    tampered = {**SAMPLE, "match_confidence": 0.99}
    assert cv.verify(tampered, "0xabc") is False


def test_verify_works_on_a_pending_tx(patched):
    patched.w3 = FakeW3(balance=0, tx={"input": cv.encode_calldata(cv.hash_post(SAMPLE)),
                                       "blockNumber": None})
    assert cv.verify(SAMPLE, "0xabc") is True


def test_verify_raises_when_tx_missing(patched):
    patched.w3 = FakeW3(balance=0, tx_error=True)
    with pytest.raises(cv.VerifyError):
        cv.verify(SAMPLE, "0xdeadbeef")


def test_verify_raises_on_foreign_calldata(patched):
    patched.w3 = FakeW3(balance=0, tx={"input": b"\x00\x01\x02 not ours", "blockNumber": 1})
    with pytest.raises(cv.VerifyError):
        cv.verify(SAMPLE, "0xabc")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_upload_unfunded_returns_exit_2(patched, tmp_path, capsys):
    patched.w3 = FakeW3(balance=0)
    f = tmp_path / "post.json"
    f.write_text(json.dumps(SAMPLE), encoding="utf-8")
    rc = cv.main(["upload", str(f)])
    assert rc == cv.EXIT_UNFUNDED
    assert json.loads(capsys.readouterr().out)["broadcast"] is False


def test_cli_verify_mismatch_returns_exit_6(patched, tmp_path, capsys):
    patched.w3 = FakeW3(balance=0, tx=_tx_carrying(SAMPLE, block=5))
    f = tmp_path / "post.json"
    f.write_text(json.dumps({**SAMPLE, "author_handle": "not-dana"}), encoding="utf-8")
    rc = cv.main(["verify", str(f), "0xabc"])
    assert rc == cv.EXIT_MISMATCH
    assert "MISMATCH" in capsys.readouterr().out


def test_cli_bad_json_returns_exit_4(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    assert cv.main(["verify", str(f), "0xabc"]) == cv.EXIT_BAD_INPUT
