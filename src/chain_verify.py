"""Piece 5 - notarize a discovered post on-chain, and verify it later.

Two public functions:

    upload_hash(post_data: dict) -> dict
        SHA-256 the post (deterministic canonical JSON), put the digest in the
        calldata of a plain self-transaction on Polygon Amoy, sign + broadcast
        with the wallet from chain_setup.py, and return
        {tx_hash, hash, polygonscan_url}. If the wallet cannot cover gas it
        STOPS before building the transaction and returns broadcast=False with
        a reason (so it never fails confusingly on "insufficient funds").

    verify(post_data: dict, tx_hash: str) -> bool
        Re-hash post_data the same way, pull the transaction back off-chain,
        read the stored digest out of its calldata, and compare. Logs exactly
        which hash came from where and whether they matched.

No smart contract: the digest lives in transaction calldata, prefixed with a
small magic marker (`HHGOAv1:`) so verify() can find and length-check it.

Calldata layout:  b"HHGOAv1:" (8 bytes)  +  sha256(post_data) (32 bytes)

Exit codes (CLI): 0 ok, 2 wallet can't cover gas, 3 web3 missing, 4 bad input,
5 RPC error, 6 hash mismatch, 7 could not fetch / parse the transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import chain_setup as cs
from pipeline_log import fail, get_logger, step

log = get_logger("chain_verify")

CALLDATA_MAGIC = b"HHGOAv1:"          # 8 bytes
DIGEST_LEN = 32                       # sha256

EXIT_OK = 0
EXIT_UNFUNDED = 2
EXIT_NO_DEPS = 3
EXIT_BAD_INPUT = 4
EXIT_RPC = 5
EXIT_MISMATCH = 6
EXIT_VERIFY_ERROR = 7

# Polygon (incl. Amoy) enforces a ~25 gwei floor via bor.
_MIN_PRIORITY_GWEI = 25
_FALLBACK_GAS = 30_000


class VerifyError(Exception):
    """Raised when a transaction can't be fetched or its calldata isn't ours."""

    exit_code = EXIT_VERIFY_ERROR


# --------------------------------------------------------------------------
# hashing  (pure, deterministic - no network, unit tested)
# --------------------------------------------------------------------------
def canonical_bytes(post_data: dict) -> bytes:
    """Deterministic serialization of post_data: JSON, keys sorted, no
    whitespace, UTF-8. The same dict always produces the same bytes regardless
    of key insertion order."""
    try:
        text = json.dumps(post_data, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
    except TypeError as exc:
        raise VerifyError(f"post_data is not JSON-serializable: {exc}") from exc
    return text.encode("utf-8")


def hash_post(post_data: dict) -> str:
    """SHA-256 of the canonical serialization, as a 0x-prefixed hex string."""
    return "0x" + hashlib.sha256(canonical_bytes(post_data)).hexdigest()


def encode_calldata(hash_hex: str) -> bytes:
    """MAGIC + 32 raw digest bytes."""
    raw = bytes.fromhex(hash_hex[2:] if hash_hex.startswith("0x") else hash_hex)
    if len(raw) != DIGEST_LEN:
        raise ValueError(f"hash must be {DIGEST_LEN} bytes, got {len(raw)}")
    return CALLDATA_MAGIC + raw


def decode_calldata(calldata: bytes) -> str:
    """Inverse of encode_calldata -> 0x-prefixed hex digest. Raises VerifyError
    if the calldata wasn't produced by upload_hash()."""
    if not calldata.startswith(CALLDATA_MAGIC):
        raise VerifyError(
            f"transaction calldata has no HH Goa hash marker ({CALLDATA_MAGIC!r}) - "
            f"this tx was not produced by upload_hash()"
        )
    body = calldata[len(CALLDATA_MAGIC):]
    if len(body) != DIGEST_LEN:
        raise VerifyError(
            f"calldata carries {len(body)} bytes after the marker, expected a "
            f"{DIGEST_LEN}-byte SHA-256 digest"
        )
    return "0x" + body.hex()


def hashes_match(a: str, b: str) -> bool:
    return a.lower().removeprefix("0x") == b.lower().removeprefix("0x")


# --------------------------------------------------------------------------
# chain helpers
# --------------------------------------------------------------------------
def _rpc_url(env_file: str | Path) -> str:
    cs.load_dotenv(env_file)
    return os.environ.get(cs.RPC_ENV) or cs.DEFAULT_RPC_URL


def _suggest_fees(w3) -> dict:
    """EIP-1559 fees, kept above Polygon's ~25 gwei floor."""
    floor = w3.to_wei(_MIN_PRIORITY_GWEI, "gwei")
    try:
        priority = int(w3.eth.max_priority_fee)
    except Exception:
        priority = floor
    priority = max(priority, floor)
    try:
        base = int(w3.eth.get_block("latest")["baseFeePerGas"])
    except Exception:
        base = floor
    return {"max_priority": priority, "max_fee": base * 2 + priority, "base": base}


def _calldata_from_tx(tx) -> bytes:
    inp = tx["input"]
    if isinstance(inp, (bytes, bytearray)):
        return bytes(inp)
    return bytes.fromhex(str(inp)[2:] if str(inp).startswith("0x") else str(inp))


def _tx_hash_hex(value) -> str:
    h = value.hex() if hasattr(value, "hex") else str(value)
    return h if h.startswith("0x") else "0x" + h


# --------------------------------------------------------------------------
# 1. upload_hash
# --------------------------------------------------------------------------
def upload_hash(
    post_data: dict,
    *,
    rpc_url: str | None = None,
    env_file: str | Path = ".env",
    wait: bool = True,
    receipt_timeout: float = 120.0,
    rpc_timeout: float = 20.0,
) -> dict:
    """Notarize post_data on Polygon Amoy. See module docstring.

    Returns {"tx_hash", "hash", "polygonscan_url", "broadcast", ["reason"]}.
    When the wallet can't cover gas: broadcast=False, tx_hash=None, and no
    transaction is built or sent.
    """
    log.info("upload_hash: notarizing a %d-key post_data on Polygon Amoy",
             len(post_data) if hasattr(post_data, "__len__") else -1)

    digest = hash_post(post_data)
    calldata = encode_calldata(digest)
    log.info("Local hash (SHA-256 of canonical JSON): %s", digest)
    log.info("Calldata: %s (%d bytes: %d marker + %d digest)",
             "0x" + calldata.hex(), len(calldata), len(CALLDATA_MAGIC), DIGEST_LEN)

    with step(log, "Loading wallet"):
        acct = cs.load_wallet(env_file)  # never creates - run chain_setup.py first
    with step(log, "Connecting to Polygon Amoy"):
        w3 = cs.connect(rpc_url or _rpc_url(env_file), timeout=rpc_timeout)
    wei, pol = cs.check_balance(w3, acct.address)

    fees = _suggest_fees(w3)
    approx_cost = _FALLBACK_GAS * fees["max_fee"]
    log.info("Estimated gas cost: ~%s POL  (%d gas x %.1f gwei)",
             w3.from_wei(approx_cost, "ether"), _FALLBACK_GAS,
             fees["max_fee"] / 1e9)

    # STOP here if the wallet cannot pay - do not build/estimate/sign anything.
    if wei == 0 or wei < approx_cost:
        reason = (
            f"wallet {acct.address} holds {cs._fmt_pol(wei, pol)} POL, which will "
            f"not cover ~{w3.from_wei(approx_cost, 'ether')} POL of gas. "
            f"Fund it at {cs.FAUCET_URL} then re-run."
        )
        log.warning("NOT broadcasting (wallet unfunded): %s", reason)
        return {"tx_hash": None, "hash": digest, "polygonscan_url": None,
                "broadcast": False, "reason": reason}

    with step(log, "Building + signing the transaction"):
        nonce = w3.eth.get_transaction_count(acct.address)
        try:
            gas = w3.eth.estimate_gas({
                "from": acct.address, "to": acct.address,
                "value": 0, "data": "0x" + calldata.hex(),
            })
        except Exception as exc:
            log.warning("estimate_gas failed (%s); using fallback %d", exc, _FALLBACK_GAS)
            gas = _FALLBACK_GAS
        gas_limit = int(gas * 1.25)
        tx = {
            "type": 2,
            "chainId": cs.AMOY_CHAIN_ID,
            "to": acct.address,          # self-send; the calldata is the payload
            "value": 0,
            "nonce": nonce,
            "data": "0x" + calldata.hex(),
            "gas": gas_limit,
            "maxFeePerGas": fees["max_fee"],
            "maxPriorityFeePerGas": fees["max_priority"],
        }
        real_cost = gas_limit * fees["max_fee"]
        if wei < real_cost:
            reason = (f"wallet holds {cs._fmt_pol(wei, pol)} POL but this tx needs "
                      f"~{w3.from_wei(real_cost, 'ether')} POL - fund it at {cs.FAUCET_URL}")
            log.warning("NOT broadcasting: %s", reason)
            return {"tx_hash": None, "hash": digest, "polygonscan_url": None,
                    "broadcast": False, "reason": reason}
        signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
        raw = getattr(signed, "raw_transaction", None)
        if raw is None:
            raw = signed.rawTransaction  # web3 < 7

    with step(log, "Broadcasting to Polygon Amoy"):
        tx_hash = _tx_hash_hex(w3.eth.send_raw_transaction(raw))
    url = f"{cs.EXPLORER_URL}/tx/{tx_hash}"
    log.info("Broadcast OK - tx %s", tx_hash)
    log.info("Polygonscan: %s", url)

    if wait:
        with step(log, "Waiting for the transaction to be mined"):
            try:
                rcpt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=receipt_timeout)
                status = rcpt.get("status") if hasattr(rcpt, "get") else rcpt["status"]
                block = rcpt.get("blockNumber") if hasattr(rcpt, "get") else rcpt["blockNumber"]
                if status == 1:
                    log.info("Mined in block %s (status 1 = success)", block)
                else:
                    log.error("Transaction reverted on-chain (status %s, block %s)",
                              status, block)
            except Exception as exc:
                log.warning("Not confirmed within %.0fs (%s) - it may still mine; "
                            "verify() with the tx hash once it does", receipt_timeout, exc)

    return {"tx_hash": tx_hash, "hash": digest, "polygonscan_url": url, "broadcast": True}


# --------------------------------------------------------------------------
# 2. verify
# --------------------------------------------------------------------------
def verify(
    post_data: dict,
    tx_hash: str,
    *,
    rpc_url: str | None = None,
    env_file: str | Path = ".env",
    rpc_timeout: float = 20.0,
) -> bool:
    """Return True iff post_data hashes to the digest stored in tx_hash's calldata."""
    local = hash_post(post_data)
    log.info("Local hash    (recomputed now from the given post_data): %s", local)

    with step(log, "Connecting to Polygon Amoy"):
        w3 = cs.connect(rpc_url or _rpc_url(env_file), timeout=rpc_timeout)

    with step(log, f"Fetching transaction {tx_hash}"):
        try:
            tx = w3.eth.get_transaction(tx_hash)
        except Exception as exc:
            raise VerifyError(
                f"could not fetch transaction {tx_hash} from Amoy: {exc} - "
                f"wrong hash, wrong network, or not mined yet"
            ) from exc

    on_chain = decode_calldata(_calldata_from_tx(tx))
    block = tx["blockNumber"] if "blockNumber" in tx else None
    where = f"block {block}" if block is not None else "PENDING (not yet mined)"
    log.info("On-chain hash (from tx %s calldata, %s): %s", tx_hash, where, on_chain)

    match = hashes_match(local, on_chain)
    if match:
        log.info("MATCH - post_data is byte-for-byte what was notarized on-chain")
    else:
        log.warning("MISMATCH - post_data does NOT match the on-chain record")
        log.warning("  recomputed : %s", local)
        log.warning("  on-chain   : %s", on_chain)
    return match


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(data).__name__}")
    return data


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chain_verify",
        description="Notarize a discovered post on Polygon Amoy (piece 5), or verify one.",
    )
    p.add_argument("--env-file", default=".env")
    p.add_argument("--rpc-url", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="hash a post_data JSON file and broadcast it")
    up.add_argument("post_file", help="path to a JSON object (e.g. a piece-2 match)")

    vf = sub.add_parser("verify", help="check a post_data JSON file against a tx hash")
    vf.add_argument("post_file")
    vf.add_argument("tx_hash")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        post = _load_json(args.post_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail(log, f"could not read {args.post_file}: {exc}", EXIT_BAD_INPUT)

    try:
        if args.cmd == "upload":
            result = upload_hash(post, rpc_url=args.rpc_url, env_file=args.env_file)
            print(json.dumps(result, indent=2))
            if not result["broadcast"]:
                return EXIT_UNFUNDED
            log.info("chain_verify done  (exit 0)")
            return EXIT_OK

        ok = verify(post, args.tx_hash, rpc_url=args.rpc_url, env_file=args.env_file)
        print("MATCH" if ok else "MISMATCH")
        return EXIT_OK if ok else EXIT_MISMATCH
    except cs.ChainSetupError as exc:
        return fail(log, str(exc), getattr(exc, "exit_code", 1))
    except VerifyError as exc:
        return fail(log, str(exc), exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
