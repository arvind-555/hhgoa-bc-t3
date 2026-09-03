"""Demo: prove the blockchain step is real, not decorative.

    1. Take a sample discovered post, notarize it on Polygon Amoy (upload_hash).
    2. verify() it -> should PASS.
    3. Tamper one field of the post.
    4. verify() the tampered post against the SAME transaction -> should FAIL.

Run from the repo root:

    python demo_tamper_verify.py

If the wallet has 0 POL the on-chain round-trip is skipped, but the demo still
shows the core guarantee: a one-character change to the post yields a totally
different SHA-256, so tampering is detectable. Fund the wallet
(`python src/chain_setup.py`) and re-run for the full on-chain proof.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chain_setup import ChainSetupError  # noqa: E402
from chain_verify import hash_post, upload_hash, verify  # noqa: E402
from pipeline_log import get_logger  # noqa: E402

log = get_logger("demo_tamper_verify")

# A stand-in for what piece 2 hands us: one social post that matched the face.
SAMPLE_POST = {
    "platform": "example.social",
    "post_url": "https://example.social/@dana/posts/8842",
    "author_handle": "dana",
    "caption": "Sunset at Anjuna beach \U0001f305 #goa2026",
    "posted_utc": "2026-01-14T17:32:00Z",
    "image_sha256": "0x" + "ab" * 32,
    "match_confidence": 0.94,
    "discovered_utc": "2026-09-03T09:00:00Z",
}


def _tamper(post: dict) -> dict:
    """Change exactly one field - the kind of edit an attacker would make."""
    bad = copy.deepcopy(post)
    bad["match_confidence"] = 0.99      # inflate the confidence after the fact
    return bad


def main() -> int:
    log.info("=== Tamper-verify demo: is the blockchain step real? ===")

    honest = SAMPLE_POST
    tampered = _tamper(honest)
    log.info("Honest post hash   : %s", hash_post(honest))
    log.info("Tampered post hash : %s", hash_post(tampered))
    if hash_post(honest) == hash_post(tampered):
        log.error("Hashes are equal - the tamper wasn't detectable. Demo is broken.")
        return 1
    log.info("One field changed -> a completely different hash. Good.")

    log.info("--- Step 1: notarize the honest post on Polygon Amoy ---")
    try:
        result = upload_hash(honest)
    except ChainSetupError as exc:
        log.warning("Could not reach the chain: %s", exc)
        log.warning("The hash-level tamper check above still holds. Once the RPC is")
        log.warning("reachable and the wallet is funded, re-run for the on-chain proof.")
        return 2

    if not result["broadcast"]:
        log.warning("On-chain round-trip skipped: %s", result.get("reason"))
        log.warning("The hash-level tamper check above still holds. Fund the wallet")
        log.warning("(python src/chain_setup.py) and re-run for the full proof.")
        return 2

    tx_hash = result["tx_hash"]
    log.info("Notarized. tx=%s", tx_hash)
    log.info("Polygonscan: %s", result["polygonscan_url"])

    log.info("--- Step 2: verify the honest post against the tx (expect PASS) ---")
    ok_honest = verify(honest, tx_hash)

    log.info("--- Step 3: verify the TAMPERED post against the same tx (expect FAIL) ---")
    ok_tampered = verify(tampered, tx_hash)

    log.info("=== Result ===")
    log.info("honest  post verifies : %s   (want True)", ok_honest)
    log.info("tampered post verifies: %s   (want False)", ok_tampered)

    if ok_honest and not ok_tampered:
        log.info("DEMO PASSED - honest data verifies, tampered data is rejected on-chain.")
        return 0
    log.error("DEMO FAILED - the blockchain step did not behave as claimed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
