"""Piece 3 (setup) - Polygon Amoy testnet wallet + RPC connectivity.

Generates a throwaway wallet ONCE, connects to the Polygon Amoy testnet with
web3.py, and reports the wallet's POL balance so you can fund it from a faucet.
Later piece-3 code reads WALLET_PRIVATE_KEY from .env to sign transactions.

  Network  : Polygon Amoy testnet
  Chain ID : 80002
  RPC      : https://rpc-amoy.polygon.technology   (override via AMOY_RPC_URL)
  Explorer : https://amoy.polygonscan.com
  Faucet   : https://bwarelabs.com/faucets/polygon-testnet

Secrets: the private key is generated locally, written ONLY to .env (which is
gitignored), never hard-coded and never printed in full - the terminal shows
just the last 4 characters for confirmation. The wallet ADDRESS is public and
safe to share (paste it into a faucet).

Handled failure modes:
  * web3 / eth-account not installed        -> exit 3
  * bad config (bad key in .env, etc.)      -> exit 4
  * cannot reach the Amoy RPC               -> exit 5
  * RPC is on the wrong chain               -> exit 6
  * wallet balance is 0 (needs a faucet)    -> exit 2  (not an error)

Usage:
    python src/chain_setup.py
    python src/chain_setup.py --rpc-url https://polygon-amoy.g.alchemy.com/v2/KEY
    python src/chain_setup.py --force-new        # replace the wallet in .env
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_log import fail, get_logger, step

log = get_logger("chain_setup")

AMOY_CHAIN_ID = 80002
DEFAULT_RPC_URL = "https://rpc-amoy.polygon.technology"
FAUCET_URL = "https://bwarelabs.com/faucets/polygon-testnet"
EXPLORER_URL = "https://amoy.polygonscan.com"

PRIV_KEY_ENV = "WALLET_PRIVATE_KEY"
ADDRESS_ENV = "WALLET_ADDRESS"
RPC_ENV = "AMOY_RPC_URL"

EXIT_OK = 0
EXIT_ZERO_BALANCE = 2
EXIT_NO_DEPS = 3
EXIT_BAD_CONFIG = 4
EXIT_RPC = 5
EXIT_WRONG_CHAIN = 6


class ChainSetupError(Exception):
    exit_code = 1


class MissingDeps(ChainSetupError):
    exit_code = EXIT_NO_DEPS


class BadConfig(ChainSetupError):
    exit_code = EXIT_BAD_CONFIG


class RPCError(ChainSetupError):
    exit_code = EXIT_RPC


class WrongChain(ChainSetupError):
    exit_code = EXIT_WRONG_CHAIN


# --------------------------------------------------------------------------
# .env helpers  (dependency-free, mirrors src/face_search.py)
# --------------------------------------------------------------------------
def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from `path` into os.environ (existing vars win)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def read_env_file_var(path: str | Path, key: str) -> str:
    """Return the value of `key` straight from the .env file (not os.environ)."""
    p = Path(path)
    if not p.is_file():
        return ""
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def set_env_var(path: str | Path, key: str, value: str) -> None:
    """Create or update an uncommented `key=value` line in the .env file at
    `path`, leaving every other line untouched."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask(secret: str) -> str:
    """Display-safe tail of a secret, e.g. '...1a2b'. Never log the whole thing."""
    tail = secret[-4:] if len(secret) >= 4 else secret
    return f"...{tail}"


# --------------------------------------------------------------------------
# wallet
# --------------------------------------------------------------------------
def _account_module():
    try:
        from eth_account import Account
    except ImportError as exc:
        raise MissingDeps("eth-account is not installed - run: pip install web3") from exc
    return Account


def _existing_key(env_path: str | Path) -> str:
    return (read_env_file_var(env_path, PRIV_KEY_ENV)
            or os.environ.get(PRIV_KEY_ENV, "")).strip()


def load_wallet(env_path: str | Path = ".env"):
    """Load the wallet already in `env_path`. Raises BadConfig if there is none
    (never creates one - use for read/spend paths so we don't silently spin up
    a fresh, empty wallet)."""
    Account = _account_module()
    existing = _existing_key(env_path)
    if not existing:
        raise BadConfig(
            f"no wallet found ({PRIV_KEY_ENV} not set in {env_path}) - "
            f"run `python src/chain_setup.py` first"
        )
    try:
        acct = Account.from_key(existing)
    except Exception as exc:
        raise BadConfig(
            f"{PRIV_KEY_ENV} in {env_path} is not a valid private key ({exc})"
        ) from exc
    log.info("Loaded wallet from %s  (private key %s)", env_path, mask(existing))
    return acct


def load_or_create_wallet(env_path: str | Path = ".env", force_new: bool = False):
    """Return (account, created: bool).

    Reuses WALLET_PRIVATE_KEY from the .env file if present - we never
    regenerate over a funded wallet - unless force_new is set. This is the
    chain_setup entry point; read/spend code should call load_wallet() instead.
    """
    Account = _account_module()
    existing = _existing_key(env_path)

    if existing and not force_new:
        try:
            acct = Account.from_key(existing)
        except Exception as exc:
            raise BadConfig(
                f"{PRIV_KEY_ENV} in {env_path} is not a valid private key ({exc}) - "
                f"fix that line or pass --force-new"
            ) from exc
        log.info("Using existing wallet from %s  (private key %s)", env_path, mask(existing))
        return acct, False

    if existing and force_new:
        log.warning("--force-new: REPLACING the wallet in %s (old key %s); any funds on "
                    "the old address will be abandoned", env_path, mask(existing))

    with step(log, "Generating a new wallet (eth_account)"):
        acct = Account.create()
    priv = acct.key.hex()
    if not priv.startswith("0x"):
        priv = "0x" + priv

    try:
        with step(log, f"Saving private key + address to {env_path}"):
            set_env_var(env_path, PRIV_KEY_ENV, priv)
            set_env_var(env_path, ADDRESS_ENV, acct.address)
    except OSError as exc:
        raise BadConfig(f"could not write the wallet to {env_path}: {exc}") from exc
    os.environ[PRIV_KEY_ENV] = priv
    os.environ[ADDRESS_ENV] = acct.address
    log.info("New wallet saved to %s  (private key %s - never shown in full)",
             env_path, mask(priv))
    return acct, True


# --------------------------------------------------------------------------
# RPC
# --------------------------------------------------------------------------
def _inject_poa_middleware(w3) -> None:
    """Polygon is proof-of-authority; block/tx decoding needs a POA shim for
    later pieces. Handles web3.py v7 and v6 names; harmless if it can't."""
    try:
        from web3.middleware import ExtraDataToPOAMiddleware  # web3 >= 7

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return
    except Exception:
        pass
    try:
        from web3.middleware import geth_poa_middleware  # web3 6.x

        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    except Exception:
        log.debug("POA middleware not injected (not needed for a balance read)")


def connect(rpc_url: str, timeout: float = 20.0):
    """Return a connected Web3 pointed at Polygon Amoy, or raise ChainSetupError."""
    try:
        from web3 import Web3
    except ImportError as exc:
        raise MissingDeps("web3 is not installed - run: pip install web3") from exc

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
    _inject_poa_middleware(w3)

    with step(log, f"Connecting to Polygon Amoy RPC ({rpc_url})"):
        try:
            connected = w3.is_connected()
        except Exception as exc:
            raise RPCError(f"could not reach the Amoy RPC at {rpc_url}: {exc}") from exc
        if not connected:
            raise RPCError(
                f"could not reach the Amoy RPC at {rpc_url} - check the URL and your "
                f"internet connection (or set {RPC_ENV} to a private endpoint)"
            )

    try:
        chain_id = w3.eth.chain_id
    except Exception as exc:
        raise RPCError(
            f"connected to {rpc_url} but it did not answer eth_chainId: {exc}"
        ) from exc

    client = ""
    try:
        client = w3.client_version
    except Exception:
        pass
    log.info("Connected: chain id %d%s", chain_id, f", node {client!r}" if client else "")

    if chain_id != AMOY_CHAIN_ID:
        raise WrongChain(
            f"the RPC at {rpc_url} reports chain id {chain_id}, but Polygon Amoy is "
            f"{AMOY_CHAIN_ID} - this is the wrong RPC URL"
        )
    return w3


def check_balance(w3, address: str) -> tuple[int, object]:
    """Return (wei, pol) for `address`. Raises RPCError on an RPC failure."""
    with step(log, f"Checking POL balance of {address}"):
        try:
            wei = w3.eth.get_balance(address)
        except Exception as exc:
            raise RPCError(f"eth_getBalance failed for {address}: {exc}") from exc
    pol = w3.from_wei(wei, "ether")
    log.info("Balance: %s POL  (%d wei)", _fmt_pol(wei, pol), wei)
    return wei, pol


def _fmt_pol(wei: int, pol) -> str:
    if wei == 0:
        return "0"
    return f"{pol:.18f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chain_setup",
        description="Generate a Polygon Amoy testnet wallet and check its POL balance "
                    "(piece 3 setup).",
    )
    p.add_argument("--env-file", default=".env",
                   help="dotenv file to read/write the wallet key (default: .env)")
    p.add_argument("--rpc-url", default=None,
                   help=f"Amoy RPC endpoint (default: ${RPC_ENV} or {DEFAULT_RPC_URL})")
    p.add_argument("--timeout", type=float, default=20.0,
                   help="RPC request timeout in seconds (default: 20)")
    p.add_argument("--force-new", action="store_true",
                   help="generate a fresh wallet even if one is already in the env file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.info("chain_setup starting  (Polygon Amoy testnet, chain id %d)", AMOY_CHAIN_ID)

    load_dotenv(args.env_file)
    rpc_url = args.rpc_url or os.environ.get(RPC_ENV) or DEFAULT_RPC_URL

    try:
        acct, created = load_or_create_wallet(args.env_file, force_new=args.force_new)
        w3 = connect(rpc_url, timeout=args.timeout)
        wei, pol = check_balance(w3, acct.address)
    except ChainSetupError as exc:
        return fail(log, str(exc), exc.exit_code)

    # The address is PUBLIC and safe to share - print it plainly on stdout.
    print(acct.address)
    log.info("Wallet address (public, safe to share): %s", acct.address)
    log.info("Explorer: %s/address/%s", EXPLORER_URL, acct.address)

    if wei == 0:
        log.warning("Wallet holds 0 POL - fund it before piece 3 can write to chain")
        print()
        print(f"Get free testnet POL: paste this address into {FAUCET_URL}")
        print(f"    {acct.address}")
        return EXIT_ZERO_BALANCE

    log.info("chain_setup done: wallet funded with %s POL  (exit 0)", _fmt_pol(wei, pol))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
