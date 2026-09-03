"""Piece 3 setup - plumbing tests.

No RPC is contacted here. These cover the .env read/write helpers, the secret
masking, and (if eth-account is installed) the local wallet lifecycle. The real
check is the manual run in the README ("Piece 3 - Polygon Amoy setup").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_setup as cs  # noqa: E402


def test_mask_only_shows_last_4():
    assert cs.mask("0x" + "a" * 60 + "dead") == "...dead"
    assert cs.mask("ab") == "...ab"


def test_set_env_var_creates_file(tmp_path):
    env = tmp_path / ".env"
    cs.set_env_var(env, "WALLET_ADDRESS", "0xABC")
    assert env.read_text(encoding="utf-8") == "WALLET_ADDRESS=0xABC\n"


def test_set_env_var_updates_in_place_and_keeps_others(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LENSO_API_KEY=secret\nWALLET_PRIVATE_KEY=\n# a comment\n", encoding="utf-8")
    cs.set_env_var(env, "WALLET_PRIVATE_KEY", "0xdeadbeef")
    text = env.read_text(encoding="utf-8")
    assert "LENSO_API_KEY=secret" in text
    assert "WALLET_PRIVATE_KEY=0xdeadbeef" in text
    assert "# a comment" in text
    assert text.count("WALLET_PRIVATE_KEY=") == 1


def test_set_env_var_appends_when_absent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LENSO_API_KEY=secret\n", encoding="utf-8")
    cs.set_env_var(env, "WALLET_ADDRESS", "0xABC")
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "LENSO_API_KEY=secret"
    assert lines[-1] == "WALLET_ADDRESS=0xABC"


def test_set_env_var_ignores_commented_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# WALLET_ADDRESS=old\n", encoding="utf-8")
    cs.set_env_var(env, "WALLET_ADDRESS", "0xNEW")
    text = env.read_text(encoding="utf-8")
    assert "# WALLET_ADDRESS=old" in text
    assert "WALLET_ADDRESS=0xNEW" in text


def test_read_env_file_var(tmp_path):
    env = tmp_path / ".env"
    env.write_text('WALLET_ADDRESS="0xABC"\n# WALLET_PRIVATE_KEY=nope\n', encoding="utf-8")
    assert cs.read_env_file_var(env, "WALLET_ADDRESS") == "0xABC"
    assert cs.read_env_file_var(env, "WALLET_PRIVATE_KEY") == ""
    assert cs.read_env_file_var(tmp_path / "missing", "X") == ""


def test_exit_codes_distinct():
    codes = [cs.EXIT_OK, cs.EXIT_ZERO_BALANCE, cs.EXIT_NO_DEPS,
             cs.EXIT_BAD_CONFIG, cs.EXIT_RPC, cs.EXIT_WRONG_CHAIN]
    assert len(codes) == len(set(codes))
    assert cs.AMOY_CHAIN_ID == 80002
    assert cs.RPCError("x").exit_code == cs.EXIT_RPC
    assert cs.WrongChain("x").exit_code == cs.EXIT_WRONG_CHAIN


def test_fmt_pol():
    assert cs._fmt_pol(0, 0) == "0"
    assert cs._fmt_pol(10 ** 18, __import__("decimal").Decimal("1")) == "1"
    assert cs._fmt_pol(5 * 10 ** 17, __import__("decimal").Decimal("0.5")) == "0.5"


# ---- wallet lifecycle (needs eth-account, which ships with web3) -------------
def test_wallet_lifecycle(tmp_path, monkeypatch):
    Account = pytest.importorskip("eth_account").Account
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    monkeypatch.delenv(cs.ADDRESS_ENV, raising=False)
    env = tmp_path / ".env"

    acct1, created1 = cs.load_or_create_wallet(env)
    assert created1 is True
    assert acct1.address.startswith("0x") and len(acct1.address) == 42

    text = env.read_text(encoding="utf-8")
    saved_key = [ln for ln in text.splitlines()
                 if ln.startswith("WALLET_PRIVATE_KEY=")][0].split("=", 1)[1]
    assert saved_key.startswith("0x") and len(saved_key) == 66
    assert Account.from_key(saved_key).address == acct1.address
    assert f"WALLET_ADDRESS={acct1.address}" in text

    # reuse: same wallet, not recreated (read straight from the file)
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    acct2, created2 = cs.load_or_create_wallet(env)
    assert created2 is False
    assert acct2.address == acct1.address

    # --force-new: a different wallet, file rewritten
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    acct3, created3 = cs.load_or_create_wallet(env, force_new=True)
    assert created3 is True
    assert acct3.address != acct1.address
    assert cs.read_env_file_var(env, "WALLET_ADDRESS") == acct3.address


def test_wallet_rejects_bad_key(tmp_path, monkeypatch):
    pytest.importorskip("eth_account")
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    env = tmp_path / ".env"
    env.write_text("WALLET_PRIVATE_KEY=not-a-real-key\n", encoding="utf-8")
    with pytest.raises(cs.BadConfig):
        cs.load_or_create_wallet(env)


def test_load_wallet_never_creates(tmp_path, monkeypatch):
    Account = pytest.importorskip("eth_account").Account
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    env = tmp_path / ".env"

    # no wallet -> BadConfig, and nothing written
    with pytest.raises(cs.BadConfig, match="run .*chain_setup"):
        cs.load_wallet(env)
    assert not env.exists()

    # wallet present -> loads it, no (account, bool) tuple, just the account
    acct = Account.create()
    priv = acct.key.hex()
    priv = priv if priv.startswith("0x") else "0x" + priv
    env.write_text(f"WALLET_PRIVATE_KEY={priv}\n", encoding="utf-8")
    monkeypatch.delenv(cs.PRIV_KEY_ENV, raising=False)
    loaded = cs.load_wallet(env)
    assert loaded.address == acct.address
