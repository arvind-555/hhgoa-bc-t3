"""Tests for the shared pipeline_log helper."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline_log as pl  # noqa: E402


def test_get_logger_is_stderr_and_singleton():
    a = pl.get_logger("thing")
    b = pl.get_logger("another")
    assert a is pl.get_logger("thing")          # same name -> same logger
    assert a.name == "hhgoa.thing" and b.name == "hhgoa.another"
    root = logging.getLogger("hhgoa")
    assert root.propagate is False
    # exactly one handler installed by pipeline_log (pytest may add its own),
    # and repeated get_logger() calls don't add more
    ours = [h for h in root.handlers if h.get_name() == "pipeline_log"]
    assert len(ours) == 1 and isinstance(ours[0], logging.StreamHandler)
    pl.get_logger("thing")
    pl.get_logger("x")
    assert len([h for h in root.handlers if h.get_name() == "pipeline_log"]) == 1


def test_step_logs_start_and_done(caplog):
    log = pl.get_logger("t1")
    with caplog.at_level(logging.INFO, logger="hhgoa.t1"):
        with pl.step(log, "Doing a thing"):
            pass
    msgs = [r.message for r in caplog.records]
    assert any(m.startswith("START  Doing a thing") for m in msgs)
    assert any(m.startswith("DONE   Doing a thing") for m in msgs)


def test_step_logs_failure_and_reraises(caplog):
    log = pl.get_logger("t2")
    with caplog.at_level(logging.INFO, logger="hhgoa.t2"):
        with pytest.raises(ValueError):
            with pl.step(log, "Risky step"):
                raise ValueError("boom")
    err = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert err and "FAILED Risky step" in err[0].message
    assert "ValueError: boom" in err[0].message


def test_fail_returns_code_and_logs(caplog):
    log = pl.get_logger("t3")
    with caplog.at_level(logging.ERROR, logger="hhgoa.t3"):
        code = pl.fail(log, "could not do X because Y", 4)
    assert code == 4
    assert "ABORTED (exit 4): could not do X because Y" in caplog.records[-1].message
