"""Shared terminal logging for the HH Goa Task 3 pipeline.

Goal: while any pipeline script runs you can watch the terminal and see exactly
what step it is on, what it found, and - if something breaks - which step failed
and why (a plain sentence, not a raw traceback).

Rules for every script in this project:
  * Log progress to STDERR. STDOUT stays clean for machine-readable output
    (the JSON records piece 1 / piece 2 print, the human report, etc.).
  * Wrap each major operation in `with step(log, "..."):` so its start, end and
    duration are logged, and any exception is reported as
    "FAILED <step> ... <ErrorType>: <message>" before it propagates.
  * Log what you found right after: counts, hashes, URLs, chain tx ids, ...
  * On a fatal error, end `main()` with `return fail(log, msg, EXIT_CODE)`.

Usage:
    from pipeline_log import get_logger, step, fail

    log = get_logger("face_encoder")

    with step(log, "Detecting faces (model=hog)"):
        boxes = detect(...)
    log.info("Detected %d face(s)", len(boxes))

    def main() -> int:
        try:
            ...
        except SomeError as exc:
            return fail(log, f"could not read the image: {exc}", EXIT_BAD_INPUT)

Environment:
    HHGOA_LOG_LEVEL = DEBUG | INFO (default) | WARNING | ERROR
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Iterator

_ROOT = "hhgoa"


def _install_root_handler() -> None:
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return
    # keep stdout clean - all logging goes to stderr
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.set_name("pipeline_log")
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(levelname)-7s %(message)s",
                          datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    level = os.environ.get("HHGOA_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    # best effort: make the arrows / box chars survive a legacy Windows codepage
    for stream in (sys.stderr,):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig:
            try:
                reconfig(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


def get_logger(name: str) -> logging.Logger:
    """Return the logger a script should use, e.g. get_logger("face_encoder")."""
    _install_root_handler()
    return logging.getLogger(f"{_ROOT}.{name}")


@contextlib.contextmanager
def step(log: logging.Logger, description: str) -> Iterator[None]:
    """Log START / DONE (+elapsed) around a block of work.

    If the block raises, log one ERROR line naming the step, the exception type
    and its message - then re-raise unchanged so the caller still handles it.
    """
    log.info("START  %s", description)
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - we log and re-raise
        elapsed = time.monotonic() - started
        log.error("FAILED %s  (%.1fs)  %s: %s",
                  description, elapsed, type(exc).__name__, exc)
        raise
    elapsed = time.monotonic() - started
    log.info("DONE   %s  (%.1fs)", description, elapsed)


def fail(log: logging.Logger, message: str, exit_code: int) -> int:
    """Log a final, single-line failure summary and return `exit_code`.

    Use at the end of main():  return fail(log, "reason ...", EXIT_BAD_INPUT)
    """
    log.error("ABORTED (exit %d): %s", exit_code, message)
    return exit_code


def note_exception(log: logging.Logger, prefix: str, exc: BaseException) -> None:
    """Log an exception as a readable line (type + message), no traceback."""
    log.error("%s  %s: %s", prefix, type(exc).__name__, exc)
