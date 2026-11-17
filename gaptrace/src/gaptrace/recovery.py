"""Durable checkpoints, bounded recovery, and exclusive research runs."""
from __future__ import annotations

import errno
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError
from uuid import uuid4


class TransientFailure(RuntimeError):
    """An interrupted request that can be repeated without changing its inputs."""


def transient(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, HTTPError):
            return exc.code in {408, 429} or 500 <= exc.code < 600
        if isinstance(exc, (TransientFailure, TimeoutError, ConnectionError, URLError, IncompleteRead)):
            return True
        if isinstance(exc, OSError) and (exc.errno in {
            errno.ECONNREFUSED, errno.ECONNRESET, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT,
        } or getattr(exc, "winerror", None) in {10054, 10060, 10061, 10065}):
            return True
        exc = exc.__cause__
    return False


def recover(operation, stage, retries=2, on_retry=None):
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not transient(exc) or attempt == retries:
                raise
            event = {"stage": stage, "attempt": attempt + 1, "max_retries": retries,
                     "delay_seconds": min(2 ** (attempt + 1), 30),
                     "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat()}
            if on_retry:
                on_retry(event)
            sleep(event["delay_seconds"])


def atomic_json(path, value):
    _atomic_write(path, lambda stream: json.dump(value, stream, ensure_ascii=False, indent=2))


def atomic_text(path, value):
    _atomic_write(path, lambda stream: stream.write(value))


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_checkpoint(path, payload):
    checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    atomic_json(path, {"checkpoint_version": 1, "sha256": checksum, "payload": payload})


def read_checkpoint(path, validate=lambda payload: payload):
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "checkpoint_version" in payload:
            if payload["checkpoint_version"] != 1:
                raise ValueError("Unknown checkpoint version")
            value = payload["payload"]
            checksum = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            if checksum != payload["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            payload = value
        return validate(payload)
    except (ValueError, KeyError, TypeError, AttributeError):
        # Keep damaged files for inspection. Rebuild only this derived cache entry.
        path.replace(path.with_name(path.name + ".corrupt-" + uuid4().hex[:8]))
        return None


@contextmanager
def run_lock(destination):
    destination = Path(destination).resolve()
    directory = destination.parent / ".gaptrace-locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(destination).casefold().encode()).hexdigest()
    with (directory / f"{key}.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"This research directory is already running: {destination}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
