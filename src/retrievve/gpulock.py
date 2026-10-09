"""Shared Mac GPU lock (docs/experiments/cloud_budget.md §4b). Every mlx / ollama / embedding job takes it, so two GPU
jobs never run at once on the 36 GB Mac.

Mechanism: an advisory kernel lock (`fcntl.flock`, LOCK_EX) on the persistent file `~/retrievve-runs/GPU.lock`.
The kernel releases it when the holding process exits for ANY reason, SIGKILL included, so there is no stale lock to
detect or break (no break race, no deadlock on a half-written holder file). The file's JSON content (owner, job, pid,
host, started_at) is informational only, for `status`. The pre-2026-10-10 convention (a DIRECTORY `GPU.lock/` with an
`owner` file) is still respected: while such a directory exists and its pid is alive, nobody acquires.

Python:
    from retrievve.gpulock import gpu_lock
    with gpu_lock(owner="retrievve-32", job="nt-realign embeddings"):
        ...
Shell (holds the lock for the whole child process GROUP; leftover grandchildren are terminated before release;
a separate acquire/release pair can't work with a kernel lock). Send the runner SIGTERM, not SIGKILL, so it can reap:
    python -m retrievve.gpulock run --owner retrievve-32 --job "nt embeddings" -- bash scripts/gpu_job.sh
    python -m retrievve.gpulock status
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

LOCK_PATH = Path(os.environ.get("RETRIEVVE_GPU_LOCK", Path.home() / "retrievve-runs" / "GPU.lock"))

_held: dict[str, int] = {}                     # path -> open fd holding the flock (this process)


class GpuBusy(RuntimeError):
    """Raised when the lock is held and the caller doesn't wait (or the wait times out)."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _legacy_holder(path: Path) -> dict | None:
    """A live legacy directory lock (GPU.lock/owner with 'pid N'), or None. A dead-pid legacy dir is removed."""
    if not path.is_dir():
        return None
    owner = path / "owner"
    text = owner.read_text(encoding="utf-8", errors="replace") if owner.exists() else ""
    m = re.search(r"pid\D{0,3}(\d+)", text)
    pid = int(m.group(1)) if m else None
    if pid is not None and not _alive(pid):
        for p in path.iterdir():
            p.unlink()
        path.rmdir()
        return None
    return {"legacy_dir": True, "pid": pid, "raw": text.strip()}


def _try_lock(path: Path) -> int | None:
    """Open + non-blocking flock. Returns the fd on success, None if held by someone else."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    try:                                        # the path must still name the inode we locked (not unlinked/replaced
        same = os.fstat(fd).st_ino == os.stat(path).st_ino   # between open and flock, e.g. by a legacy `rm`)
    except FileNotFoundError:
        same = False
    if not same:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return _try_lock(path)
    return fd


def read_holder(path: Path | None = None) -> dict | None:
    """Current holder info, or None if free. Never takes the lock for longer than the probe."""
    path = Path(path or LOCK_PATH)
    legacy = _legacy_holder(path)
    if legacy or path.is_dir():
        return legacy or {"legacy_dir": True, "pid": None}
    if not path.exists():
        return None
    if str(path) in _held:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    fd = _try_lock(path)
    if fd is not None:                          # nobody holds it
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}") or {"pid": None, "note": "holder info not written"}
    except json.JSONDecodeError:
        return {"pid": None, "note": "holder info unreadable"}


def acquire(owner: str, job: str, *, wait: bool = True, poll_s: float = 5.0, timeout_s: float | None = None,
            path: Path | None = None) -> dict:
    """Take the lock (blocking by default). Re-entrant use within one process is refused."""
    path = Path(path or LOCK_PATH)
    if str(path) in _held:
        raise RuntimeError(f"{path} is already held by this process")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        fd = None if _legacy_holder(path) or path.is_dir() else _try_lock(path)
        if fd is not None:
            break
        if not wait or (timeout_s is not None and time.monotonic() - t0 > timeout_s):
            raise GpuBusy(f"GPU lock {path} held by {read_holder(path)}")
        time.sleep(poll_s)
    me = {"owner": owner, "job": job, "pid": os.getpid(), "host": socket.gethostname(),
          "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps(me, ensure_ascii=False).encode(), 0)
    os.fsync(fd)
    _held[str(path)] = fd
    return me


def release(path: Path | None = None) -> None:
    """Release this process's lock (no-op if we don't hold it). Holder info is cleared before unlocking."""
    path = Path(path or LOCK_PATH)
    fd = _held.pop(str(path), None)
    if fd is None:
        return
    try:
        os.ftruncate(fd, 0)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def gpu_lock(owner: str, job: str, **kw):
    """Held for the block; released on exit and exceptions. If the process is killed, the kernel releases it."""
    path = Path(kw.pop("path", None) or LOCK_PATH)
    acquire(owner, job, path=path, **kw)
    try:
        yield
    finally:
        release(path)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_group(pgid: int, grace_s: float = 10.0) -> None:
    """Terminate every process left in the job's group (grandchildren from uv run, pipelines, ...): TERM, then KILL."""
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 2.0)):
        if not _group_alive(pgid):
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
        t0 = time.monotonic()
        while _group_alive(pgid) and time.monotonic() - t0 < wait_s:
            time.sleep(0.1)


def _run(owner: str, job: str, cmd: list[str], timeout_s: float | None, grace_s: float = 10.0) -> int:
    """Run `cmd` holding the lock. The child gets its own process group; on exit or signal, the WHOLE group is
    terminated before the lock is released, so no grandchild can keep using the GPU after release."""
    with gpu_lock(owner, job, timeout_s=timeout_s):
        child = subprocess.Popen(cmd, start_new_session=True)
        pgid = child.pid

        def fwd(sig, _frame):                   # forward SIGTERM/SIGINT to the whole job group
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, sig)
        old = {s: signal.signal(s, fwd) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            return child.wait()
        finally:
            _reap_group(pgid, grace_s)          # before the lock is released (gpu_lock's finally runs after this)
            for s, h in old.items():
                signal.signal(s, h)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m retrievve.gpulock")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    r = sub.add_parser("run", help="hold the GPU lock while running a command: run --owner X --job Y -- cmd ...")
    r.add_argument("--owner", required=True)
    r.add_argument("--job", required=True)
    r.add_argument("--timeout", type=float, default=None, help="max seconds to wait for the lock")
    r.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            ap.error("run needs a command after --")
        try:
            return _run(a.owner, a.job, cmd, a.timeout)
        except GpuBusy as e:
            print(e, file=sys.stderr)
            return 75                           # EX_TEMPFAIL
    holder = read_holder()
    print("GPU lock: free" if holder is None else f"GPU lock: held: {json.dumps(holder, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
