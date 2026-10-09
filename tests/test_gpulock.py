"""Shared GPU lock (kernel flock): mutual exclusion across processes, auto-release on SIGKILL, no deadlock on an
empty/garbage lock file, legacy directory lock respected, `run` CLI holds the lock for the child's lifetime."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from retrievve import gpulock

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _holder_proc(path, sleep_s=30):
    """A separate process that takes the lock and sleeps (prints READY once held)."""
    code = textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {SRC!r})
        from pathlib import Path
        from retrievve import gpulock
        gpulock.acquire("holder", "test", path=Path({str(path)!r}), wait=False)
        print("READY", flush=True); time.sleep({sleep_s})
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "READY"
    return p


def test_roundtrip_and_holder_info(tmp_path):
    p = tmp_path / "GPU.lock"
    me = gpulock.acquire("t", "job", path=p, wait=False)
    assert json.loads(p.read_text())["pid"] == os.getpid() == me["pid"]
    gpulock.release(p)
    assert gpulock.read_holder(p) is None


def test_other_process_blocks_then_sigkill_releases(tmp_path):
    p = tmp_path / "GPU.lock"
    h = _holder_proc(p)
    try:
        with pytest.raises(gpulock.GpuBusy):
            gpulock.acquire("t", "job", path=p, wait=False)
        assert gpulock.read_holder(p)["owner"] == "holder"
    finally:
        h.send_signal(signal.SIGKILL)
        h.wait()
    gpulock.acquire("t", "job", path=p, wait=False)        # kernel released it: no stale lock to break
    gpulock.release(p)


def test_empty_or_garbage_lock_file_never_deadlocks(tmp_path):
    p = tmp_path / "GPU.lock"
    p.write_text("")                                       # e.g. a creator killed before writing holder info
    gpulock.acquire("t", "job", path=p, wait=False)
    gpulock.release(p)
    p.write_text("{not json")
    gpulock.acquire("t", "job", path=p, wait=False)
    gpulock.release(p)


def test_live_legacy_directory_lock_is_respected_dead_one_removed(tmp_path):
    p = tmp_path / "GPU.lock"
    p.mkdir()
    (p / "owner").write_text(f"owner=codex pid {os.getppid()}")
    with pytest.raises(gpulock.GpuBusy):
        gpulock.acquire("t", "job", path=p, wait=False)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (p / "owner").write_text(f"owner=codex pid {dead.pid}")
    gpulock.acquire("t", "job", path=p, wait=False)       # dead legacy holder removed
    assert p.is_file()
    gpulock.release(p)


def test_release_only_affects_own_lock_and_context_manager(tmp_path):
    p = tmp_path / "GPU.lock"
    gpulock.release(p)                                     # not held: no-op
    with pytest.raises(ValueError):
        with gpulock.gpu_lock("t", "job", path=p, wait=False):
            raise ValueError("boom")
    assert gpulock.read_holder(p) is None


def test_run_cli_holds_lock_for_child(tmp_path, monkeypatch):
    p = tmp_path / "GPU.lock"
    monkeypatch.setattr(gpulock, "LOCK_PATH", p)
    probe = tmp_path / "probe.json"
    child = (f"import sys,json; sys.path.insert(0,{SRC!r}); from pathlib import Path; from retrievve import gpulock; "
             f"open({str(probe)!r},'w').write(json.dumps(gpulock.read_holder(Path({str(p)!r}))))")
    rc = gpulock.main(["run", "--owner", "cli", "--job", "probe", "--", sys.executable, "-c", child])
    assert rc == 0
    assert json.loads(probe.read_text())["owner"] == "cli"   # held while the child ran
    assert gpulock.read_holder(p) is None                     # released after


def test_contention_is_mutually_exclusive(tmp_path):
    """Two processes racing: never both inside the critical section."""
    p, log = tmp_path / "GPU.lock", tmp_path / "log"
    code = textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {SRC!r})
        from pathlib import Path
        from retrievve import gpulock
        for _ in range(5):
            with gpulock.gpu_lock("w", "race", path=Path({str(p)!r}), poll_s=0.01):
                open({str(log)!r}, "a").write("in\\n"); time.sleep(0.02); open({str(log)!r}, "a").write("out\\n")
    """)
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(3)]
    assert all(pr.wait(timeout=60) == 0 for pr in procs)
    lines = log.read_text().split()
    assert lines == ["in", "out"] * 15                     # strictly alternating: no overlap
    time.sleep(0)


def test_lock_on_replaced_inode_is_not_kept(tmp_path, monkeypatch):
    """If the file is unlinked/recreated between open and flock, _try_lock re-locks the current inode."""
    p = tmp_path / "GPU.lock"
    real_flock = gpulock.fcntl.flock
    calls = {"n": 0}

    def flock_then_replace(fd, op):
        real_flock(fd, op)
        if op & gpulock.fcntl.LOCK_EX and calls["n"] == 0:
            calls["n"] += 1
            p.unlink()
            p.write_text("")
    monkeypatch.setattr(gpulock.fcntl, "flock", flock_then_replace)
    fd = gpulock._try_lock(p)
    assert fd is not None and os.fstat(fd).st_ino == os.stat(p).st_ino
    real_flock(fd, gpulock.fcntl.LOCK_UN)
    os.close(fd)


def test_run_reaps_grandchildren_before_release(tmp_path, monkeypatch):
    """A child that spawns a long-lived grandchild and exits: the grandchild is terminated before the lock is freed."""
    p = tmp_path / "GPU.lock"
    monkeypatch.setattr(gpulock, "LOCK_PATH", p)
    pidfile = tmp_path / "gc.pid"
    child = (f"import subprocess,sys; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
             f"open({str(pidfile)!r},'w').write(str(g.pid))")
    rc = gpulock.main(["run", "--owner", "cli", "--job", "gc", "--", sys.executable, "-c", child])
    assert rc == 0
    gc = int(pidfile.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(gc, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(gc, signal.SIGKILL)
        pytest.fail("grandchild survived the runner")
    assert gpulock.read_holder(p) is None
