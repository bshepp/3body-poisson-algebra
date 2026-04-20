"""Subprocess watchdog: hard wall-clock timeout + resident-set-size cap.

Used to safely babysit the simplify experiment workers so they can't run
away with the machine the way this morning's L3 Schwarzschild sweep did.

Public API: ``run_with_limits(args, timeout_s, ram_cap_gb, ...) -> dict``.

Returns one of:
  {"status": "done",               "exit_code": int, "elapsed_s": float, "peak_rss_gb": float}
  {"status": "timeout",            "elapsed_s": float, "peak_rss_gb": float}
  {"status": "ram_cap_exceeded",   "elapsed_s": float, "peak_rss_gb": float, "rss_gb": float}
  {"status": "launch_failed",      "error": str}

Polling interval defaults to 5 s. We measure RSS for the launched process
plus all of its descendants (relevant because SymPy's compute_growth can
spawn worker processes via multiprocessing in some code paths).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import psutil


def _tree_rss_gb(proc: psutil.Process) -> float:
    """Sum RSS over a process and all its living descendants."""
    total = 0
    try:
        with proc.oneshot():
            total += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                with child.oneshot():
                    total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total / (2 ** 30)


def _kill_tree(proc: psutil.Process, grace_s: float = 5.0) -> None:
    """Terminate a process and all descendants, then escalate to kill."""
    try:
        children = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    targets = children + [proc]
    for t in targets:
        try:
            t.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(targets, timeout=grace_s)
    for t in alive:
        try:
            t.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def run_with_limits(
    args: Sequence[str],
    *,
    timeout_s: float,
    ram_cap_gb: float,
    poll_s: float = 5.0,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
    log_path: str | os.PathLike | None = None,
    on_tick=None,
) -> dict:
    """Launch ``args`` as a subprocess and enforce wall-clock + RSS limits.

    Parameters
    ----------
    args
        Command and arguments, e.g. ``["python", "worker.py", "cancel", "schwarz_l2", "out.json"]``.
    timeout_s
        Wall-clock kill threshold. Float seconds.
    ram_cap_gb
        Total RSS (process + descendants) above which we kill. Float GB.
    poll_s
        How often to check RSS / wall clock. 5s default; lower means more accurate
        cap enforcement but more polling overhead.
    cwd, env
        Forwarded to ``subprocess.Popen``.
    log_path
        If provided, the subprocess's stdout+stderr is appended to this file
        instead of inheriting the parent's streams.
    on_tick
        Optional callable ``f(elapsed_s, rss_gb) -> None`` invoked on every poll.
        Useful for live progress logging from the orchestrator.
    """
    log_handle = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        stdout = log_handle
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None

    t0 = time.perf_counter()
    try:
        popen = subprocess.Popen(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    except (OSError, ValueError) as exc:
        if log_handle:
            log_handle.close()
        return {"status": "launch_failed", "error": str(exc)}

    try:
        proc = psutil.Process(popen.pid)
    except psutil.NoSuchProcess:
        if log_handle:
            log_handle.close()
        return {
            "status": "launch_failed",
            "error": f"process {popen.pid} vanished immediately",
        }

    peak_rss_gb = 0.0
    deadline = t0 + timeout_s

    try:
        while True:
            ret = popen.poll()
            if ret is not None:
                elapsed = time.perf_counter() - t0
                return {
                    "status": "done",
                    "exit_code": int(ret),
                    "elapsed_s": elapsed,
                    "peak_rss_gb": peak_rss_gb,
                }

            now = time.perf_counter()
            elapsed = now - t0

            try:
                rss_gb = _tree_rss_gb(proc)
            except Exception:
                rss_gb = 0.0
            if rss_gb > peak_rss_gb:
                peak_rss_gb = rss_gb

            if on_tick:
                try:
                    on_tick(elapsed, rss_gb)
                except Exception:
                    pass

            if now >= deadline:
                _kill_tree(proc)
                popen.wait(timeout=10)
                return {
                    "status": "timeout",
                    "elapsed_s": elapsed,
                    "peak_rss_gb": peak_rss_gb,
                }

            if rss_gb > ram_cap_gb:
                _kill_tree(proc)
                popen.wait(timeout=10)
                return {
                    "status": "ram_cap_exceeded",
                    "elapsed_s": elapsed,
                    "peak_rss_gb": peak_rss_gb,
                    "rss_gb": rss_gb,
                }

            time.sleep(poll_s)
    except KeyboardInterrupt:
        _kill_tree(proc)
        popen.wait(timeout=10)
        elapsed = time.perf_counter() - t0
        return {
            "status": "interrupted",
            "elapsed_s": elapsed,
            "peak_rss_gb": peak_rss_gb,
        }
    finally:
        if log_handle:
            log_handle.flush()
            log_handle.close()


# --------------------------------------------------------------------------- #
# Quick self-test (manual)
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    """Run two tiny self-tests: a fast success and a forced timeout."""
    print("Self-test 1: trivial command exits cleanly")
    r1 = run_with_limits(
        [sys.executable, "-c", "print('hello'); import time; time.sleep(0.5)"],
        timeout_s=10, ram_cap_gb=1.0, poll_s=0.5,
    )
    print(f"  result: {r1}")
    assert r1["status"] == "done" and r1["exit_code"] == 0

    print("\nSelf-test 2: hung command hits the timeout")
    r2 = run_with_limits(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=2.0, ram_cap_gb=1.0, poll_s=0.5,
    )
    print(f"  result: {r2}")
    assert r2["status"] == "timeout"

    print("\nSelf-test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
