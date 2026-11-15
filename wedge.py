"""Detect a wedged llama-server and kill it, so the next turn works.

Background (`wedge-hunt`, TEAM-LOG 2026-08-17 23:55): when Ollama "freezes",
`ollama serve` is entirely healthy. The stall lives inside ONE `llama-server.exe`
child, and it is never spontaneous — every observed recovery was a restart. The
correct recovery is therefore to kill that child, NOT to restart `ollama serve`:
Ollama respawns a runner on the next request.

The probe that makes this safe is /health vs /slots. In llama.cpp's server,
/health is answered by the HTTP thread directly, while /slots posts a task to
the INFERENCE task queue and waits for it. So:

    /health fast + /slots hanging  →  the inference queue is stuck   → wedged
    /health fast + /slots 200      →  the server is fine, even if busy

That second line is measured, not assumed: `/slots` was probed 8 times DURING a
live generation and answered 200 in 0-47 ms every time, with
`is_processing=True`. A busy server looks nothing like a wedged one, which is
what makes killing on this signal safe rather than a coin flip.

Nothing here may raise and nothing may block for long: it runs on the failure
path of a voice turn, where a second fault would replace a recoverable stall
with a dead assistant.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import urllib.error
import urllib.request

import config

log = logging.getLogger("bastion.wedge")

_PROBE_TIMEOUT_S = float(getattr(config, "WEDGE_PROBE_TIMEOUT_S", 5.0))
_PS_TIMEOUT_S = float(getattr(config, "WEDGE_PS_TIMEOUT_S", 15.0))

# Windows creation flag: no console window for the PowerShell helpers.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ps(command: str, timeout: float) -> str:
    """Run a PowerShell one-liner and return stdout, or "" on any failure."""
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=timeout,
            creationflags=_NO_WINDOW).stdout
    except Exception as exc:
        log.debug("PowerShell helper failed: %s", exc)
        return ""


def runners() -> list[tuple[int, int]]:
    """Every live llama-server as (pid, port). Empty if none or on any error.

    Ollama picks the port at spawn time and passes it as `--port`, so the
    command line is the only place it can be read from.
    """
    out = _ps(
        "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
        "| ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }",
        _PS_TIMEOUT_S)
    found: list[tuple[int, int]] = []
    for line in out.splitlines():
        m_pid = re.match(r"\s*(\d+)\s", line)
        m_port = re.search(r"--port\s+(\d+)", line)
        if m_pid and m_port:
            found.append((int(m_pid.group(1)), int(m_port.group(1))))
    return found


def _get(url: str, timeout: float) -> tuple[float, object, bytes]:
    """(elapsed, status, body). Never raises.

    `status` is an int for any HTTP response — including error responses, where
    urllib raises rather than returns — and the exception CLASS NAME for a
    connection-level failure (timeout, refused). The distinction matters: a
    refused connection and a 503 mean completely different things here, and
    collapsing every HTTPError to the string "HTTPError" lost the status code
    that tells "still loading" (503) apart from "this runner is erroring" (500).
    """
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return time.monotonic() - t0, r.status, r.read()
    except urllib.error.HTTPError as exc:
        # An HTTP error IS an answer: the server is up and speaking.
        try:
            body = exc.read()
        except Exception:
            body = b""
        return time.monotonic() - t0, exc.code, body
    except Exception as exc:
        return time.monotonic() - t0, type(exc).__name__, b""


def is_loading(port: int, timeout: float | None = None) -> bool:
    """True if the runner is answering 503 'Loading model' — up, but not ready.

    A cold load off the HDD legitimately sits here for minutes, so this is NOT
    on its own a fault. It only becomes one once it has outlasted the cold-load
    budget; see `recover(after_seconds=...)`.
    """
    t = _PROBE_TIMEOUT_S if timeout is None else timeout
    _dt, st, _body = _get(f"http://127.0.0.1:{port}/health", t)
    # 503 SPECIFICALLY, not "any HTTP error". Caught in review by test-engineer:
    # the original check treated every HTTPError as "Loading model", so a runner
    # answering 500 would have been killed once the cold budget passed — a
    # different fault, and one a kill does not fix.
    if st == 503:
        return True
    if isinstance(st, int) and st != 200:
        log.info("Runner :%d answered HTTP %s — not 'loading', so not killed.",
                 port, st)
    return False


def is_wedged(port: int, timeout: float | None = None) -> bool:
    """True only for the measured wedge signature: /health up, /slots hung.

    Deliberately conservative. Every ambiguous outcome returns False, because
    the cost of a false negative is one stalled turn the owner already noticed,
    while the cost of a false positive is killing a healthy runner mid-answer.
    """
    t = _PROBE_TIMEOUT_S if timeout is None else timeout
    h_dt, h_st, _ = _get(f"http://127.0.0.1:{port}/health", t)
    if h_st != 200:
        # Not the signature: the whole server is down or unreachable, which is
        # a different fault and not one a kill fixes.
        log.info("Runner :%d /health %s in %.0f ms — not the wedge signature.",
                 port, h_st, h_dt * 1000)
        return False

    s_dt, s_st, s_body = _get(f"http://127.0.0.1:{port}/slots", t)
    if s_st == 200:
        busy = "?"
        try:
            busy = str(json.loads(s_body)[0].get("is_processing"))
        except Exception:
            pass
        log.info("Runner :%d healthy — /slots 200 in %.0f ms (processing=%s).",
                 port, s_dt * 1000, busy)
        return False

    log.error("Runner :%d WEDGED — /health 200 in %.0f ms but /slots %s after "
              "%.0f ms.", port, h_dt * 1000, s_st, s_dt * 1000)
    return True


def _kill(pid: int) -> bool:
    out = _ps(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue; "
              f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) "
              f"{{ 'alive' }} else {{ 'gone' }}", _PS_TIMEOUT_S)
    return "gone" in out


def recover(after_seconds: float | None = None) -> bool:
    """Probe every runner and kill the wedged ones. True if one was killed.

    Called from the stall path in bastion.py. `ollama serve` is never touched —
    it is healthy during a wedge, and restarting it would evict the model and
    turn a 20 s recovery into a cold load.

    `after_seconds` is the deadline that fired in bastion.py, and it decides
    whether the SECOND wedge shape counts. Observed live 2026-08-18 17:14: the
    runner loaded its tensors onto the GPU fine, then stuck in `warming up the
    model with an empty run` at 100% GPU utilisation with **196 MiB of VRAM
    free**, answering 503 'Loading model' forever. It never reaches /health 200,
    so the /slots probe above cannot see it at all. But a load that has outlived
    the cold-load budget is not loading any more, it is stuck — so a persistent
    503 counts as wedged ONLY when the deadline that fired was the cold budget.
    At the shorter stall deadline a 503 is left alone, because a genuine cold
    load off the HDD takes over three minutes and must not be killed at 45 s.
    """
    if not getattr(config, "WEDGE_RECOVERY_ENABLED", True):
        log.info("Wedge recovery disabled in config; leaving the runner alone.")
        return False

    try:
        found = runners()
    except Exception as exc:            # belt and braces — runners() is guarded
        log.error("Could not enumerate llama-server processes: %s", exc)
        return False

    if not found:
        log.warning("Stall reported but no llama-server process found.")
        return False

    # A 503 only counts once the cold-load budget has been outlived.
    stuck_load_after = float(getattr(config, "WEDGE_LOADING_KILL_AFTER_S",
                                     getattr(config, "OLLAMA_COLD_LOAD_TIMEOUT_S",
                                             300.0)))
    allow_loading_kill = (after_seconds is not None
                          and after_seconds >= stuck_load_after)

    killed = []
    for pid, port in found:
        try:
            if is_wedged(port):
                reason = "/slots hung while /health answered"
            elif allow_loading_kill and is_loading(port):
                reason = (f"still 'Loading model' after {after_seconds:.0f}s, "
                          f"past the {stuck_load_after:.0f}s cold budget")
                log.error("Runner :%d STUCK LOADING — %s.", port, reason)
            else:
                continue
            if _kill(pid):
                killed.append((pid, port, reason))
                log.error("Killed wedged llama-server pid=%d port=%d (%s). "
                          "Ollama will respawn a runner on the next request.",
                          pid, port, reason)
            else:
                log.error("Could not kill wedged llama-server pid=%d.", pid)
        except Exception as exc:
            log.error("Wedge recovery failed for pid=%d: %s", pid, exc)

    if killed:
        try:
            import alerts
            # Metadata only: pid and port. No prompt, no reply, no transcript.
            alerts.send_alert(
                "bastion.wedge", "RunnerWedged",
                "llama-server " +
                ", ".join(f"pid={p} port={q} ({why})" for p, q, why in killed) +
                " was killed; Ollama respawns on the next request.")
        except Exception:
            pass
    return bool(killed)
