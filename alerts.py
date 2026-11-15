"""
Bastion failure alerts → Discord webhook.

THIS IS THE ONE DELIBERATE OUTBOUND CALL IN Bastion, and it is the owner-approved.
Local-first has NOT been abandoned: the LLM, speech-to-text, text-to-speech and
vision paths all still run entirely on this machine and never touch the network.
The single exception is this module, which POSTs *failure metadata only* to a
Discord webhook so a broken Bastion reports itself to the owner's phone instead of
sitting dead until he notices.

What leaves the machine, and nothing else:
  - a component name  ("bastion.chat", "tools.dispatch", …)
  - an exception class / short error label
  - a short detail string written by the caller
  - a timestamp

NEVER put user speech, transcripts, model output, file contents, screenshots,
clipboard contents, tokens or the webhook URL into an alert. If you are about to
format user content into `detail`, that is the failure mode this module exists to
avoid — don't.

Design rules:
  - It must never break Bastion. Every path is wrapped; send_alert() cannot raise.
  - It must never block. The POST runs on a short-lived daemon thread with a
    short timeout, so a hung network can't freeze the assistant.
  - It must never flood. Identical component+error pairs are suppressed inside
    ALERT_DEDUPE_WINDOW_S, and the next alert that gets through says how many
    repeats were swallowed.
"""

import atexit
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import config

log = logging.getLogger("bastion.alerts")

# component+error → [last_sent_monotonic, suppressed_count]
_recent: dict[tuple[str, str], list] = {}
_lock = threading.Lock()

# Sender threads still in flight. They are daemons so they can never hold the
# process open, but that cuts both ways: on an immediate exit Python kills them
# mid-POST and the alert is lost silently. That would drop the single most
# likely alert of all — "Ollama isn't running" at startup, where Bastion prints
# the error and quits straight away. _flush() below closes that window.
_inflight: list[threading.Thread] = []


def _flush() -> None:
    """
    Give pending alerts a brief chance to finish before the process dies.

    Registered with atexit so every exit path is covered without call sites
    having to remember. Bounded by the same timeout as the POST itself, so a
    hung network delays shutdown by seconds, not indefinitely — and like
    everything else here, it can never raise.
    """
    try:
        deadline = time.monotonic() + config.ALERT_TIMEOUT_S
        for thread in list(_inflight):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
    except Exception:
        pass


atexit.register(_flush)

_ALERT_COLOR = 0xE0464B  # red — "go look now" in a phone notification list

# Discord's Cloudflare edge answers 403 (error 1010) to the default
# "Python-urllib/3.x" agent. An explicit UA is required or nothing ever posts.
_USER_AGENT = "Bastion-Alerts (https://localhost, 1.0)"


def _webhook_url() -> str:
    """
    Read the webhook out of the Discord channel .env at call time. Never cached
    to a module constant, never logged, never included in an alert body.
    """
    try:
        with open(config.ALERTS_ENV_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition("=")
                if key.strip() == config.ALERTS_ENV_KEY:
                    return value.strip()
    except Exception:
        pass
    return ""


def _should_send(component: str, error: str) -> tuple[bool, int]:
    """
    Rate-limit gate. Returns (send?, suppressed_since_last). Repeats of the same
    component+error inside the window are counted, not sent — a crash loop must
    not turn into 500 Discord messages.
    """
    now = time.monotonic()
    key = (component, error)
    with _lock:
        entry = _recent.get(key)
        if entry and now - entry[0] < config.ALERT_DEDUPE_WINDOW_S:
            entry[1] += 1
            return False, entry[1]
        suppressed = entry[1] if entry else 0
        _recent[key] = [now, 0]
        # Bound the table: drop entries older than the window so a long-running
        # session with many distinct errors can't grow it without limit.
        for k, v in list(_recent.items()):
            if now - v[0] > config.ALERT_DEDUPE_WINDOW_S * 10:
                _recent.pop(k, None)
        return True, suppressed


def _post(payload: dict) -> None:
    """Fire the POST. Runs on its own thread; swallows absolutely everything."""
    url = _webhook_url()
    if not url:
        log.debug("No alert webhook configured; alert dropped.")
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.ALERT_TIMEOUT_S) as resp:
            if resp.status >= 300:
                log.debug("Alert webhook returned %s", resp.status)
    except urllib.error.HTTPError as exc:
        # Log the status only — the URL is in exc.url and must not reach the log.
        log.debug("Alert webhook HTTP %s", getattr(exc, "code", "?"))
    except Exception as exc:
        log.debug("Alert send failed: %s", type(exc).__name__)


def send_alert(component: str, error: str, detail: str = "") -> None:
    """
    Report a Bastion failure to the #alerts Discord channel. Fire-and-forget.

    component — where it broke, e.g. "bastion.chat" or "hud_server.broadcast"
    error     — short label, normally the exception class name
    detail    — short, non-sensitive context. NO user content, ever.

    Returns immediately and never raises, whatever the network does.
    """
    try:
        if not config.ALERTS_ENABLED:
            return

        component = str(component)[:100]
        error = str(error)[:200]
        detail = str(detail)[:500]

        send, suppressed = _should_send(component, error)
        if not send:
            return

        lines = [f"**Component** · `{component}`", f"**Error** · `{error}`"]
        if detail:
            lines.append(f"**Detail** · {detail}")
        if suppressed:
            lines.append(f"_{suppressed} identical alert(s) suppressed since the last one._")

        # Embed text can't trigger a mention — only top-level content can. So the
        # tag goes in `content`, and allowed_mentions is narrowed to exactly this
        # one user id: never @everyone, never a role, never anything an alert
        # string could smuggle in.
        mention_id = getattr(config, "ALERT_MENTION_USER_ID", "")
        content = f"<@{mention_id}>" if mention_id else ""
        allowed = {"parse": [], "users": [mention_id]} if mention_id else {"parse": []}

        payload = {
            "username": "Bastion Alerts",
            "content": content,
            "allowed_mentions": allowed,
            "embeds": [{
                "title": f"Bastion failure · {component}",
                "description": "\n".join(lines),
                "color": _ALERT_COLOR,
                "footer": {"text": "Bastion self-monitoring"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

        def _run() -> None:
            try:
                _post(payload)
            finally:
                # Drop out of the in-flight list so _flush() never waits on a
                # thread that has already finished, and the list can't grow.
                with _lock:
                    if thread in _inflight:
                        _inflight.remove(thread)

        thread = threading.Thread(target=_run, daemon=True, name="bastion-alert")
        with _lock:
            _inflight.append(thread)
        thread.start()
    except Exception:
        # A monitoring feature that crashes the thing it monitors is worse than
        # no monitoring. Nothing escapes this function.
        pass
