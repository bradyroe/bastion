"""
Bastion HUD bridge — a localhost WebSocket server the wallpaper connects to.

The wallpaper (HTML/JS running in Lively Wallpaper) opens a WebSocket to
ws://127.0.0.1:8765 and receives JSON state messages in real time:

  {"type": "state",      "value": <one of HUD_STATES>, "since": 1755...}
  {"type": "transcript", "role": "user"|"bastion", "text": "..."}
  {"type": "reasoning",  "text": "..."}          empty string clears it
  {"type": "stats",      "cpu": 6.2, "gpu": 38, "gpu_temp": 62,
                         "ram": 37.7, "ram_used_gb": 12.9, "ram_total_gb": 32.0,
                         "disk": 88.6}
  {"type": "weather",    "location": "Springfield", "temp_f": 68, "temp_c": 20,
                         "condition": "clear sky", "humidity": 53, "wind_kmh": 5}
  {"type": "time",       "time": "11:42 PM", "date": "Fri, Jun 20", "iso": "..."}
  {"type": "detail",     "title": "FORECAST", "text": "...", "ttl_s": 180,
                         "truncated": false}
  {"type": "queue",      "waiting": 1, "text": "...", "since": 1755...}
  {"type": "reload"}                             the HUD file changed; refresh

"state" carries "since" — the epoch second the state was entered, not the second
it was sent. The wallpaper counts up from it, so a turn that is 12 s into thinking
still reads 12 s after a reconnect instead of restarting at zero. Same machine,
same clock, so no skew to correct for. The whole point of the field is that the
silence while the model reasons (measured 5-10 s, worst seen 29.4 s) stops being
silence with nothing on screen.

The "queue" message says a turn was HEARD and is waiting behind one already in
flight. It is deliberately not a "state" value: state is single-valued, so
queued would replace thinking on the stage and hide the very thing the owner needs
to see. Two concurrent facts, two messages. See "The waiting queue" below.

The "detail" message exists so Bastion can *speak a short answer and show the long
one*. Every spoken character costs synthesis time and, far more, the owner's patience;
putting the supporting detail on the wallpaper instead of into the voice is the
cheapest second there is. split_reply()/send_reply() below decide that split.
Lifetime is decided here, not by the caller — see send_detail().

Before any of that flows, the client must authenticate:

  client -> {"type": "auth", "token": HUD_TOKEN}      (first frame, within 5s)
  server -> {"type": "auth_ok"}                       then the snapshot + feed

Security:
  - Bound to 127.0.0.1 ONLY. Never reachable from the network.
  - Loopback is NOT sufficient on its own: WebSockets are exempt from the
    same-origin policy, so any web page the owner visits could otherwise open this
    socket and stream his transcripts and live reasoning. Hence the Origin
    allowlist below plus a shared token, both enforced fail-closed.
  - Server is send-only from our side; after the auth frame all inbound client
    messages are ignored.
  - No tool execution happens here — this is a one-way status feed.
"""

import asyncio
import json
import re
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import psutil

import alerts
import config
import tools as T

log = logging.getLogger("bastion.hud")

HOST = "127.0.0.1"
PORT = 8765

# Auth policy lives in config.py with the rest of the security policy — these are
# re-exported here only so existing callers (and test_hud.py) keep working.
# See config.py for what each one is defending against.
HUD_TOKEN = config.HUD_TOKEN
HUD_ALLOWED_ORIGINS = config.HUD_ALLOWED_ORIGINS
HUD_ALLOWED_ORIGIN_HOSTS = config.HUD_ALLOWED_ORIGIN_HOSTS
HUD_AUTH_TIMEOUT_S = config.HUD_AUTH_TIMEOUT_S

# ws -> that client's outbound queue. Every message a client receives goes
# through its own queue, including the connect snapshot, which is what makes
# "replay what's current" and "here's a live update" a single ordered stream
# instead of two racing ones. See _deliver() for why that matters.
_clients: dict = {}
_QUEUE_MAX = 256          # a stalled client drops its oldest frames, never grows

_loop: asyncio.AbstractEventLoop | None = None
# Retained state, replayed to a client when it connects. These are owned by the
# event loop — only an `apply` callback passed to broadcast() may write them.
_state: str = "idle"
_state_since: float = time.time()
_last_weather: dict | None = None
_last_detail: dict | None = None
_detail_expires_at: float = 0.0
_last_queue: dict | None = None
_last_tool: dict | None = None   # loop-owned, replayed in the connect snapshot
_tool_shown: dict | None = None  # calling-thread mirror; see send_tool()
_last_brain: dict | None = None  # loop-owned, replayed in the connect snapshot

# Caller-thread mirrors, used *only* to decide whether to publish at all. They
# exist because the values above are now written on the loop, so a publisher
# reading them back would race its own previous call.
_state_intent: str = "idle"
_detail_shown: bool = False

# The waiting-turn counter. Owned by the calling threads (voice.py's main loop
# and anything else that blocks on the turn lock), so it has its own mutex —
# unlike the retained values above, it is *computed* by callers rather than
# merely mirrored.
_queue_lock = threading.Lock()
_queue_waiting: int = 0
_queue_since: float = 0.0
_queue_text: str = ""

# Detail-panel policy. Kept here beside HOST/PORT because they're properties of
# this transport, not security policy — config.py stays the source of truth for
# the auth/allowlist settings that hud_server re-exports above.
HUD_DETAIL_MAX_CHARS = 4000   # hard cap; longer text is cut with a visible marker
HUD_DETAIL_TTL_S     = 180    # default lifetime; 0 from a caller means "no timeout"

# Every value set_state() is expected to carry. The server does not enforce it —
# an unknown value still broadcasts and the wallpaper falls back to showing it
# verbatim — but this is the list the HUD has a designed appearance for, and the
# list V0_PROMPT.md documents. "transcribing" and "answering" were added
# 2026-08-17 to close the two silent gaps in a turn: Whisper running after the
# mic closes (~2.9 s of "listening" that is no longer listening), and the model
# writing the answer after it has stopped reasoning.
HUD_STATES = ("idle", "listening", "transcribing", "thinking", "answering", "speaking")

# Voice/screen split policy. Kept beside the detail constants because it is the
# same decision: what is worth spending speech on.
HUD_SPOKEN_MAX_CHARS = 240    # soft budget for the spoken line; whole sentences only
# Above this many bullets, a flattened list stops sounding like a sentence and
# starts sounding like a recitation, whatever its length — 20 one-word items is
# only 148 characters and is still miserable to hear. A judgement, not a
# measurement: the char budget alone does not catch it, because the cost of a
# list scales with the number of items, not with its length.
HUD_SPOKEN_MAX_ITEMS = 6
HUD_HANDOFF_PHRASE   = "The rest is on the screen."
HUD_SCREEN_ONLY_LINE = "It is on the screen."


# ─── WebSocket handling ───────────────────────────────────────────────────────

def _origin_allowed(origin) -> bool:
    """
    Fail-closed Origin check: only the local wallpaper's origins are allowed.

    Matched by exact literal or by parsed host — never by prefix. A prefix test
    against "http://localhost" also accepts http://localhost.evil.com, which is
    a domain anyone can register; that hole was real here until it was caught by
    testing this function against it.
    """
    if origin in HUD_ALLOWED_ORIGINS:
        return True
    if not isinstance(origin, str):
        return False
    try:
        parts = urlsplit(origin.strip().lower())
    except ValueError:
        return False
    # An Origin is scheme://host[:port] and nothing else — a path, query, or
    # userinfo means it isn't one, so refuse rather than try to interpret it.
    if parts.path or parts.query or parts.fragment or "@" in parts.netloc:
        return False
    if parts.scheme != "http":
        return False
    try:
        host = parts.hostname
    except ValueError:
        return False
    return host in HUD_ALLOWED_ORIGIN_HOSTS


def _process_request(connection, request):
    """Reject the handshake outright for any origin that isn't the wallpaper's."""
    try:
        origin = request.headers.get("Origin")
    except Exception:
        origin = None
    if not _origin_allowed(origin):
        log.warning("HUD connection refused — disallowed Origin %r", origin)
        return connection.respond(403, "forbidden\n")
    log.debug("HUD handshake accepted from Origin %r", origin)
    return None


async def _authenticate(ws) -> bool:
    """Require {"type":"auth","token":...} as the first frame. No token, no feed."""
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=HUD_AUTH_TIMEOUT_S)
        msg = json.loads(raw)
        if msg.get("type") == "auth" and str(msg.get("token", "")) == HUD_TOKEN:
            # auth_ok is sent by _handler, *after* the client is registered and
            # its snapshot queued — see the comment there. Sending it from here
            # told the client it was live while the server still had nowhere to
            # put a message for it.
            return True
    except Exception as exc:
        log.warning("HUD auth failed (%s)", type(exc).__name__)
    else:
        log.warning("HUD auth failed — bad or missing token")
    try:
        # 4401: application-level "unauthorised". The client should not retry
        # in a tight loop on this, but reconnecting is harmless either way.
        await ws.close(code=4401, reason="unauthorized")
    except Exception:
        pass
    return False


async def _writer(ws, q: asyncio.Queue) -> None:
    """
    Drain one client's queue. The only place anything is awaited on a socket,
    which is what keeps message order per client independent of message size —
    a big frame yielding on flow control can no longer be overtaken by a small
    one scheduled behind it.
    """
    try:
        while True:
            data = await q.get()
            await ws.send(data)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Send failed on a socket the handler still thinks is live. Drop the
        # client and close, so the handler unwinds instead of queueing forever.
        _clients.pop(ws, None)
        try:
            await ws.close()
        except Exception:
            pass


async def _handler(ws):
    """One connected wallpaper client. Authenticate, snapshot, then ignore inbound."""
    if not await _authenticate(ws):
        return

    q: asyncio.Queue = asyncio.Queue()
    # ── This block must not await. ────────────────────────────────────────────
    # Reading the retained state and registering the client have to be one
    # indivisible step on the loop, or a message published in between is either
    # lost (client not yet in _clients) or delivered twice (in the snapshot AND
    # in the fan-out). Both were live: the duplicate is what made a caller that
    # published on seeing auth_ok read the whole stream off by one.
    q.put_nowait(json.dumps({"type": "state", "value": _state, "since": _state_since}))
    if _last_weather:
        q.put_nowait(json.dumps(_last_weather))
    # The reload watcher refreshes the wallpaper whenever the HTML changes, so a
    # live detail panel has to survive a reconnect or it vanishes mid-read.
    # Expired ones are simply not replayed.
    if _last_detail and (_detail_expires_at == 0.0 or time.time() < _detail_expires_at):
        q.put_nowait(json.dumps(_last_detail))
    # Same reasoning for a queued turn: the wallpaper reloading must not lose
    # the one message that says "you were heard, you are next".
    if _last_queue:
        q.put_nowait(json.dumps(_last_queue))
    # And for a tool in flight: look_at_screen alone runs ~30 s, which is long
    # enough for a reload to land in the middle of it and leave the stage with
    # nothing under THINKING again — the exact hole this message fills.
    if _last_tool:
        q.put_nowait(json.dumps(_last_tool))
    # Which brain is answering is a PERSISTENT fact, not an event — a wallpaper
    # that reloaded and did not get it back would sit there implying "local"
    # while Claude was answering, which is the one thing this message exists to
    # prevent. So it is always replayed when known.
    if _last_brain:
        q.put_nowait(json.dumps(_last_brain))
    _clients[ws] = q
    # ── awaiting is safe again from here ──────────────────────────────────────

    log.info("HUD client connected (%d total)", len(_clients))
    writer = None
    try:
        # Only now is the client told it's live, so anything it publishes in
        # response is guaranteed to queue *behind* its own snapshot.
        await ws.send(json.dumps({"type": "auth_ok"}))
        writer = asyncio.create_task(_writer(ws, q))
        async for _ in ws:
            pass  # we don't accept commands over this channel — status only
    except Exception:
        pass
    finally:
        _clients.pop(ws, None)
        if writer:
            writer.cancel()
        log.info("HUD client disconnected (%d total)", len(_clients))


async def _deliver(message: dict, apply=None) -> None:
    """
    Publish one message. Runs on the loop and **never awaits**, which is the
    whole point: `apply` (updating the retained snapshot) and the fan-out become
    a single atomic step, so a client connecting at that instant sees the
    message exactly once — from its snapshot or from the queue, never both.

    It also fixes ordering. The old version awaited ws.send() per client, so two
    concurrently-scheduled broadcasts could interleave and a small message could
    overtake a large one — a clear arriving before the panel it was clearing.
    Handing off to a per-client queue makes order the queue's problem.
    """
    if apply is not None:
        apply()
    if not _clients:
        return
    data = json.dumps(message)
    for ws, q in list(_clients.items()):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            # A client that has stopped reading must not grow the queue without
            # bound. Drop its oldest frame — the feed is state, not a ledger.
            try:
                q.get_nowait()
                q.put_nowait(data)
            except Exception:
                pass


def broadcast(message: dict, apply=None) -> None:
    """
    Thread-safe publish — callable from voice.py's synchronous code.

    `apply` is an optional callable that updates the retained snapshot state. It
    runs **on the event loop**, together with the fan-out, rather than in the
    calling thread; that is what makes the two atomic. Mutating a global here and
    broadcasting separately is the bug this signature exists to prevent.
    """
    if _loop is None:
        # No loop yet (or it's gone): still apply, so the retained state a later
        # client snapshots stays correct.
        if apply is not None:
            apply()
        return
    try:
        asyncio.run_coroutine_threadsafe(_deliver(message, apply), _loop)
    except Exception as exc:
        log.debug("broadcast failed: %s", exc)
        # The event loop is gone — the HUD has stopped updating entirely. Deduped
        # to one alert per window so a dead loop can't spam once per second.
        alerts.send_alert("hud_server.broadcast", type(exc).__name__,
                          "HUD broadcast failed; wallpaper feed is not updating.")


def set_state(value: str) -> None:
    """Update and broadcast the assistant state (idle/listening/thinking/speaking)."""
    global _state_intent
    prev, _state_intent = _state_intent, value
    # Entering "thinking" means a new answer is being composed, so anything still
    # on the detail panel is now the answer to the *previous* question. Clear it
    # here rather than on "listening": while he's talking he may well still be
    # reading the panel, and clearing it out from under him is the wrong moment.
    # Published before the state change so the two arrive in that order.
    if value == "thinking" and prev != "thinking":
        clear_detail()
    # A tool belongs to the round that called it. Cleared on entering any state
    # that is NOT thinking — never on entering "thinking", because voice.py
    # re-enters it after every tool round and that would wipe the line the
    # moment the next round began. So "answering", "speaking", "idle" and a new
    # "listening" all take it down, and nothing can leave a finished tool name
    # sitting on the stage claiming to be current.
    if value != "thinking":
        clear_tool()

    # Stamped once, here, and both broadcast and retained — so the wallpaper's
    # elapsed counter measures from the state change rather than from whenever a
    # client happened to connect. A reconnect mid-think must not reset it to 0.
    since = time.time()

    def _apply():
        global _state, _state_since
        _state = value
        _state_since = since

    broadcast({"type": "state", "value": value, "since": since}, _apply)


def send_transcript(role: str, text: str) -> None:
    broadcast({"type": "transcript", "role": role, "text": text})


def send_detail(text: str, title: str = "DETAIL", ttl_s: int | None = None) -> None:
    """
    Show a longer written answer on the HUD's detail panel.

    The point is that Bastion says one or two sentences out loud and puts the rest
    here, so the reply the owner waits on is short. `text` is plain text — newlines
    and simple "- " lists survive, HTML does not (the wallpaper escapes it).

    Lifetime, in the order they fire:
      1. the next turn — set_state("thinking") clears it automatically;
      2. `ttl_s` seconds, default HUD_DETAIL_TTL_S (180). Pass 0 to pin it until
         the next turn, for something he asked to leave up;
      3. clear_detail(), or send_detail("").
    A wallpaper that reconnects mid-life gets the panel back in its snapshot.
    """
    text = str(text or "")
    if not text.strip():
        clear_detail()
        return
    truncated = len(text) > HUD_DETAIL_MAX_CHARS
    if truncated:
        # Never silently swallow the tail — the panel says so on screen.
        text = text[:HUD_DETAIL_MAX_CHARS]
    ttl = HUD_DETAIL_TTL_S if ttl_s is None else max(0, int(ttl_s))

    msg = {
        "type":      "detail",
        "title":     str(title or "DETAIL"),
        "text":      text,
        "ttl_s":     ttl,
        "truncated": truncated,
    }

    global _detail_shown
    _detail_shown = True

    def _apply():
        global _last_detail, _detail_expires_at
        _detail_expires_at = (time.time() + ttl) if ttl else 0.0
        _last_detail = msg

    broadcast(msg, _apply)


def clear_detail() -> None:
    """Take the detail panel down. Empty text is the clear signal, as with reasoning."""
    global _detail_shown
    # Decided on the mirror, not on _last_detail: that one is now owned by the
    # event loop and may not have been assigned yet when this runs, which would
    # skip the clear and strand a panel on screen.
    if not _detail_shown:
        return
    _detail_shown = False

    def _apply():
        global _last_detail, _detail_expires_at
        _last_detail = None
        _detail_expires_at = 0.0

    broadcast({"type": "detail", "text": ""}, _apply)


# ─── The waiting queue ────────────────────────────────────────────────────────
#
# Measured 2026-08-17 22:54: the owner asked an expensive question (2,421 thinking
# chars, ~18 s), then asked a second one nine seconds later. The second turn
# waited on the first. He saw silence, concluded Bastion had hung, and went to
# ask a human. First audio on that turn was 37.14 s.
#
# Nothing was broken. The only thing missing was the HUD saying "heard you,
# you're next" — which is cheaper than any latency work and fixes the actual
# complaint, which was not "it was slow" but "I could not tell if it heard me".
#
# WHY THIS IS NOT A `state` VALUE, which is the obvious first idea and is wrong:
# `state` is single-valued and mutually exclusive. Publishing state="queued"
# would REPLACE "thinking" on the stage — so the moment a second question
# arrived, the HUD would stop showing that the first one was still being worked
# on. That makes the long-think problem worse, not better, and it is exactly the
# information the owner was missing. The two facts are concurrent, so they are two
# messages:
#
#   {"type":"queue","waiting":1,"text":"What are you playing on Spotify?",
#    "since":1787009592.4}
#
#   waiting  how many turns are blocked behind the one in flight. 0 clears it.
#   text     the newest waiting request if it is known — it usually is NOT at
#            the moment of queueing, because a wake word queues before Whisper
#            has run. Empty is normal and the HUD handles it.
#   since    the epoch second the OLDEST waiter began waiting, same convention
#            as state.since, so the wallpaper can count up and survive a reload.
#
# Old wallpapers ignore an unknown message type harmlessly, so this half can
# ship before the voice.py call sites do — exactly how "transcribing" and
# "answering" shipped.

def _publish_queue() -> None:
    """Broadcast the current queue depth. Caller must hold _queue_lock."""
    msg = {
        "type":    "queue",
        "waiting": _queue_waiting,
        "text":    _queue_text,
        "since":   _queue_since,
    }

    def _apply():
        global _last_queue
        # Retained so a wallpaper that reloads mid-wait (the reload watcher
        # fires on every HTML edit) gets the banner back instead of losing it.
        _last_queue = msg if msg["waiting"] > 0 else None

    broadcast(msg, _apply)


def queue_enter(text: str = "") -> None:
    """
    A turn has started WAITING for one already in flight. Call it the moment the
    wait begins, not when the turn eventually runs.

    `text` is what was heard, if it is known yet. Pass "" when it isn't — the
    count alone already answers the question the owner was actually asking.
    """
    global _queue_waiting, _queue_since, _queue_text
    with _queue_lock:
        _queue_waiting += 1
        if _queue_waiting == 1:
            # First waiter sets the clock; later ones join an existing wait.
            _queue_since = time.time()
        text = str(text or "").strip()
        if text:
            _queue_text = text
        _publish_queue()


def queue_leave() -> None:
    """
    A waiting turn has stopped waiting — it got the lock and is now running, or
    it gave up. Pair it with queue_enter() in a finally, never on the happy path
    alone: a banner that says a question is pending after it has been answered
    is a worse lie than showing nothing.
    """
    global _queue_waiting, _queue_since, _queue_text
    with _queue_lock:
        if _queue_waiting <= 0:
            # Unbalanced leave. Clamped rather than allowed negative, so one
            # missed enter() can't make the counter permanently un-clearable.
            _queue_waiting = 0
            return
        _queue_waiting -= 1
        if _queue_waiting == 0:
            _queue_since = 0.0
            _queue_text = ""
        _publish_queue()


def clear_queue() -> None:
    """Force the queue banner down. For recovery paths, where the count may have
    drifted — an aborted turn, a crash mid-wait, the start of a fresh session."""
    global _queue_waiting, _queue_since, _queue_text
    with _queue_lock:
        if _queue_waiting == 0 and not _queue_text:
            return
        _queue_waiting = 0
        _queue_since = 0.0
        _queue_text = ""
        _publish_queue()


def queue_depth() -> int:
    """How many turns are currently waiting. For tests and for voice.py's logs."""
    with _queue_lock:
        return _queue_waiting


# ─── The running tool ─────────────────────────────────────────────────────────
#
# Measured 2026-08-18: a voice turn took 105.8 s before Bastion spoke a word. The
# HUD did not go blank — bastion.py retracts the streamed preamble on every tool
# round, so voice.py puts the state back to "thinking" each time — but under the
# SPLIT thinking policy those rounds run with think=False, so NO reasoning is
# produced: the rail sat on THINKING with nothing beneath it for a hundred
# seconds. From across the room that is indistinguishable from a hung assistant,
# which is exactly how the owner described it.
#
# The missing fact is WHICH TOOL IS RUNNING. That turn was tv_control, then
# look_at_screen, then tv_control — and seeing those three names go by answers
# both questions at once: it is alive, and this is what is taking the time.
#
#   {"type":"tool","name":"look_at_screen","active":true,"since":1787021759.63}
#
#   name    the tool's name and NOTHING else. See below.
#   active  true while dispatch() is running it, false the moment it returns.
#           A finished name stays on screen, dimmed, until the state leaves
#           thinking — "it just did this" is still useful while the model
#           phrases the answer, and a line that vanished instantly would flicker
#           through three tools in a turn and read as noise.
#   since   epoch second the tool started, same convention as state.since, so
#           the wallpaper can count up and survive a reload.
#
# **ARGUMENTS ARE DELIBERATELY NOT SENT AND MUST NOT BE.** Tool arguments
# routinely carry what the owner said, file paths, search queries and clipboard
# text, and this goes onto a wallpaper on a screen other people can see. The
# name alone is the whole benefit at none of the risk. `send_tool` takes no
# argument parameter, so adding one is a visible change to this signature rather
# than a caller quietly passing more.
#
# Old wallpapers ignore an unknown message type harmlessly, as with `queue`.

# The model writes the tool name, so it is untrusted text: capped and reduced to
# an identifier-shaped string here rather than trusting the wallpaper's escaping
# alone. Both wallpapers escape it too — this is the belt.
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_.\- ]+")
HUD_TOOL_NAME_MAX = 48


def _clean_tool_name(name: str) -> str:
    return _TOOL_NAME_RE.sub("", str(name or "")).strip()[:HUD_TOOL_NAME_MAX]


def send_tool(name: str, active: bool = True) -> None:
    """
    Say which tool is running (or has just run). An empty name clears the line,
    same convention as `reasoning` and `detail`.

    Pair it in a `finally` — `send_tool(n)` before dispatch, `send_tool(n, False)`
    after — so a tool that raises does not leave the HUD claiming it is still
    running. A stale "look_at_screen, 90.0s" is the same lie as a stale THINKING.
    """
    name = _clean_tool_name(name)
    if not name:
        clear_tool()
        return
    global _tool_shown
    active = bool(active)
    # A tool that has finished keeps the start time it was given, so the dimmed
    # line reads "that took 30.2 s" rather than restarting at zero.
    since = time.time()
    if not active and _tool_shown and _tool_shown.get("name") == name:
        since = _tool_shown.get("since", since)
    msg = {"type": "tool", "name": name, "active": active, "since": since}

    # The mirror is updated HERE, in the calling thread, and the guard in
    # clear_tool() reads it — deliberately, and for the same reason as
    # `_detail_shown` above. `_last_tool` is owned by the event loop and may not
    # have been assigned yet when a clear follows a send closely, which would
    # skip the clear and strand a tool name on the stage.
    _tool_shown = msg

    def _apply():
        global _last_tool
        _last_tool = msg

    broadcast(msg, _apply)


def clear_tool() -> None:
    """Take the tool line down. Cheap no-op when nothing is showing."""
    global _tool_shown
    if _tool_shown is None:
        return
    _tool_shown = None

    def _apply():
        global _last_tool
        _last_tool = None

    broadcast({"type": "tool", "name": "", "active": False}, _apply)


def current_tool() -> dict | None:
    """What the HUD is showing on the tool line, or None. For tests."""
    return _tool_shown


# ─── The brain, and how much of it is left ────────────────────────────────────
#
# Bastion can now answer from the LOCAL model or from CLAUDE via the CLI. Those
# two are not interchangeable from where the owner sits: in local mode his speech
# stays on this machine, and in Claude mode it does not. **He must never have to
# guess which one he is talking to**, so this is a persistent indicator, not a
# notification — it says the same thing between turns as it does during one.
#
# The second half is usage. The CLI reports a rate limit as:
#
#   {"type":"rate_limit_event","rate_limit_info":{
#       "status":"allowed","resetsAt":1787116200,
#       "rateLimitType":"five_hour","overageStatus":"rejected"}}
#
# Three of his agents hit that limit in one evening, so "exhausted" is a state
# that WILL occur and not a hypothetical.
#
#   {"type":"brain","brain":"claude","label":"claude-opus-5","remote":true,
#    "since":1787100000.0,
#    "usage":{"level":"warn","status":"allowed_warning","window":"five_hour",
#             "resets_at":1787116200.0,"overage":"rejected",
#             "percent":94.0,"threshold":90.0}}
#
#   brain   free string. **An EMPTY brain clears the band**, and that is what
#           "local" means here — see set_brain(). Anything else is treated as
#           remote unless the caller says otherwise.
#   remote  DOES HIS SPEECH LEAVE THE MACHINE. The one bit that matters.
#   label   model id or similar, for the small line under the name. May be "".
#   usage   the rate-limit half, or null for **"no call yet"** — see below.
#   since   epoch second the brain last changed, same convention as state.since.
#
# **LOCAL IS SILENCE.** There is no local `brain` message: switching back to the
# on-device model clears the band, so a local HUD is byte-for-byte what it was
# before this feature existed, and **the band being on screen at all is itself
# the signal that speech is leaving the machine**. A band that sat there saying
# LOCAL all day would be furniture inside a week.
#
# **`usage: null` on a remote brain means "NO CALL YET", not "zero".** The CLI
# only reports the window as a side effect of a request, so there is no way to
# poll it — a fresh process genuinely does not know, and that is what the owner sees
# most mornings. Rendering it as 0, or as an empty bar, would read as "out of
# usage" when it means "we have not spent a call to find out". The wallpapers
# print the words NO CALL YET.
#
# **ONE message type, two setters.** set_brain() and set_usage() each republish
# the whole retained record rather than emitting a fragment of it. A usage event
# does not know which brain is loaded and a toggle does not know the rate limit,
# so emitting them separately would let the wallpaper render half a truth — the
# expensive version of which is "LOCAL" next to a Claude usage bar. Merging in
# one place makes that unrepresentable.

# What the CLI's `status` can say is NOT documented and only "allowed" has been
# observed. So this classifier is a guess, and it is written to fail in the safe
# direction: a value it does not recognise becomes "unknown" and the wallpaper
# prints it VERBATIM rather than colouring it green. The failure mode being
# avoided is a new value for "you are out" quietly reading as "you are fine".
_USAGE_OK = {
    "allowed", "ok", "active", "available", "normal", "none", "under_limit",
}
_USAGE_WARN = {
    "allowed_warning", "warning", "warn", "approaching", "approaching_limit",
    "nearing_limit", "near_limit", "low", "limited",
}
_USAGE_OUT = {
    "rejected", "exhausted", "blocked", "denied", "exceeded", "over_limit",
    "limit_reached", "rate_limited", "unavailable",
}


def _fraction(value) -> float | None:
    """
    A 0-1 fraction, or None. Anything outside that range is not believed.

    Deliberately does NOT rescue a value like `94` as "already a percentage":
    0.94 and 94 are both plausible readings of the same field and guessing wrong
    draws a confident meter at the wrong length. No meter beats a wrong one.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if 0.0 <= out <= 1.0 else None


def _usage_level(status: str) -> str:
    """Classify a rate-limit status. Unknown in, "unknown" out — never "ok"."""
    s = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return "unknown"
    if s in _USAGE_OK:
        return "ok"
    if s in _USAGE_WARN:
        return "warn"
    if s in _USAGE_OUT:
        return "out"
    return "unknown"


# Free text off a subprocess, headed for a wallpaper other people can see. Both
# wallpapers set it with textContent as well; this is the belt, as with tools.
_LABEL_RE = re.compile(r"[^A-Za-z0-9_.:\-/ ]+")
HUD_BRAIN_LABEL_MAX = 40


def _clean_label(value, limit: int = HUD_BRAIN_LABEL_MAX) -> str:
    return _LABEL_RE.sub("", str(value or "")).strip()[:limit]


# Every observed `resetsAt` is integer unix SECONDS (1787116200) — but every
# observation is from an "allowed" response, so an exhausted one using a
# different encoding is unproven rather than ruled out (`brain-toggle`,
# 2026-08-18). Hence a sanity window rather than trust.
_EPOCH_MIN = 1e9                     # 2001-09; anything earlier is not a date
_EPOCH_MAX_AHEAD = 400 * 86400       # a rate-limit window is not a year long


def _epoch(value) -> float | None:
    """
    A unix timestamp from the CLI, or None if it isn't plausibly one.

    **None means "unknown", and the wallpapers then show no countdown at all** —
    which is the right answer for a value we cannot read. The trap being avoided
    is that a garbage timestamp is not inert: a small number is a 1970 date, so
    it renders as "reset due" and quietly claims the window has already reset.
    A wrong fact is worse here than a missing one.
    """
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    # Measured as seconds, but a JS-side producer sending milliseconds would
    # otherwise render as a countdown fifty thousand years long.
    if n > 1e11:
        n /= 1000.0
    if n < _EPOCH_MIN or n > time.time() + _EPOCH_MAX_AHEAD:
        return None
    return n


# Retained brain record, or None. None is deliberate and is NOT "local": it
# means nothing has reported yet, and the wallpaper then shows no brain band and
# leaves its static header alone — i.e. exactly today's HUD. The alternative,
# defaulting to "local", would have a silent backend assert the private case
# while the remote one was running, which is the one error that actually costs
# something here.
_brain: dict | None = None
_brain_lock = threading.Lock()


def _publish_brain() -> None:
    """Republish the whole retained brain record. Caller holds _brain_lock."""
    msg = dict(_brain) if _brain else None
    if msg is None:
        return

    def _apply():
        global _last_brain
        _last_brain = msg

    broadcast(msg, _apply)


def set_brain(brain: str, label: str = "", remote: bool | None = None) -> None:
    """
    Say which brain is answering. Call it on every toggle.

    `brain` is a free string — "local" for the on-device model, anything else
    (e.g. "claude") is taken as remote unless `remote` says otherwise. `label`
    is the model id shown small underneath, and may be empty.

    **LOCAL EMITS NOTHING.** Passing "local" (or anything non-remote) does not
    publish a local brain — it CLEARS the record, so a local-mode HUD is
    byte-for-byte what it was before this feature existed. Two reasons, and the
    second is the good one:

      1. anyone comparing a local HUD before and after this change sees zero
         difference, so this cannot be blamed for anything;
      2. **the band's mere presence then means "your speech is leaving this
         machine".** A band that is always there, saying LOCAL most of the time,
         is furniture — it stops being read within a week. A band that is
         normally absent is impossible to miss when it appears.

    Cannot raise: the HUD must never be able to break a voice turn.
    """
    try:
        name = _clean_label(brain, 24).lower() or "unknown"
        if remote is None:
            remote = name != "local"
        if not remote:
            # Local is the silent default, not a state worth a message.
            clear_brain()
            return
        with _brain_lock:
            global _brain
            prev = _brain or {}
            # `since` only moves when the brain itself changes, so a stream of
            # rate-limit events does not keep resetting the "switched N ago"
            # counter on the stage.
            since = prev.get("since", 0.0)
            if prev.get("brain") != name:
                since = time.time()
            _brain = {
                "type": "brain",
                "brain": name,
                "label": _clean_label(label),
                "remote": bool(remote),
                "usage": prev.get("usage"),
                "since": since,
            }
            _publish_brain()
    except Exception as exc:            # pragma: no cover - belt and braces
        log.debug("set_brain failed: %s", exc)


def set_usage(info) -> None:
    """
    Report a rate limit. Pass the CLI's `rate_limit_info` dict **as-is**.

    Reads `status`, `resetsAt`, `rateLimitType`, `overageStatus` (snake_case
    accepted too). Interpreting it here rather than at the call site means there
    is exactly one place to fix when the real value set is known.

    Note what is deliberately NOT done: the sample carries
    `overageStatus: "rejected"` alongside `status: "allowed"`, so a classifier
    that matched "rejected" anywhere in the payload would show EXHAUSTED on a
    completely healthy account. Only `status` decides the level.
    """
    try:
        if info is None:
            clear_usage()
            return
        if not isinstance(info, dict):
            info = {"status": info}
        status = info.get("status", info.get("rate_limit_status", ""))
        window = info.get("rateLimitType", info.get("rate_limit_type", ""))
        overage = info.get("overageStatus", info.get("overage_status", ""))
        # THERE IS A REAL PERCENTAGE. This file, V0_PROMPT.md and TEAM-LOG all
        # said there was not, and all three were wrong — the earlier survey only
        # ever saw `allowed` events, which genuinely carry no fraction.
        # `allowed_warning` carries `utilization` (0.94 = 94% of the five-hour
        # window spent) plus the `surpassedThreshold` that tripped it, recovered
        # 2026-08-18 by replaying the owner's own 94% event.
        frac = _fraction(info.get("utilization", info.get("utilisation")))
        thresh = _fraction(info.get("surpassedThreshold",
                                    info.get("surpassed_threshold")))
        usage = {
            "level": _usage_level(status),
            "status": _clean_label(status, 32),
            "window": _clean_label(window, 32),
            "resets_at": _epoch(info.get("resetsAt", info.get("resets_at"))),
            "overage": _clean_label(overage, 32),
            # **None means NOT REPORTED, and 0 means NOTHING USED.** Opposite
            # facts, and most events carry no fraction at all, so conflating
            # them would draw an empty meter — "you have used none of it" — at
            # the exact moment nobody knows. Same rule as `usage: null` itself.
            "percent": None if frac is None else round(frac * 100.0, 1),
            "threshold": None if thresh is None else round(thresh * 100.0, 1),
        }
        with _brain_lock:
            global _brain
            if _brain is None:
                # No brain is showing, which now means LOCAL (see set_brain).
                # A late rate-limit event from a Claude call that has already
                # been switched away from must not raise a band on a local HUD.
                return
            _brain = {**_brain, "type": "brain", "usage": usage}
            _publish_brain()
    except Exception as exc:            # pragma: no cover - belt and braces
        log.debug("set_usage failed: %s", exc)


def clear_brain() -> None:
    """
    Take the whole brain band down — i.e. "back on the local model".

    Broadcast even though the retained record becomes None, because a wallpaper
    that is already showing CLAUDE has to be told to stop. Only the retained
    copy is dropped, so nothing is replayed to a client connecting afterwards.
    """
    try:
        with _brain_lock:
            global _brain
            if _brain is None:
                return
            _brain = None

            def _apply():
                global _last_brain
                _last_brain = None

            broadcast({"type": "brain", "brain": "", "label": "",
                       "remote": False, "usage": None, "since": time.time()},
                      _apply)
    except Exception as exc:            # pragma: no cover - belt and braces
        log.debug("clear_brain failed: %s", exc)


def clear_usage() -> None:
    """Drop the usage half, leaving the brain shown. No-op if nothing is set."""
    try:
        with _brain_lock:
            global _brain
            if not _brain or _brain.get("usage") is None:
                return
            _brain = {**_brain, "usage": None}
            _publish_brain()
    except Exception as exc:            # pragma: no cover - belt and braces
        log.debug("clear_usage failed: %s", exc)


def current_brain() -> dict | None:
    """The retained brain record, or None if nothing has reported. For tests."""
    with _brain_lock:
        return dict(_brain) if _brain else None


# ─── The voice/screen split ───────────────────────────────────────────────────
#
# The problem this solves: Bastion reads long replies aloud and the owner sits through
# them. Median reply is 306 chars, but the tail is long, and a list read out loud
# is miserable — you cannot skim speech, and you cannot go back.
#
# The rule: **the spoken line must stand alone.** Someone who never looks at the
# wallpaper still gets a complete, true, useful answer. The screen carries the
# part that is better read than heard — the enumeration, the numbers, the steps.
#
# Two decisions that look wrong until you have watched it:
#
#   1. The panel gets the WHOLE reply, not the unspoken remainder. He looks up
#      several seconds late, and a paragraph that begins mid-thought is worse to
#      read than one sentence of overlap is to re-read.
#   2. A spoken line is never cut mid-sentence, even when that overruns the
#      budget. Half a sentence does not stand alone, and the budget is a budget,
#      not a limit.

# A sentence ends at . ! ? possibly followed by a closing quote/bracket, then
# whitespace. Deliberately simple: the failure mode of a clever splitter is
# speaking half of "e.g." and the failure mode of this one is speaking one
# sentence too many.
# Two alternatives rather than one, because the closer has to be matched by a
# LOOKBEHIND, not consumed. As part of the separator it was deleted outright:
# 'She said "no." I agreed.' split to ['She said "no.', 'I agreed.'] — the
# closing quote silently gone. Python needs fixed-width lookbehind, so a run of
# two or more closers ('.")') still will not split; that is a missed split,
# which is harmless, rather than a corrupted one, which is not.
_SENTENCE_SPLIT_RE = re.compile(r"""(?<=[.!?])\s+|(?<=[.!?]['")\]])\s+""")
# "- ", "* ", "• ", "1. ", "2) " — the shapes that are painful to hear read out.
_LIST_LINE_RE      = re.compile(r'^\s*(?:[-*•–]\s+|\d+[.)]\s+)')


def split_reply(text: str, max_chars: int | None = None) -> tuple[str, str]:
    """
    Split a reply into (spoken, detail).

    `detail` is empty unless the reply is genuinely worth reading — a list, or
    longer than the spoken budget. Most turns therefore put nothing on screen at
    all, which is the point: a HUD that lights up on every "yes, sir" is noise.
    """
    text = str(text or "").strip()
    if not text:
        return "", ""
    budget = HUD_SPOKEN_MAX_CHARS if max_chars is None else max(1, int(max_chars))

    lines    = text.splitlines()
    has_list = any(_LIST_LINE_RE.match(ln) for ln in lines)

    # The only two reasons to use the screen: it is a list (painful to hear, and
    # you cannot skim speech), or it is simply long. Blank lines and paragraph
    # shape are deliberately NOT triggers — a three-line reply of 90 characters
    # is a fine thing to just say.
    if not has_list and len(text) <= budget:
        return text, ""

    if has_list:
        # Everything before the first bullet is the lead-in the model wrote to
        # introduce the list, and that is exactly the sentence worth speaking.
        lead = []
        for ln in lines:
            if _LIST_LINE_RE.match(ln):
                break
            lead.append(ln)
        source = " ".join(" ".join(lead).split())

        # The bullets, flattened into something sayable. Computed for BOTH
        # shapes — with a lead-in and without — because the rule is the same
        # either way: if it is short enough and few enough to say, say it.
        #
        # Taken from the first bullet ONWARDS rather than from the bullets
        # alone, so a trailing prose line after the list ("That's all.") is
        # still spoken instead of being silently dropped. The spoken line has to
        # stand alone, and a line quietly missing from it does not.
        # The bullets and the trailing prose are kept APART, and that separation
        # is the whole point of this block. Found 2026-08-17 23:47 by
        # `voice-split-fix` while making voice.py's streaming splitter agree with
        # this one: joining everything from the first bullet onwards with ", "
        # comma-splices a closing SENTENCE onto the list —
        #
        #   "You need:" / "- milk" / "- eggs" / "I would do the first one
        #    tonight, sir, because the disk is nearly full."
        #     -> "You need: milk, eggs, I would do the first one tonight, sir, …"
        #
        # which sounds like a fourth item that runs for a paragraph. The tail is
        # its own sentence, so it is joined with a full stop and a space.
        first = next(i for i, ln in enumerate(lines) if _LIST_LINE_RE.match(ln))
        items: list[str] = []
        tail:  list[str] = []
        for ln in lines[first:]:
            if not ln.strip():
                continue
            if _LIST_LINE_RE.match(ln):
                # A bullet AFTER prose means that prose was an aside inside the
                # list, not a closing line. Fold it back in where it stood
                # rather than reordering what the model wrote. Its full stop
                # goes, because it is about to be followed by ", " and ".," is a
                # stumble to a TTS engine. ! and ? are kept: they carry meaning
                # a comma cannot, and the sentence is no longer final anyway.
                items.extend(t[:-1].rstrip() if t.endswith(".") else t
                             for t in tail)
                tail = []
                items.append(" ".join(_LIST_LINE_RE.sub("", ln).split()))
            else:
                tail.append(" ".join(ln.split()))
        items = [i for i in items if i]
        tail  = [t for t in tail if t]
        flat  = ", ".join(items)
        if flat and flat[-1] not in ".!?":
            flat += "."
        if tail:
            flat = (flat + " " + " ".join(tail)).strip()
        # The tail is prose, not an item, so it does not count against
        # HUD_SPOKEN_MAX_ITEMS — only against the char budget, which is checked
        # by both callers below.
        few_enough = bool(items) and len(items) <= HUD_SPOKEN_MAX_ITEMS

        if not source:
            # A reply that is nothing but bullets, with no lead-in to speak.
            #
            # This used to hand off unconditionally, and that was wrong: it
            # withheld "- milk\n- eggs" — thirteen characters — from the voice
            # entirely, so a voice-first user standing away from the monitor got
            # nothing but "It is on the screen." That contradicts the rule at the
            # top of this section, which is the one rule here that matters: the
            # spoken line has to stand alone.
            #
            # So: if it is short enough AND few enough to say, say it. Hand off
            # only when hearing it really is worse than reading it.
            if few_enough and len(flat) <= budget:
                if flat[-1] not in ".!?":
                    flat += "."
                # Nothing was withheld, so there is nothing to hand off — but the
                # panel still gets it, because a list is worth a glance even when
                # it has been spoken in full.
                return flat, text
            return HUD_SCREEN_ONLY_LINE, text

        # A lead-in AND a short list — the same case, one line further on, and it
        # was still broken after the bullets-only fix:
        #
        #   split_reply("You need:\n- milk\n- eggs")
        #     -> "You need. The rest is on the screen."
        #
        # The colon-closing below turns the lead-in into the fragment "You need."
        # and then sends him to a monitor to be told "milk, eggs" — eleven
        # characters. So the flattening applies here too: the colon becomes the
        # join rather than a full stop, and the whole thing is spoken as one
        # sentence, "You need milk, eggs."
        #
        # Both guards still apply, and they catch different things: the char
        # budget, and HUD_SPOKEN_MAX_ITEMS, because the cost of hearing a list
        # scales with the number of items rather than with its length.
        if few_enough:
            # The colon is KEPT here rather than closed, because it is no longer
            # trailing — the items follow it, so the promise is kept rather than
            # broken. Dropping it garbled the lead-ins that are not verb phrases:
            # "Here are the steps:" + "milk, eggs" became "Here are the steps
            # milk, eggs". A colon is a pause to a TTS engine and correct
            # grammar to a reader.
            joined = (source + " " + flat).strip()
            if len(joined) <= budget:
                if joined[-1] not in ".!?":
                    joined += "."
                # Spoken in full, so nothing to hand off — the panel still gets
                # it, because a list is worth a glance either way.
                return joined, text
    else:
        source = " ".join(text.split())

    # Whole sentences until the next one would overrun; always at least one,
    # however long that one is.
    parts  = [p for p in _SENTENCE_SPLIT_RE.split(source) if p]
    spoken = parts[0] if parts else source
    for part in parts[1:]:
        if len(spoken) + 1 + len(part) > budget:
            break
        spoken += " " + part
    spoken = spoken.strip()

    # A trailing colon is a promise the voice cannot keep — "You need:" and then
    # silence. Closing it with a full stop was the first fix and it was worse
    # than the disease: it spoke the bare fragment "You need." Complete it with
    # the handoff clause instead, so the colon introduces something real —
    # "You need: the rest is on the screen." Promise kept, no fragment, and the
    # phrase is not then appended a second time.
    #
    # This path is only reached for a list too long or too many-itemed to say
    # (HUD_SPOKEN_MAX_ITEMS), or for prose over budget. A short list with a
    # lead-in never gets here; it is spoken in full above.
    hands_off = "on the screen" not in spoken.lower()
    if spoken.endswith(":"):
        if hands_off:
            spoken = (spoken[:-1].rstrip() + ": "
                      + HUD_HANDOFF_PHRASE[0].lower() + HUD_HANDOFF_PHRASE[1:])
            hands_off = False
        else:
            spoken = spoken[:-1].rstrip() + "."
    # Matched on the bare word "screen" until 2026-08-17, which silently ate the
    # handoff for anything that merely mentioned one — "Your screen is 1920 by
    # 1080…" published a panel and never said so. Worst on the vision path
    # ("look at my screen"), whose replies are long and descriptive by nature and
    # therefore exactly the ones that get a panel.
    if hands_off:
        spoken = spoken + " " + HUD_HANDOFF_PHRASE
    return spoken, text


def send_reply(text: str, title: str = "ANSWER", ttl_s: int | None = None,
               max_chars: int | None = None) -> str:
    """
    Publish the written half of a reply and return the half worth speaking.

    One call site does the whole thing:

        spoken = hud.send_reply(response)
        speak(tts, spoken)

    Returns `text` unchanged, and shows nothing, when the reply is short enough
    to simply say — so wiring this in cannot make a short answer worse.
    """
    spoken, detail = split_reply(text, max_chars)
    if detail:
        send_detail(detail, title=title, ttl_s=ttl_s)
    return spoken


def send_thinking(text: str) -> None:
    """Live reasoning stream for the HUD ('watch it think'). Empty clears it."""
    broadcast({"type": "reasoning", "text": text or ""})


# ─── Periodic pushers (plain threads, so psutil's blocking calls don't stall
#     the asyncio event loop) ────────────────────────────────────────────────

def _stats_loop() -> None:
    """Push system stats + clock roughly once per second."""
    while True:
        try:
            cpu  = psutil.cpu_percent(interval=None)
            ram  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            stats = {
                "type":         "stats",
                "cpu":          round(cpu, 1),
                "ram":          ram.percent,
                "ram_used_gb":  round(ram.used / 1e9, 1),
                "ram_total_gb": round(ram.total / 1e9, 1),
                "disk":         disk.percent,
            }

            # GPU stats reuse the existing nvidia-smi path in tools (best-effort)
            gpu = _gpu_stats()
            if gpu:
                stats.update(gpu)

            broadcast(stats)

            now = datetime.now()
            broadcast({
                "type": "time",
                "time": now.strftime("%I:%M %p").lstrip("0"),
                "date": now.strftime("%a, %b %d"),
                "iso":  now.isoformat(),
            })
        except Exception as exc:
            log.debug("stats loop error: %s", exc)

        time.sleep(1.0)


_NVIDIA_SMI: str | None | bool = False   # False = not yet resolved


def _nvidia_smi_path() -> str | None:
    """
    Absolute path to nvidia-smi.exe, or None if it isn't installed.

    Never invoke it by bare name: Windows CreateProcess searches the *current
    directory* before PATH, so a planted nvidia-smi.exe in the working directory
    would be run once per second. shutil.which() has the same flaw on Windows
    (it prepends the cwd), so the search is done explicitly here and the cwd is
    excluded. Resolved once and cached.
    """
    global _NVIDIA_SMI
    if _NVIDIA_SMI is not False:
        return _NVIDIA_SMI  # type: ignore[return-value]

    candidates = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
    ]
    cwd = Path.cwd().resolve()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip('" ')
        if not entry:
            continue
        try:
            d = Path(entry).resolve()
        except OSError:
            continue
        if d == cwd:
            continue  # the exact hazard we're avoiding
        candidates.append(d / "nvidia-smi.exe")

    _NVIDIA_SMI = None
    for c in candidates:
        try:
            if c.is_file():
                _NVIDIA_SMI = str(c)
                log.debug("nvidia-smi resolved to %s", _NVIDIA_SMI)
                break
        except OSError:
            continue
    if _NVIDIA_SMI is None:
        log.debug("nvidia-smi not found — GPU stats disabled.")
    return _NVIDIA_SMI


def _gpu_stats() -> dict | None:
    """Lightweight GPU read via nvidia-smi (returns None if unavailable)."""
    import subprocess
    exe = _nvidia_smi_path()
    if not exe:
        return None
    try:
        smi = subprocess.run(
            [exe,
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, shell=False,
            # Windows: suppress the console window that would otherwise flash
            # once per second under pythonw (no parent console).
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if smi.returncode == 0:
            t, u, mu, mt = [p.strip() for p in smi.stdout.strip().split(",")]
            return {
                "gpu":          int(u),
                "gpu_temp":     int(t),
                "gpu_mem_used": int(mu),
                "gpu_mem_total": int(mt),
            }
    except Exception:
        pass
    return None


def _weather_loop() -> None:
    """Refresh weather every 10 minutes."""
    while True:
        try:
            raw = T.get_weather(config.DEFAULT_LOCATION)
            data = json.loads(raw)
            wx = {
                "type":      "weather",
                "location":  data.get("location", "").split(",")[0],
                "temp_f":    data.get("temp_f"),
                "temp_c":    data.get("temp_c"),
                "condition": data.get("condition"),
                "humidity":  data.get("humidity_pct"),
                "wind_kmh":  data.get("wind_kmh"),
            }

            def _apply(wx=wx):
                global _last_weather
                _last_weather = wx

            broadcast(wx, _apply)
        except Exception as exc:
            log.debug("weather loop error: %s", exc)
        time.sleep(600)


# ─── Startup ──────────────────────────────────────────────────────────────────

def _run_server() -> None:
    global _loop
    import websockets

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def _serve():
        async with websockets.serve(_handler, HOST, PORT,
                                    process_request=_process_request):
            log.info("HUD WebSocket server listening on ws://%s:%d", HOST, PORT)
            # Printed to the local console only, never logged to file: it is
            # the URL to open the HUD with, including this install's token.
            print(f"HUD: open wallpaper/index.html#token={HUD_TOKEN}")
            await asyncio.Future()  # run forever

    try:
        _loop.run_until_complete(_serve())
    except Exception as exc:
        log.error("HUD server failed: %s", exc)
        alerts.send_alert("hud_server", type(exc).__name__,
                          f"WebSocket server stopped on ws://{HOST}:{PORT}.")


def _reload_watcher() -> None:
    """
    Watch the HUD wallpaper files and broadcast a reload when one changes, so the
    live wallpaper refreshes itself — no manual re-import in Lively after edits.
    """
    here  = Path(__file__).resolve().parent
    paths = [here / "wallpaper" / "index.html", here / "wallpaper_h" / "index.html"]
    mtimes = {}
    for p in paths:
        try:
            mtimes[p] = p.stat().st_mtime
        except OSError:
            mtimes[p] = None
    while True:
        time.sleep(2)
        for p in paths:
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if mtimes.get(p) is not None and m != mtimes[p]:
                log.info("HUD file changed (%s) — broadcasting reload.", p.name)
                broadcast({"type": "reload"})
            mtimes[p] = m


def start() -> None:
    """Start the HUD server + periodic pushers as daemon threads."""
    threading.Thread(target=_run_server,   daemon=True, name="hud-server").start()
    threading.Thread(target=_stats_loop,   daemon=True, name="hud-stats").start()
    threading.Thread(target=_weather_loop, daemon=True, name="hud-weather").start()
    threading.Thread(target=_reload_watcher, daemon=True, name="hud-reload").start()
    log.info("HUD bridge started.")


# ─── Standalone test mode ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    start()
    print(f"HUD bridge running on ws://{HOST}:{PORT}")
    print("Cycling demo states every 3s. Ctrl+C to stop.")
    states = ["idle", "listening", "thinking", "speaking"]
    i = 0
    try:
        while True:
            set_state(states[i % len(states)])
            if states[i % len(states)] == "speaking":
                send_transcript("user", "What's the weather?")
                send_transcript("bastion", "It's 68 degrees and clear in Springfield, sir.")
            i += 1
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nStopped.")
