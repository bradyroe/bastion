"""Ollama stall deadline — correctness harness (no real model, no audio).

Bastion had NO timeout on the model call at all. On 2026-08-17 at 22:36 Ollama
wedged and Bastion sat mute for over three minutes: no error, no log line, and
from the owner's side it looked exactly like thinking.

This proves the fix against a real socket rather than a mock of one — a
deliberately wedged HTTP server that speaks Ollama's streaming NDJSON. The
property under test is that the deadline measures SILENCE, not duration: a
20-second generation that keeps producing tokens must survive a 2-second stall
budget, or every real turn with long reasoning dies.

Usage: python test_stall.py
"""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

# bastion.chat() alerts the owner's phone on a stall — which is the point of the
# feature and must NOT happen from a test run. Stub it and record the calls.
import alerts

SENT_ALERTS = []
_real_send_alert = alerts.send_alert
alerts.send_alert = lambda *a, **k: SENT_ALERTS.append((a, k))

import bastion

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# ── A wedged Ollama ──────────────────────────────────────────────────────────

def _ndjson(content="", done=False):
    return json.dumps({
        "model": config.MODEL,
        "created_at": "2026-08-17T22:36:00Z",
        "message": {"role": "assistant", "content": content},
        "done": done,
        **({"done_reason": "stop"} if done else {}),
    }).encode() + b"\n"


def _handler(mode, resident, gap, count):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _chunk(self, payload):
            self.wfile.write(hex(len(payload))[2:].encode() + b"\r\n"
                             + payload + b"\r\n")
            self.wfile.flush()

        def do_GET(self):
            if self.path.startswith("/api/ps"):
                loaded = [{"name": config.MODEL, "model": config.MODEL, "size": 1,
                           "digest": "d", "size_vram": 1, "details": {}}]
                self._json({"models": loaded if resident else []})
            else:
                self._json({"models": []})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                if mode in ("partial", "slow"):
                    for i in range(count):
                        self._chunk(_ndjson(f"word{i} "))
                        time.sleep(gap)
                if mode == "slow":
                    self._chunk(_ndjson("", True))
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                # WEDGE: hold the connection open and send nothing, ever.
                time.sleep(120)
            except Exception:
                pass

    return H


def wedged_ollama(mode, resident=True, gap=0.1, count=3):
    """Start a fake Ollama on a free port and point config at it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), _handler(mode, resident, gap, count))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    config.OLLAMA_HOST = f"http://127.0.0.1:{port}"
    bastion._clients.clear()      # cached per timeout, so drop the old host's client
    return srv


def run_turn(user_input="what are you up to right now", stream=True):
    """One chat() turn against whatever fake server is running. Returns
    (reply, streamed_text, seconds)."""
    deltas = []
    sink = (lambda d: deltas.append(d)) if stream else None
    t0 = time.perf_counter()
    reply, _msgs = bastion.chat([{"role": "system", "content": "sys"}],
                               user_input, on_content=sink)
    return reply, "".join(d for d in deltas if d), time.perf_counter() - t0


STALL, COLD = 2.0, 5.0
bastion._STALL_S, bastion._COLD_S = STALL, COLD
STALL_LINE = bastion._STALL_MESSAGE.split(",")[0]

# ── 1. a wedge is caught, and Bastion says so ─────────────────────────────────

print("\n─ a wedged model ─")

srv = wedged_ollama("silent")
reply, streamed, secs = run_turn()
srv.shutdown()

check("gives up instead of hanging forever", secs < STALL + 3, f"{secs:.1f}s")
check("waits the whole stall budget first", secs >= STALL, f"{secs:.1f}s")
check("returns something to say, not an empty string", bool(reply.strip()), repr(reply))
check("says the model stopped responding", "stopped responding" in reply.lower(), repr(reply))
check("nothing was streamed, so voice.py speaks the returned line", streamed == "")

# ── 2. the alert ─────────────────────────────────────────────────────────────

print("\n─ the alert ─")

stall_alerts = [a for a, _k in SENT_ALERTS if a[1] == "ModelStalled"]
check("exactly one alert per stall", len(stall_alerts) == 1, str(SENT_ALERTS))
check("alert names the component", stall_alerts and stall_alerts[0][0] == "bastion.chat")
detail = stall_alerts[0][2].lower() if stall_alerts else ""
check("alert carries model and host", config.MODEL in detail and "127.0.0.1" in detail, detail)
check("harness fired no real alerts", alerts.send_alert is not _real_send_alert)

# Nothing the user said may leave the machine — alerts.py's one rule.
SENT_ALERTS.clear()
srv = wedged_ollama("silent")
secret = "my bank password is hunter2 and I live at 14 Elm Street"
run_turn(secret)
srv.shutdown()
blob = " ".join(" ".join(str(x) for x in a) for a, _k in SENT_ALERTS).lower()
leaked = [w for w in ("hunter2", "elm", "password", "bank") if w in blob]
check("no user speech in the alert", not leaked, f"leaked {leaked}: {blob}")

# ── 3. a SLOW turn is not a stalled one ──────────────────────────────────────
# The whole design rests on this. Thinking alone was measured at 5-10 s routinely
# and 29.4 s at the worst observed, so a total-duration cap would kill healthy
# turns. Eight chunks 1.2 s apart run ~10 s against a 2 s budget.

print("\n─ a slow but healthy turn ─")

srv = wedged_ollama("slow", gap=1.2, count=8)
reply, streamed, secs = run_turn()
srv.shutdown()

check("a generation far longer than the budget survives", "word7" in reply,
      f"{secs:.1f}s {reply!r}")
check("every chunk arrived", all(f"word{i}" in reply for i in range(8)), repr(reply))
check("it really did outlast the budget", secs > STALL * 2, f"{secs:.1f}s")
check("the answer streamed as it arrived", "word0" in streamed)

# ── 4. a stall AFTER speech has started ──────────────────────────────────────
# voice.py speaks the returned string only when no deltas reached it. Once half a
# sentence is in the air that fallback cannot fire, so the failure line has to go
# down the same channel it is already speaking from.

print("\n─ stalling mid-sentence ─")

srv = wedged_ollama("partial", gap=0.1, count=3)
reply, streamed, secs = run_turn()
srv.shutdown()

check("still caught once bytes have flowed", "stopped responding" in reply.lower(), repr(reply))
check("the deadline restarted on each chunk", secs >= STALL, f"{secs:.1f}s")
check("what was already generated was spoken", "word0" in streamed)
check("the failure line is spoken too, not only returned",
      STALL_LINE.lower() in streamed.lower(), repr(streamed[-80:]))

# ── 5. a cold load is not a wedge ────────────────────────────────────────────
# A model that is not resident sends nothing while it loads, measured at over
# three minutes off the owner's drive. Silence there is legitimate, so residency
# decides the budget.

print("\n─ cold load ─")

srv = wedged_ollama("silent", resident=False)
reply, _s, secs = run_turn()
srv.shutdown()

check("a non-resident model gets the longer budget", secs >= COLD, f"{secs:.1f}s")
check("but it is still bounded", secs < COLD + 3, f"{secs:.1f}s")
check("and it still reports the failure", "stopped responding" in reply.lower())

srv = wedged_ollama("silent", resident=True)
# Which budget this turn gets is decided by the /api/ps probe, and
# _model_resident() (bastion.py:351) answers False for ANY failure — a blipped
# probe silently turns this into a cold-budget turn, and the timing check below
# then fails for a reason that has nothing to do with the deadline. Asserting
# the precondition separately tells the two apart instead of leaving a check
# that looks intermittently broken.
#
# Not hypothetical: forcing exactly this one probe to fail reproduces
# "24 passed, 1 failed" with this as the only failure, which is the result
# reported on 2026-08-18 with no code change between runs.
check("the residency probe answers, so the warm budget is the one under test",
      bastion._model_resident() is True,
      "the /api/ps probe failed — the deadline below was never exercised")
_r, _s, warm_secs = run_turn()
srv.shutdown()
check("a resident model is not given the cold allowance", warm_secs < COLD,
      f"{warm_secs:.1f}s vs cold {COLD}s")

# ── 6. the settings are sane ─────────────────────────────────────────────────

print("\n─ configuration ─")

live_stall = float(getattr(config, "OLLAMA_STALL_TIMEOUT_S", 45.0))
live_cold = float(getattr(config, "OLLAMA_COLD_LOAD_TIMEOUT_S", 300.0))
check("the stall budget clears the worst measured thinking time (29.4 s)",
      live_stall > 29.4, f"{live_stall}s")
check("the cold budget clears the measured cold load (>3 min)",
      live_cold >= 240, f"{live_cold}s")
check("cold is the more generous of the two", live_cold > live_stall)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
