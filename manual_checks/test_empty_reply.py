"""The silent turn — correctness harness (no real model, no audio).

On 2026-08-18 at 19:01 Bastion turned the owner's TV on, then looped on
`tv_control status` until it hit `Tool call limit reached (6)`, and returned
`""`. He heard NOTHING — no answer, no error — and reported it as "taking
forever", because from the outside a silent turn and a crashed one are the
same event.

Two things caused it and both are under test here:

  1. The closing round after the tool-call cap was documented as "without
     tools" but was still handed T.TOOL_DEFINITIONS, so the model answered the
     cap with yet another tool call and no words at all. `allow_tools=False`
     withholds them.
  2. Nothing checked that the final string was non-empty, and — the subtler
     half — with streaming TTS a fallback that is RETURNED but never pushed
     through on_content is a fallback nobody hears.

Like test_stall.py this runs against a real socket speaking Ollama's streaming
NDJSON rather than a mock of the client, so the tools-withheld property is
proved on the WIRE: the fake server records whether each request carried tool
definitions.

Usage: python test_empty_reply.py
"""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

# chat() alerts the owner's phone on some failure paths. Nothing here should fire
# one, but stub it before importing bastion so a regression cannot reach him.
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
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))


# ── A fake Ollama that answers everything with a tool call ───────────────────
# The rule it plays by is the one the real bug turned on: WHILE IT IS OFFERED
# TOOLS it calls one, forever. So the only thing that can end the turn is the
# closing round genuinely withholding them. If allow_tools ever regresses, this
# server will happily loop until the cap and hand back "" again.

TOOL_NAME = "not_a_real_tool"   # dispatch() returns "Unknown tool: ..." — no side effects


def _ndjson(content="", done=False, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps({
        "model": config.MODEL,
        "created_at": "2026-08-18T19:01:00Z",
        "message": msg,
        "done": done,
        **({"done_reason": "stop"} if done else {}),
    }).encode() + b"\n"


class Recorder:
    """What the server was asked, in order. One entry per /api/chat POST."""

    def __init__(self):
        self.requests = []

    @property
    def with_tools(self):
        return [r for r in self.requests if r["tools"]]

    @property
    def without_tools(self):
        return [r for r in self.requests if not r["tools"]]


def _handler(rec, mode, tool_round_content, closing_content, no_tool_content,
             resident):
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
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                body = {}
            offered = body.get("tools") or []
            rec.requests.append({
                "tools": list(offered),
                "messages": body.get("messages") or [],
                "think": body.get("think"),
            })

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            if mode == "loop" and offered:
                # Tools in hand → answer with a tool call. This is what the real
                # model did six times in a row.
                if tool_round_content:
                    self._chunk(_ndjson(tool_round_content))
                self._chunk(_ndjson("", True, [
                    {"function": {"name": TOOL_NAME, "arguments": {}}}]))
            else:
                # Either the closing round (tools withheld) or a plain turn from
                # a server that never calls a tool at all.
                text = closing_content if mode == "loop" else no_tool_content
                for piece in _split(text):
                    self._chunk(_ndjson(piece))
                self._chunk(_ndjson("", True))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return H


def _split(text, n=3):
    """Emit in a few chunks, the way a real stream arrives."""
    if not text:
        return []
    size = max(1, len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), size)]


def fake_ollama(mode="loop", tool_round_content="", closing_content="",
                no_tool_content="", resident=True):
    """mode="loop": call a tool on every round that is offered tools (the bug).
       mode="answer": never call a tool, just reply with no_tool_content."""
    rec = Recorder()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = ThreadingHTTPServer(
        ("127.0.0.1", port),
        _handler(rec, mode, tool_round_content, closing_content, no_tool_content,
                 resident))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    config.OLLAMA_HOST = f"http://127.0.0.1:{port}"
    bastion._clients.clear()      # cached per timeout, so drop the old host's client
    return srv, rec


def run_turn(user_input="what is on my tv", stream=True):
    """One chat() turn. Returns (reply, streamed_text, messages)."""
    deltas = []
    sink = (lambda d: deltas.append(d)) if stream else None
    reply, msgs = bastion.chat([{"role": "system", "content": "sys"}],
                              user_input, on_content=sink)
    return reply, "".join(d for d in deltas if d), msgs


# Keep the deadlines short: nothing here is meant to stall, and a wrong turn
# should fail the run in seconds rather than sitting on the real 45 s budget.
bastion._STALL_S, bastion._COLD_S = 5.0, 5.0

CAP = config.MAX_TOOL_CALLS_PER_TURN


# ── 1. the turn that actually happened ───────────────────────────────────────

print("\n─ the silent turn, reproduced ─")

srv, rec = fake_ollama(closing_content="")      # closing round says nothing
t0 = time.perf_counter()
reply, streamed, msgs = run_turn()
secs = time.perf_counter() - t0
srv.shutdown()

check("the tool loop ends instead of running forever", secs < 30, f"{secs:.1f}s")
check("a turn that exhausts the tool cap still returns WORDS",
      bool(reply.strip()), repr(reply))
check("...specifically the tool-limit line", reply == config.TOOL_LIMIT_MESSAGE,
      repr(reply))
check("the fallback is SPOKEN, not merely returned",
      config.TOOL_LIMIT_MESSAGE in streamed,
      f"streamed={streamed!r} — a string voice.py never receives is unheard")
check("the loop really did hit the cap",
      len(rec.with_tools) > CAP, f"{len(rec.with_tools)} tool rounds, cap {CAP}")
check("every round before the cap was offered its tools",
      all(r["tools"] for r in rec.requests[:-1]), str(len(rec.with_tools)))
check("the CLOSING round is called with the tools withheld",
      len(rec.without_tools) == 1 and rec.requests[-1] is rec.without_tools[0],
      "tools=None is the only thing that makes this round produce words")
check("exactly one round follows the cap — it cannot loop again",
      len(rec.without_tools) == 1, str(len(rec.without_tools)))
closing_msgs = rec.requests[-1]["messages"]
check("the closing round carries the answer nudge",
      any("all the tool calls allowed" in (m.get("content") or "")
          for m in closing_msgs), "the _CLOSING_ANSWER_MSG system message")
check("the closing round does not reason again", rec.requests[-1]["think"] in (False, None),
      str(rec.requests[-1]["think"]))
check("history ends with the spoken line, not an empty assistant turn",
      msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == reply,
      repr(msgs[-1]))
check("the transient nudge is not left in the history",
      not any("all the tool calls allowed" in (m.get("content") or "")
              for m in msgs), "it must not shape the NEXT turn")


# ── 2. a closing round that does speak is left alone ─────────────────────────
# The fallback is a floor, not a filter. If the model finds words after the cap
# they must reach the user verbatim — substituting the canned line over a real
# answer would be a worse bug than the one being fixed.

print("\n─ a closing round with something to say ─")

ANSWER = "The television is on, sir, and showing the input you asked for."
srv, rec = fake_ollama(closing_content=ANSWER)
reply, streamed, _msgs = run_turn()
srv.shutdown()

check("a real closing answer is returned untouched", reply == ANSWER, repr(reply))
check("...and no fallback is bolted onto it",
      config.TOOL_LIMIT_MESSAGE not in reply, repr(reply))
check("...and the fallback is not spoken either",
      config.TOOL_LIMIT_MESSAGE not in streamed, repr(streamed))
check("the real answer streamed as it arrived", ANSWER in streamed, repr(streamed))


# ── 3. an empty reply with no tools involved ─────────────────────────────────

print("\n─ an ordinary turn that says nothing ─")

srv, rec = fake_ollama("answer", no_tool_content="")
reply, streamed, msgs = run_turn("who are you")
srv.shutdown()

check("one round, no tool call, no closing round",
      len(rec.requests) == 1 and bool(rec.requests[0]["tools"]),
      f"{len(rec.requests)} rounds — this path never reaches the cap")
check("an empty ordinary reply becomes the empty-reply line",
      reply == config.EMPTY_REPLY_MESSAGE, repr(reply))
check("...and it is spoken", config.EMPTY_REPLY_MESSAGE in streamed, repr(streamed))
check("...and it is the EMPTY line, not the tool-limit one",
      config.TOOL_LIMIT_MESSAGE not in reply,
      "the two failures are different and must not be conflated")
check("history records the spoken line", msgs[-1]["content"] == reply)


# ── 4. whitespace is not an answer ───────────────────────────────────────────
# A model that emits " \n\t " has said nothing. Kokoro renders it as silence,
# which is the exact failure being fixed, so it must count as empty.

print("\n─ whitespace only ─")

srv, rec = fake_ollama("answer", no_tool_content="   \n\t   \n ")
reply, streamed, _msgs = run_turn("who are you")
srv.shutdown()

check("whitespace-only counts as empty", reply == config.EMPTY_REPLY_MESSAGE,
      repr(reply))
check("...and the fallback still reaches the speaker",
      config.EMPTY_REPLY_MESSAGE in streamed, repr(streamed))


# ── 5. a normal reply is untouched ───────────────────────────────────────────

print("\n─ a normal reply ─")

NORMAL = "I'm Bastion, sir."
srv, rec = fake_ollama("answer", no_tool_content=NORMAL)
reply, streamed, _msgs = run_turn("who are you")
srv.shutdown()

check("a non-empty reply is passed through byte for byte", reply == NORMAL,
      repr(reply))
check("no fallback text is added",
      config.EMPTY_REPLY_MESSAGE not in reply and config.TOOL_LIMIT_MESSAGE not in reply)
check("it is streamed exactly once, not doubled by the fallback path",
      streamed.count(NORMAL[:8]) == 1, repr(streamed))


# ── 6. the action-shaped path, where nothing is streamed until the end ───────
# "start the tv" matches _ACTION_RE, so chat() withholds on_content from every
# loop round in case a forced tool retry discards the text. That is exactly the
# shape of the real incident, and it is the case where a returned-but-unstreamed
# fallback would be silent. The closing round must stream regardless.

print("\n─ an action command that burns the cap ─")

srv, rec = fake_ollama(closing_content="")
reply, streamed, _msgs = run_turn("start the tv")
srv.shutdown()

check("an action command still gets a spoken answer",
      config.TOOL_LIMIT_MESSAGE in streamed,
      "loop rounds are unstreamed here, so the closing round is the only voice")
check("...and returns it as well", reply == config.TOOL_LIMIT_MESSAGE, repr(reply))
check("the closing round withheld tools on this path too",
      len(rec.without_tools) == 1, str(len(rec.without_tools)))


# ── 7. _stream_round(allow_tools=...) on the wire ────────────────────────────

print("\n─ allow_tools ─")

srv, rec = fake_ollama("answer", no_tool_content="hello")
bastion._stream_round([{"role": "user", "content": "hi"}], False, None, None,
                     allow_tools=False)
check("allow_tools=False sends no tool definitions at all",
      rec.requests[-1]["tools"] == [], str(len(rec.requests[-1]["tools"])))

bastion._stream_round([{"role": "user", "content": "hi"}], False, None, None)
check("the default still offers the full tool schema",
      len(rec.requests[-1]["tools"]) == len(bastion.T.TOOL_DEFINITIONS),
      f"{len(rec.requests[-1]['tools'])} vs {len(bastion.T.TOOL_DEFINITIONS)}")
srv.shutdown()


# ── 8. _nonempty_reply itself ────────────────────────────────────────────────

print("\n─ _nonempty_reply ─")

FB = "fallback line"


def spoken(content, on_content=None):
    got = []
    out = bastion._nonempty_reply(content, on_content or got.append, FB)
    return out, got


out, got = spoken("a real answer")
check("real content is returned unchanged", out == "a real answer")
check("...and nothing extra is pushed to the speaker", got == [], str(got))

for label, value in (("empty string", ""), ("None", None),
                     ("spaces", "   "), ("newlines and tabs", "\n\t \r\n")):
    out, got = spoken(value)
    check(f"{label} is treated as empty", out == FB, repr(out))
    check(f"...and the fallback is pushed to on_content for {label}", got == [FB],
          str(got))


def _raises(_d):
    raise RuntimeError("the speaker died")


check("an on_content that raises does not take the turn down",
      bastion._nonempty_reply("", _raises, FB) == FB)
check("no on_content at all still returns the fallback",
      bastion._nonempty_reply("", None, FB) == FB)
check("a zero-length fallback is still returned, not looped on",
      bastion._nonempty_reply("", None, "") == "")


# ── 9. the lines themselves ──────────────────────────────────────────────────

print("\n─ the spoken lines ─")

check("the tool-limit line exists and is speakable",
      bool(config.TOOL_LIMIT_MESSAGE.strip()) and len(config.TOOL_LIMIT_MESSAGE) > 20)
check("the empty-reply line exists and is speakable",
      bool(config.EMPTY_REPLY_MESSAGE.strip()) and len(config.EMPTY_REPLY_MESSAGE) > 20)
check("they are different lines",
      config.TOOL_LIMIT_MESSAGE != config.EMPTY_REPLY_MESSAGE,
      "the user should be able to tell the two failures apart")
check("neither claims the action succeeded or failed",
      not any(w in (config.TOOL_LIMIT_MESSAGE + config.EMPTY_REPLY_MESSAGE).lower()
              for w in ("succeeded", "failed", "worked", "didn't work")),
      "nothing in the loop knows which it was at that point")
check("no alert was fired by any of this",
      SENT_ALERTS == [], str(SENT_ALERTS))
check("the harness never armed a real alert", alerts.send_alert is not _real_send_alert)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
