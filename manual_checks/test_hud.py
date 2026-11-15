"""
Start the HUD bridge, connect a client, and check the wire contract.

Covers the original "do messages flow at all" smoke check plus, since
2026-08-17, the two contract additions the co-pilot stage depends on:

  * every "state" frame carries "since" — the epoch second the state was
    ENTERED. The wallpaper counts up from it, so a reload mid-think shows the
    true wait instead of restarting at zero;
  * the "transcribing" and "answering" values, which close the two silent gaps
    in a turn (Whisper running after the mic shuts, and the model writing the
    answer after it has stopped reasoning).

Plus split_reply(), which decides what Bastion says out loud and what it puts on
the screen. That one is pure and needs no socket, so it runs first and fails
fast.

And, since the horizontal HUD was ported on 2026-08-17, a static check of BOTH
wallpapers against the contract. There are two of them — wallpaper/ on the
vertical panel and wallpaper_h/ on the working monitor — and the failure mode
is silent: a wallpaper that has not been taught a state, or whose token no
longer matches config.py, shows stale or blank data with no error anywhere.

Run it in a FRESH process. A HUD test that passes warm and fails cold has
happened here before.
"""
import asyncio
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import hud_server as hud

import websockets

NL = "\n"
_fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("  " + str(detail) if detail else ""))
    if not ok:
        _fails.append(label)
    return ok


# ─── Both wallpapers, statically (no socket, no browser) ──────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
WALLPAPERS = {
    "vertical  (wallpaper/index.html)":     os.path.join(HERE, "wallpaper", "index.html"),
    "horizontal (wallpaper_h/index.html)":  os.path.join(HERE, "wallpaper_h", "index.html"),
}


def wallpaper_tests():
    print("\nthe wallpapers — both sides of the contract, checked as text\n")
    for name, path in WALLPAPERS.items():
        if not check(f"{name} exists", os.path.isfile(path), path):
            continue
        with open(path, encoding="utf-8") as fh:
            src = fh.read()

        # The shared secret. Wrong here and the HUD sits on "Linking" forever,
        # which looks exactly like the server being down.
        check(f"{name}: HUD_TOKEN matches config.py",
              ('"' + hud.HUD_TOKEN + '"') in src)

        # Every state the server can emit must be handled by name, or the HUD
        # prints a raw value it was never taught.
        missing = [v for v in hud.HUD_STATES if v not in src]
        check(f"{name}: handles all {len(hud.HUD_STATES)} states by name",
              not missing, missing)

        # "since" is what keeps the elapsed counter honest across a reload.
        check(f"{name}: reads 'since' off the state frame", "since" in src)

        # The one lie this panel must not tell.
        check(f"{name}: resets the stage when the socket drops",
              'applyState("link")' in src)

        for step in ("Heard", "Thinking", "Answering", "Speaking"):
            if step not in src:
                check(f"{name}: rail step {step!r} present", False)
                break
        else:
            check(f"{name}: all four rail steps present", True)

        # Local-first, and it has to work offline: no remote assets, ever.
        remote = re.findall(r'(?:src|href)\s*=\s*["\']https?:', src) + \
                 re.findall(r'@import\s+url\(\s*["\']?https?:', src)
        check(f"{name}: self-contained, no remote assets", not remote, remote)

        # The queue band (added 2026-08-17). It is a separate message type on
        # purpose: "state" is single-valued, so a "queued" state would REPLACE
        # "thinking" on the stage and hide the running turn — which is the very
        # thing the owner could not see. Both must be renderable at once.
        check(f"{name}: handles the 'queue' message", '"queue"' in src)
        check(f"{name}: has a queue band to show it in", "queueBar" in src)
        check(f"{name}: reads 'waiting' off the queue frame", "waiting" in src)
        check(f"{name}: takes the queue band down on a dropped socket",
              "applyQueue(0)" in src)
        # The thought meter: what tells a 29-second think from a hang. Elapsed
        # time rises either way; a reasoning count only rises while it works.
        check(f"{name}: has the thought meter", "thoughtVal" in src and "thoughtFeed" in src)

        # The tool line (added 2026-08-18). Checked in BOTH files for the same
        # reason as the queue band, and this one has already bitten: the
        # horizontal wallpaper got the CSS and the applyTool function but not
        # the `case "tool":` in its applyData switch, so it was taught the
        # message and still ignored every one that arrived. Handling it means
        # BOTH — the renderer and the wiring.
        check(f"{name}: handles the 'tool' message", '"tool"' in src)
        check(f"{name}: has a renderer for it", "applyTool" in src)
        check(f"{name}: and actually calls it on an incoming frame",
              src.count("applyTool") >= 2,
              "one mention is a definition nobody calls — the exact bug that "
              "shipped in wallpaper_h on 2026-08-18")


# ─── The voice/screen split (no socket needed) ────────────────────────────────

def split_tests():
    print("\nsplit_reply — what gets spoken, what goes on the panel\n")

    spoken, detail = hud.split_reply("It is 68 and clear in Springfield, sir.")
    check("a short prose reply is spoken whole, panel stays dark",
          spoken == "It is 68 and clear in Springfield, sir." and detail == "")

    # Rewritten 2026-08-17 23:30. These two asserted the PRE-fix behaviour —
    # "Here are the steps. The rest is on the screen." for a 29-character list —
    # and were left behind when hud_server.py gained the lead-in+short-list
    # flattening at 23:16. Restoring them would re-break the HIGH bug the
    # reviewer filed: a reply inside the spoken budget is just spoken.
    spoken, detail = hud.split_reply("Here are the steps:" + NL + "- one" + NL + "- two")
    check("a lead-in plus a SHORT list is spoken in full, not handed off",
          spoken == "Here are the steps: one, two." and "- two" in detail,
          repr(spoken))
    # The colon is kept when the items follow it — the promise is kept, not
    # broken. What must never happen is a colon left dangling at the END of the
    # spoken line, which is silence where content was promised.
    check("the trailing colon is closed, not left hanging in speech",
          not spoken.rstrip().endswith(":"), repr(spoken))

    # The same lead-in with too many items to say: this DOES hand off, and the
    # fragment is the bug — "You need." was what it spoke before 23:30. The
    # colon now introduces the handoff clause instead of being amputated.
    spoken, detail = hud.split_reply("You need:" + NL
                                     + NL.join("- item %d" % i for i in range(7)))
    check("a lead-in plus a LONG list hands off",
          spoken.lower().endswith("on the screen.") and "- item 6" in detail,
          repr(spoken))
    check("...and the colon completes into the handoff, never a bare fragment",
          spoken == "You need: the rest is on the screen.", repr(spoken))
    check("...and the handoff is not then doubled",
          spoken.lower().count("on the screen") == 1, repr(spoken))

    # Found 2026-08-17 23:47 by voice-split-fix, while making voice.py's
    # streaming splitter agree with this one. The items were taken from the
    # first bullet ONWARDS — which is right, a closing line must not be dropped
    # — but then ", ".join'ed, so a closing SENTENCE was comma-spliced onto the
    # list and sounded like a fourth item running for a paragraph.
    spoken, detail = hud.split_reply(
        "You need:" + NL + "- milk" + NL + "- eggs" + NL
        + "I would do the first one tonight, sir, because the disk is nearly full.")
    check("a closing sentence after a list is a SENTENCE, not a fourth item",
          spoken == ("You need: milk, eggs. I would do the first one tonight, "
                     "sir, because the disk is nearly full."), repr(spoken))
    check("...and the same with no lead-in to speak",
          hud.split_reply("- a" + NL + "- b" + NL + "That is all, sir.")[0]
          == "a, b. That is all, sir.",
          repr(hud.split_reply("- a" + NL + "- b" + NL + "That is all, sir.")[0]))
    # Prose BETWEEN bullets is an aside, not a tail: it stays where the model put
    # it rather than being reordered to the end, and loses its full stop so the
    # spoken line does not stumble over ".,".
    check("prose between bullets stays in place and does not become '.,'",
          hud.split_reply("Steps:" + NL + "- one" + NL + "A note here." + NL
                          + "- two")[0] == "Steps: one, A note here, two.",
          repr(hud.split_reply("Steps:" + NL + "- one" + NL + "A note here."
                               + NL + "- two")[0]))

    # Changed 2026-08-17 after review. This previously asserted that a
    # bullets-only reply is NEVER read out, which withheld "- milk / - eggs" —
    # thirteen characters — from the voice entirely. A voice-first user away
    # from the monitor got nothing but "It is on the screen.", which
    # contradicts the one rule that matters: the spoken line stands alone.
    spoken, detail = hud.split_reply("- milk" + NL + "- eggs")
    check("a SHORT bullets-only reply is spoken in full, not withheld",
          spoken == "milk, eggs." and detail.count("-") == 2, repr(spoken))
    check("...and it still reaches the panel, because a list is worth a glance",
          bool(detail))

    # Two guards on that, and they fail for different reasons — a char budget
    # alone does not catch the second, because the cost of hearing a list
    # scales with the number of items rather than its length.
    long_items = NL.join(
        "- a considerably longer item name that runs on for a while, number %d" % i
        for i in range(4))
    check("a bullets-only reply over the CHAR budget still hands off",
          hud.split_reply(long_items)[0] == hud.HUD_SCREEN_ONLY_LINE)
    many_items = NL.join("- t%d" % i for i in range(20))
    check("20 one-word bullets (148 chars, under budget) still hand off",
          hud.split_reply(many_items)[0] == hud.HUD_SCREEN_ONLY_LINE,
          repr(hud.split_reply(many_items)[0]))

    long_prose = "One. " * 200
    spoken, detail = hud.split_reply(long_prose)
    check("long prose is cut to whole sentences within the budget",
          len(spoken) <= hud.HUD_SPOKEN_MAX_CHARS + len(hud.HUD_HANDOFF_PHRASE) + 1
          and spoken.startswith("One. One."), len(spoken))
    check("the panel gets the WHOLE reply, not the unspoken remainder",
          detail == long_prose.strip())

    # A single sentence longer than the budget is spoken anyway: half a sentence
    # does not stand alone, and the budget is a budget, not a limit.
    one_huge = "A" * 400 + "."
    spoken, detail = hud.split_reply(one_huge)
    check("one over-long sentence is never cut mid-sentence",
          spoken.startswith("A" * 400 + "."), len(spoken))

    # Rewritten 2026-08-17 23:30 with the two above. A two-item list is inside
    # the budget, so it is now spoken in full — including the items, which the
    # old expectation ("I put it on the screen." and nothing else) dropped on
    # the floor. The doubling guard is still real, but it belongs on the path
    # that actually appends the phrase, which is the check below it.
    spoken, detail = hud.split_reply("I put it on the screen." + NL + "- a" + NL + "- b")
    check("a short list is spoken in full even when the lead-in mentions a screen",
          spoken == "I put it on the screen. a, b." and detail.count("-") == 2,
          repr(spoken))
    spoken, _ = hud.split_reply("I put the numbers on the screen. "
                                + "Filler sentence here. " * 12)
    check("the handoff is not appended twice when the reply already says it",
          spoken.lower().count("on the screen") == 1, repr(spoken[-46:]))

    # The guard used to match the bare word "screen" anywhere, so any reply that
    # merely mentioned one got a panel it was never told about — worst on the
    # vision path, whose replies are long and descriptive and therefore exactly
    # the ones that get a panel.
    spoken, detail = hud.split_reply("Your screen is 1920 by 1080. "
                                     + "Filler sentence here. " * 12)
    check("a reply that merely mentions a screen still gets the handoff",
          detail and spoken.endswith(hud.HUD_HANDOFF_PHRASE), repr(spoken[-46:]))

    # The closer was part of the separator, so re.split deleted it outright.
    spoken, _ = hud.split_reply('She said "no." I agreed. ' + "Filler here. " * 20)
    check("the sentence splitter keeps a closing quote or bracket",
          spoken.startswith('She said "no." I agreed.'), repr(spoken[:30]))

    check("empty in, empty out", hud.split_reply("") == ("", ""))
    check("send_reply returns a short reply unchanged and shows nothing",
          hud.send_reply("Done, sir.") == "Done, sir.")


# ─── The wire ─────────────────────────────────────────────────────────────────

async def client():
    await asyncio.sleep(1.5)          # let the server bind
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        # The feed is authenticated — first frame must be the auth handshake.
        await ws.send(json.dumps({"type": "auth", "token": hud.HUD_TOKEN}))
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert hello.get("type") == "auth_ok", f"auth rejected: {hello}"
        print("\nConnected + authenticated. Listening for 8 seconds...\n")

        seen, states, sinces, queues = set(), [], [], []
        end = time.time() + 8
        while time.time() < end:
            try:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            except asyncio.TimeoutError:
                continue
            seen.add(data["type"])
            if data["type"] == "state":
                states.append(data.get("value"))
                sinces.append(data.get("since"))
            if data["type"] == "queue":
                queues.append(data)
            if data["type"] in ("state", "transcript", "detail", "queue"):
                print(f"  RX [{data['type']:10}] {json.dumps(data)[:96]}")

        print(f"\nMessage types received: {sorted(seen)}")
        check("core message types flowing", {"state", "stats", "time"}.issubset(seen),
              sorted(seen))
        check("transcripts flow", "transcript" in seen)
        check("send_reply published a detail panel", "detail" in seen)

        for value in ("transcribing", "answering"):
            check(f"state '{value}' reaches the wallpaper", value in states, states)

        check("every state frame carries 'since'",
              sinces and all(isinstance(x, (int, float)) for x in sinces), sinces[:3])
        now = time.time()
        check("'since' is a plausible recent epoch second, not a duration",
              all(0 <= now - x < 120 for x in sinces if isinstance(x, (int, float))))
        check("'since' never runs backwards across a turn",
              all(b >= a for a, b in zip(sinces, sinces[1:])))

        # ── the queue, on the wire ───────────────────────────────────────────
        check("'queue' frames reach the wallpaper", bool(queues), queues)
        if queues:
            check("a queue frame carries waiting/text/since",
                  all({"waiting", "text", "since"} <= set(q) for q in queues), queues[0])
            check("the queued question is on the wire, not just the count",
                  any("Spotify" in str(q.get("text", "")) for q in queues), queues)
            check("the queue is cleared to waiting=0 when the turn starts",
                  queues[-1].get("waiting") == 0, queues[-1])
            check("'since' on a live queue frame is an epoch second, not a duration",
                  all(0 <= now - q["since"] < 120
                      for q in queues if q.get("waiting") and isinstance(q.get("since"), (int, float))),
                  queues)
        # THE POINT, asserted rather than assumed: queueing a second turn must
        # not change the state of the first. A "queued" state value would have,
        # and that is why this is a separate message.
        check("queueing a turn never touches the state feed",
              "queued" not in states, states)


def drive_states():
    """Walk one whole turn, in the order voice.py walks it."""
    time.sleep(2)
    hud.set_state("listening");    time.sleep(0.4)
    hud.set_state("transcribing"); time.sleep(0.4)
    hud.send_transcript("user", "What is eating my VRAM?")
    hud.set_state("thinking");     time.sleep(0.4)
    # A second question arrives mid-think and has to wait — the 22:54 case that
    # made the owner think Bastion had hung. The state stays "thinking" throughout.
    hud.queue_enter("What are you playing right now on Spotify?")
    time.sleep(0.6)
    hud.queue_leave()              # it got the lock; the band comes down
    time.sleep(0.3)
    hud.set_state("answering");    time.sleep(0.4)
    hud.set_state("speaking")
    spoken = hud.send_reply("Two things are:" + NL + "- qwen3:8b, 5899 MB" + NL
                            + "- Kokoro, 737 MB", title="VRAM")
    hud.send_transcript("bastion", spoken)
    time.sleep(0.4)
    hud.set_state("idle")


# ─── The waiting queue (pure — the counter, no socket needed) ─────────────────

def queue_tests():
    print("\nthe waiting queue — the counter that says 'heard you, you're next'\n")

    hud.clear_queue()
    check("nothing waiting to start with", hud.queue_depth() == 0)

    hud.queue_enter("What are you playing on Spotify?")
    check("a waiting turn is counted", hud.queue_depth() == 1)
    hud.queue_enter()
    check("a second waiter stacks", hud.queue_depth() == 2)
    hud.queue_leave()
    hud.queue_leave()
    check("leaving brings it back to nothing", hud.queue_depth() == 0)

    # An unbalanced leave is the realistic failure: an exception on the waiting
    # path, or a call site that grew a second early-return. Clamped at zero, so
    # one missed enter() cannot make the band permanently un-clearable — a
    # negative counter would leave "you are next" on the wallpaper for good.
    hud.queue_leave()
    hud.queue_leave()
    check("an unbalanced leave clamps at zero rather than going negative",
          hud.queue_depth() == 0, hud.queue_depth())
    hud.queue_enter("still works")
    check("and the counter still works afterwards", hud.queue_depth() == 1)

    hud.clear_queue()
    check("clear_queue() empties it", hud.queue_depth() == 0)

    # The design decision, asserted so it cannot be quietly undone: queueing is
    # NOT a state value. State is single-valued, so "queued" would replace
    # "thinking" on the stage and hide the turn actually being worked on —
    # exactly the information that was missing at 22:54.
    check("'queued' is deliberately NOT a state value",
          "queued" not in hud.HUD_STATES, hud.HUD_STATES)
    check("the queue API is on hud_server for voice.py to call",
          all(callable(getattr(hud, n, None))
              for n in ("queue_enter", "queue_leave", "clear_queue", "queue_depth")))


def tool_tests():
    """The tool line (pure — the frame and its lifecycle, no socket needed).

    What is on the wallpaper is what the model asked for by name, so the
    security property is the one that matters most here: the frame carries a
    NAME and never the arguments. Tool arguments routinely hold what the owner
    said, file paths, search queries and clipboard text, and this renders on a
    screen other people can see.
    """
    print("\nthe tool line — which tool is running, and what may not go with it\n")

    hud.clear_tool()
    check("nothing on the tool line to start with", hud.current_tool() is None)

    hud.send_tool("look_at_screen")
    msg = hud.current_tool()
    check("a running tool is shown", bool(msg) and msg.get("name") == "look_at_screen", msg)
    check("the frame is exactly type/name/active/since",
          set(msg or {}) == {"type", "name", "active", "since"}, sorted(msg or {}))
    check("...typed 'tool' and marked active",
          msg.get("type") == "tool" and msg.get("active") is True, msg)
    check("'since' is an epoch second, same convention as state.since",
          isinstance(msg.get("since"), (int, float)) and 0 <= time.time() - msg["since"] < 120,
          msg.get("since"))

    # THE security property, asserted two ways: nothing arg-shaped in the frame,
    # and no way to pass one in. A caller cannot leak what it cannot supply.
    check("NO arguments field is on the frame, ever",
          not any(k in (msg or {}) for k in ("arguments", "args", "params", "input")),
          sorted(msg or {}))
    import inspect
    params = list(inspect.signature(hud.send_tool).parameters)
    check("send_tool takes a name and a flag and nothing else",
          params == ["name", "active"],
          f"{params} — adding an args parameter must be a visible signature change")

    # The model writes this string, so it is untrusted. Both wallpapers escape
    # it; this is the belt underneath that.
    hostile = '<img src=x onerror=alert(1)> read_file C:\\Users\\someone\\secrets.txt'
    hud.send_tool(hostile)
    name = hud.current_tool()["name"]
    check("a hostile name is stripped of everything but identifier characters",
          not any(c in name for c in "<>/\\'\"()="), repr(name))
    check("...and capped at HUD_TOOL_NAME_MAX",
          len(name) <= hud.HUD_TOOL_NAME_MAX, f"{len(name)} > {hud.HUD_TOOL_NAME_MAX}")

    hud.send_tool("x" * (hud.HUD_TOOL_NAME_MAX + 40))
    check("a very long name is truncated, not rejected",
          len(hud.current_tool()["name"]) == hud.HUD_TOOL_NAME_MAX,
          len(hud.current_tool()["name"]))

    hud.send_tool("<<<>>>")
    check("a name that is nothing but junk takes the line down instead",
          hud.current_tool() is None, hud.current_tool())

    # The dimmed line has to read "that took 30 s", so finishing must not
    # restamp the clock. This is the check that fails if someone simplifies
    # send_tool to always use time.time().
    hud.send_tool("web_search")
    started = hud.current_tool()["since"]
    time.sleep(0.05)
    hud.send_tool("web_search", False)
    done = hud.current_tool()
    check("finishing a tool KEEPS its start time, so the HUD shows a duration",
          done["since"] == started, f"{done['since']} vs {started}")
    check("...and marks it inactive", done["active"] is False, done)

    # ...but only for the SAME tool. Inheriting another tool's start time would
    # invent a duration that never happened.
    hud.send_tool("open_app", False)
    check("a different tool does not inherit the previous one's start time",
          hud.current_tool()["since"] != started, hud.current_tool())

    # State transitions. voice.py re-enters "thinking" after every tool round,
    # so clearing on entry there would wipe the line the moment the next round
    # began — and clearing on nothing at all would leave a finished tool sitting
    # on the stage claiming to be current.
    hud.send_tool("tv_control")
    hud.set_state("thinking")
    hud.send_tool("tv_control")
    hud.set_state("thinking")
    check("re-entering 'thinking' does NOT clear the tool line",
          hud.current_tool() is not None, hud.current_tool())

    for state in ("answering", "speaking", "idle", "listening"):
        hud.set_state("thinking")
        hud.send_tool("get_weather")
        hud.set_state(state)
        check(f"entering '{state}' takes the tool line down",
              hud.current_tool() is None, hud.current_tool())

    hud.send_tool("get_time")
    hud.clear_tool()
    check("clear_tool() empties it", hud.current_tool() is None)
    hud.clear_tool()
    check("...and clearing twice is a harmless no-op", hud.current_tool() is None)

    check("the tool API is on hud_server for bastion.py to call",
          all(callable(getattr(hud, n, None))
              for n in ("send_tool", "clear_tool", "current_tool")))

    # Leave the module the way the socket half expects to find it.
    hud.set_state("idle")
    hud.clear_tool()


def _port_free(host="127.0.0.1", port=8765):
    """
    True if we can actually bind the HUD port ourselves.

    This matters more than it looks. If a real Bastion is running it already owns
    8765, our own server silently fails to bind, and the test client then
    connects to *that* server and reports a pile of failures about messages it
    never sent — which is worse than not running at all, because the failures
    look real. Checked up front so the skip is explicit.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


if __name__ == "__main__":
    wallpaper_tests()
    split_tests()
    queue_tests()
    if not _port_free():
        print("\nSOCKET TESTS SKIPPED — 127.0.0.1:8765 is already bound, which "
              "means a real Bastion is running.\n  The pure checks above still ran. "
              "Re-run this with Bastion stopped to exercise the wire.\n  (Do not kill "
              "it to make the test pass; it is the assistant.)")
        print("\n" + ("PASS — HUD contract intact (socket half skipped)." if not _fails
                      else f"FAIL — {len(_fails)}: {_fails}"))
        sys.exit(0 if not _fails else 1)
    hud.start()
    threading.Thread(target=drive_states, daemon=True).start()
    asyncio.run(client())
    print("\n" + ("PASS — HUD contract intact." if not _fails
                  else f"FAIL — {len(_fails)}: {_fails}"))
    sys.exit(0 if not _fails else 1)
