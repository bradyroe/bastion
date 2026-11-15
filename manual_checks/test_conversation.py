"""Conversational core — unprompted speech, barge-in, follow-up, sound cues.

Covers the four things added on 2026-08-17, and specifically the parts that fail
SILENTLY in a real turn: an announcement that lands mid-sentence, a barge-in that
sets the trigger but never stops the audio, a follow-up window that never closes,
and a cue that clicks or plays when it was switched off.

No model, no Ollama, no microphone. The one thing that does touch real hardware
is the barge-in latency measurement, which writes a near-silent (1e-4) buffer to
the real output device — because the whole question there is whether PortAudio
actually stops, and a fake sink cannot answer it. Skipped automatically if there
is no output device.

Usage: python test_conversation.py
"""
import re
import sys
import threading
import time

import numpy as np

import voice

# This harness drives failure and cancellation paths that call alerts.send_alert,
# which POSTs to the owner's phone — three real alerts reached it at midnight on
# 2026-08-16 from exactly this kind of harness. Stub it here, in the test, and
# record the calls. Production alerting in voice.py is untouched.
SENT_ALERTS = []
_real_send_alert = voice.alerts.send_alert
voice.alerts.send_alert = lambda *a, **k: SENT_ALERTS.append((a, k))

# Nothing here may reach the HUD socket either.
voice.hud.set_state = lambda *a, **k: None
voice.hud.send_transcript = lambda *a, **k: None
voice.hud.send_thinking = lambda *a, **k: None

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class cfg:
    """Temporarily override config values the way config.py would set them."""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(voice.config, k, KeyError)
            setattr(voice.config, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is KeyError:
                delattr(voice.config, k)
            else:
                setattr(voice.config, k, v)


def reset_state():
    """Put the module back to 'idle, nothing queued' between tests."""
    voice._trigger.clear()
    voice._barged.clear()
    voice._speaking.clear()
    voice._mic_busy.clear()
    voice._mic_free.set()
    voice._unprompted_muted.clear()
    voice._stop_listening.clear()
    while True:
        try:
            voice._unprompted_q.get_nowait()
        except Exception:
            break
    if voice._busy.locked():
        try:
            voice._busy.release()
        except RuntimeError:
            pass


# ── 1. sound cues ────────────────────────────────────────────────────────────

print("\n─ sound cues ─")

wake = voice._cue_audio("wake")
listen = voice._cue_audio("listen")

check("wake cue is short", 0.05 < len(wake) / voice._CUE_RATE < 0.5,
      f"{len(wake)/voice._CUE_RATE:.3f}s")
check("listen cue is short", 0.05 < len(listen) / voice._CUE_RATE < 0.5,
      f"{len(listen)/voice._CUE_RATE:.3f}s")
check("cues are distinguishable", len(wake) != len(listen)
      and not np.array_equal(wake[:len(listen)], listen))
check("no click at the attack", abs(float(wake[0])) < 0.02 and abs(float(listen[0])) < 0.02,
      f"{wake[0]:.4f} / {listen[0]:.4f}")
check("decays to silence", abs(float(wake[-1])) < 0.05 and abs(float(listen[-1])) < 0.05,
      f"{wake[-1]:.4f} / {listen[-1]:.4f}")
check("stays inside full scale", float(np.abs(wake).max()) <= 1.0
      and float(np.abs(listen).max()) <= 1.0)
check("cue audio is float32", wake.dtype == np.float32 and listen.dtype == np.float32)
check("cues are cached, not resynthesised", voice._cue_audio("wake") is wake)

played = []
_real_play = voice.sd.play
voice.sd.play = lambda a, **k: played.append((np.asarray(a), k))

with cfg(SOUND_CUES_ENABLED=True, SOUND_CUE_VOLUME=0.18):
    voice.play_cue("wake")
check("enabled cue plays", len(played) == 1)
check("cue is scaled by SOUND_CUE_VOLUME",
      played and abs(float(np.abs(played[0][0]).max()) - float(np.abs(wake).max()) * 0.18) < 1e-6)
check("cue blocks so it cannot be recorded by the mic it precedes",
      played and played[0][1].get("blocking") is True)

played.clear()
with cfg(SOUND_CUES_ENABLED=False):
    voice.play_cue("wake")
    voice.play_cue("listen")
check("SOUND_CUES_ENABLED=False plays nothing", not played)

played.clear()
with cfg(SOUND_CUES_ENABLED=True, SOUND_CUE_VOLUME=0.0):
    voice.play_cue("wake")
check("volume 0 plays nothing", not played)

played.clear()
voice.sd.play = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device"))
with cfg(SOUND_CUES_ENABLED=True, SOUND_CUE_VOLUME=0.18):
    try:
        voice.play_cue("wake")
        ok = True
    except Exception:
        ok = False
check("a dead output device does not break the turn", ok)
voice.sd.play = lambda a, **k: played.append((np.asarray(a), k))


# ── 2. unprompted speech ─────────────────────────────────────────────────────

print("\n─ unprompted speech ─")

SPOKEN = []
voice._unprompted_sink = lambda text, source: SPOKEN.append((text, source))
voice._ensure_unprompted_worker()


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


reset_state()
SPOKEN.clear()
voice.speak_unprompted("Your ten minute timer is up, sir.", source="timer")
check("idle announcement is spoken", wait_for(lambda: len(SPOKEN) == 1),
      f"spoken={SPOKEN}")
check("source is carried through", SPOKEN and SPOKEN[0][1] == "timer")

reset_state()
SPOKEN.clear()
check("empty text is refused", voice.speak_unprompted("   ", "timer") is False)
check("nothing queued for empty text", voice._unprompted_q.qsize() == 0)

# The load-bearing one: it must WAIT rather than talk over an in-flight turn.
reset_state()
SPOKEN.clear()
voice._busy.acquire()
voice.speak_unprompted("The kettle has boiled.", source="timer")
time.sleep(0.8)
check("does not speak while a turn holds _busy", not SPOKEN, f"spoken={SPOKEN}")
voice._busy.release()
check("speaks as soon as the turn ends", wait_for(lambda: len(SPOKEN) == 1),
      f"spoken={SPOKEN}")
check("_busy is released again afterwards", wait_for(lambda: not voice._busy.locked()))

# _trigger set means the owner is about to talk; his turn wins.
reset_state()
SPOKEN.clear()
voice._trigger.set()
voice.speak_unprompted("Reminder: stand up.", source="reminder")
time.sleep(0.8)
check("does not jump in front of a pending turn", not SPOKEN, f"spoken={SPOKEN}")
voice._trigger.clear()
check("speaks once the pending turn clears", wait_for(lambda: len(SPOKEN) == 1))

reset_state()
SPOKEN.clear()
voice.set_unprompted_muted(True)
check("muted() reports true", voice.unprompted_muted() is True)
check("muted refuses new items", voice.speak_unprompted("Anything.", "timer") is False)
time.sleep(0.4)
check("muted speaks nothing", not SPOKEN)
voice.set_unprompted_muted(False)
check("unmuting does not replay what was suppressed", not SPOKEN)

reset_state()
SPOKEN.clear()
voice._busy.acquire()
voice.speak_unprompted("Queued then muted.", "timer")
time.sleep(0.4)
voice.set_unprompted_muted(True)
voice._busy.release()
time.sleep(0.6)
check("muting mid-wait drops the waiting item", not SPOKEN, f"spoken={SPOKEN}")
voice.set_unprompted_muted(False)

reset_state()
SPOKEN.clear()
with cfg(UNPROMPTED_MAX_WAIT_S=0.5):
    voice._busy.acquire()
    voice.speak_unprompted("Stale timer.", "timer")
    time.sleep(1.2)
    voice._busy.release()
    time.sleep(0.6)
check("an item that waited past UNPROMPTED_MAX_WAIT_S is dropped, not spoken late",
      not SPOKEN, f"spoken={SPOKEN}")

reset_state()
SPOKEN.clear()
with cfg(UNPROMPTED_SPEECH_ENABLED=False):
    check("UNPROMPTED_SPEECH_ENABLED=False refuses",
          voice.speak_unprompted("No.", "timer") is False)
time.sleep(0.3)
check("disabled speaks nothing", not SPOKEN)

reset_state()
SPOKEN.clear()
voice._busy.acquire()
with cfg(UNPROMPTED_QUEUE_MAX=3):
    for i in range(6):
        voice.speak_unprompted(f"item {i}", "timer")
    check("queue is capped", voice._unprompted_q.qsize() <= 3,
          f"qsize={voice._unprompted_q.qsize()}")
voice._busy.release()
check("the NEWEST items survive the cap", wait_for(
    lambda: len(SPOKEN) >= 1 and SPOKEN[-1][0] == "item 5"), f"spoken={SPOKEN}")

reset_state()
SPOKEN.clear()
errs = []


def from_another_thread():
    try:
        for i in range(5):
            voice.speak_unprompted(f"thread {i}", "timer")
    except Exception as exc:
        errs.append(exc)


ts = [threading.Thread(target=from_another_thread) for _ in range(3)]
[t.start() for t in ts]
[t.join() for t in ts]
check("safe to call from any thread", not errs, str(errs))
check("all 15 are delivered", wait_for(lambda: len(SPOKEN) == 15, 10.0), f"n={len(SPOKEN)}")

reset_state()
SPOKEN.clear()
check("no alert fired from the unprompted path", not SENT_ALERTS, str(SENT_ALERTS))


# ── 3. barge-in ──────────────────────────────────────────────────────────────

print("\n─ barge-in ─")


class FakeOww:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


reset_state()
check("threshold is the normal one when idle",
      voice._wake_threshold() == voice.config.WAKEWORD_THRESHOLD)
voice._mark_speaking_started()
with cfg(BARGE_IN_THRESHOLD=0.6):
    check("threshold is raised while speaking, to resist self-hearing",
          voice._wake_threshold() == 0.6)
    check("raised threshold is above every measured self-reference (0.31)",
          voice._wake_threshold() > 0.31)
voice._mark_speaking_ended()

reset_state()
oww = FakeOww()
voice._mark_speaking_started()
with cfg(BARGE_IN_ENABLED=True, BARGE_IN_GUARD_S=5.0):
    took = voice._handle_wake(oww, 0.9)
check("a detection inside the guard window is ignored", took is False)
check("guard window leaves the trigger alone", not voice._trigger.is_set())
check("guard window leaves playback alone", voice._speaking.is_set())

reset_state()
oww = FakeOww()
voice._mark_speaking_started()
voice._speaking_since -= 3.0          # pretend we are 3s into the reply
stopped = []
_real_stop = voice.stop_speaking
voice.stop_speaking = lambda: stopped.append(time.perf_counter())
with cfg(BARGE_IN_ENABLED=True, BARGE_IN_GUARD_S=0.7, SOUND_CUES_ENABLED=False):
    took = voice._handle_wake(oww, 0.9)
check("a detection past the guard barges in", took is True)
check("barge-in actually calls stop_speaking", len(stopped) == 1)
check("barge-in sets the trigger so the mic reopens", voice._trigger.is_set())
check("barge-in marks the turn as interrupted", voice._barged.is_set())
check("barge-in clears the speaking flag", not voice._speaking.is_set())
check("barge-in resets the wake model so its own tail cannot refire",
      oww.resets >= 1)

reset_state()
oww = FakeOww()
stopped.clear()
voice._mark_speaking_started()
voice._speaking_since -= 3.0
with cfg(BARGE_IN_ENABLED=False):
    took = voice._handle_wake(oww, 0.99)
check("BARGE_IN_ENABLED=False is a real off switch", took is False and not stopped)
check("disabled barge-in leaves the reply playing", voice._speaking.is_set())

reset_state()
oww = FakeOww()
stopped.clear()
voice._busy.acquire()                  # a turn is running, nothing audible: thinking
with cfg(BARGE_IN_ENABLED=True):
    took = voice._handle_wake(oww, 0.99)
check("wake while thinking is ignored, not a half-cancel", took is False and not stopped)
check("thinking wake does not set the trigger", not voice._trigger.is_set())
voice._busy.release()

reset_state()
oww = FakeOww()
with cfg(SOUND_CUES_ENABLED=False):
    took = voice._handle_wake(oww, 0.9)
check("an ordinary idle wake still starts a turn", took is True and voice._trigger.is_set())

# The hotkey has no guard window on purpose — a key press is never a false
# positive, which is what makes it the reliable way to shut Bastion up.
reset_state()
stopped.clear()
voice._mark_speaking_started()
with cfg(BARGE_IN_ENABLED=True, SOUND_CUES_ENABLED=False):
    voice._on_hotkey()
check("hotkey barges in with no guard window", len(stopped) == 1)
check("hotkey barge-in sets the trigger", voice._trigger.is_set())

reset_state()
stopped.clear()
with cfg(SOUND_CUES_ENABLED=False):
    voice._on_hotkey()
check("idle hotkey starts a turn without stopping anything",
      voice._trigger.is_set() and not stopped)

reset_state()
stopped.clear()
voice._busy.acquire()
with cfg(SOUND_CUES_ENABLED=False):
    voice._on_hotkey()
check("hotkey mid-turn with no audio does not re-enter", not voice._trigger.is_set())
voice._busy.release()
voice.stop_speaking = _real_stop

# stop_speaking must reach the live stream, not just sd.stop().
reset_state()


class _Cancellable:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


fake = _Cancellable()
with voice._speaking_lock:
    voice._active_stream = fake
voice.stop_speaking()
check("stop_speaking cancels the active stream", fake.stopped)
with voice._speaking_lock:
    voice._active_stream = None


# ── 4. barge-in latency, through the real output device ──────────────────────

print("\n─ barge-in latency (real PortAudio) ─")


class QuietTTS(voice._KokoroTTS):
    """Six seconds of inaudible audio, so a real device can be measured without
    anyone in the room hearing the test.

    Subclasses _KokoroTTS deliberately: _synth_chunks() dispatches on isinstance,
    so a plain stand-in takes the Piper branch and dies on synthesize_wav."""
    sample_rate = 24000

    def __init__(self):
        pass                                   # no model, no device

    def generate(self, text):
        yield (np.zeros(self.sample_rate * 6, dtype="float32") + 1e-4)

    def synth(self, text):
        return next(self.generate(text))


try:
    voice.sd.play = _real_play
    st = voice.SpeechStream(QuietTTS())
    with voice._speaking_lock:
        voice._active_stream = st
    st.feed("This is a long sentence that will be playing for six seconds. ")
    t_ready = time.perf_counter()
    while st.first_audio_at is None and time.perf_counter() - t_ready < 10:
        time.sleep(0.005)
    if st.first_audio_at is None:
        print("  skip  no audio ever started (no output device?)")
    else:
        time.sleep(0.4)
        t0 = time.perf_counter()
        voice.stop_speaking()
        st.finish(timeout=10)
        ms = (time.perf_counter() - t0) * 1000.0
        check(f"audio stops promptly after a barge-in ({ms:.0f} ms of a 6.0 s buffer)",
              ms < 700, f"{ms:.0f} ms")
        check("barge-in returns nothing to speak the slow way", st._leftover() == "")
except Exception as exc:
    print(f"  skip  real-device latency check unavailable: {exc}")
finally:
    with voice._speaking_lock:
        voice._active_stream = None
    voice.sd.play = lambda a, **k: played.append((np.asarray(a), k))


# ── 5. the follow-up window ──────────────────────────────────────────────────

print("\n─ follow-up window ─")

REC_CALLS = []
_real_rec = voice.sd.rec


def fake_rec(frames, **kw):
    REC_CALLS.append(kw)
    return np.zeros((frames, 1), dtype="float32")      # pure silence


voice.sd.rec = fake_rec

reset_state()
REC_CALLS.clear()
with cfg(LISTEN_WINDOW_S=4.0):
    audio, found = voice.record_until_silence(None)
check("a summoned turn waits ~4s for speech to start", len(REC_CALLS) == 9,
      f"{len(REC_CALLS)} x 0.5s chunks")
check("silence is reported as no speech", found is False)

reset_state()
REC_CALLS.clear()
voice.record_until_silence(None, open_window_s=2.5)
check("a follow-up turn gives up after ~2.5s", len(REC_CALLS) == 6,
      f"{len(REC_CALLS)} x 0.5s chunks")

reset_state()
REC_CALLS.clear()
voice.record_until_silence(None, open_window_s=0.0)
check("a zero window still records at least one chunk", len(REC_CALLS) >= 1)

reset_state()
REC_CALLS.clear()
voice._mic_free.clear()               # pretend the listener holds the device
held = []


def release_soon():
    time.sleep(0.2)
    held.append(voice._mic_busy.is_set())
    voice._mic_free.set()


threading.Thread(target=release_soon).start()
t0 = time.perf_counter()
voice.record_until_silence(None, open_window_s=0.5)
waited = time.perf_counter() - t0
check("the recorder waits for the wake-word listener to release the mic",
      held == [True], f"held={held}")
check("the recorder signals _mic_busy while it owns the device", waited >= 0.2)
check("the mic is handed back afterwards", not voice._mic_busy.is_set())

reset_state()
REC_CALLS.clear()
voice._mic_free.clear()
t0 = time.perf_counter()
voice._acquire_mic(timeout=0.3)
check("a listener that never releases does not deadlock the turn",
      0.3 <= time.perf_counter() - t0 < 1.5)
voice._release_mic()
voice._mic_free.set()

voice.sd.rec = _real_rec


# ── 5b. the streaming voice/screen split ─────────────────────────────────────
#
# The reference is hud_server.split_reply() on the finished string. _VoiceSplit
# has to reach the same decision while the reply is still arriving, one token at
# a time, which is a strictly harder problem — so every case here is checked
# against split_reply() rather than against what _VoiceSplit happens to do.

print("\n─ voice/screen split (streaming) ─")


def stream_split(text, feed_size=7, budget=None):
    """Push text through _VoiceSplit like a token stream; return what was spoken."""
    spoken = []
    vs = voice._VoiceSplit(lambda d: spoken.append(d) if d is not None else spoken.clear(),
                           budget=budget)
    for i in range(0, len(text), feed_size):
        vs.feed(text[i:i + feed_size])
    vs.close()
    return "".join(spoken).strip(), vs


def norm(t):
    return " ".join(t.split())


PROSE = "The disk is ninety-eight percent full. That is the thing worth fixing."
LIST = ("Three things are worth doing.\n"
        "- Clear the pip cache.\n"
        "- Move the model files.\n"
        "- Uninstall the old CUDA toolkit.")
BULLETS_ONLY = "- Clear the pip cache.\n- Move the model files.\n- Uninstall CUDA."
NUMBERED = "Do it in this order.\n1. Stop Ollama.\n2. Clear the cache.\n3. Restart."
LONG = ("Rust offers performance and memory safety without garbage collection, which "
        "makes it a good fit for systems programming. Python excels in rapid "
        "development and ecosystem breadth for automation and data analysis. If your "
        "scripting is lightweight, deepen your Python skills instead. If you need "
        "speed and low-level control, Rust is worth the learning curve. Prioritise "
        "based on what your projects actually demand of you today.")

for name, text, feed in [("short prose", PROSE, 7),
                         ("short prose, 1-char deltas", PROSE, 1),
                         ("short prose, whole string", PROSE, 10_000)]:
    spoken, vs = stream_split(text, feed)
    ref, detail = voice.hud.split_reply(text)
    check(f"{name}: spoken whole", norm(spoken) == norm(ref), f"{spoken!r}")
    check(f"{name}: nothing on screen", vs.truncated is False and detail == "")

# Rewritten 2026-08-18. These three used to assert "the bullets are NEVER
# spoken" and "reason == 'list'" — the behaviour, not the rule, which is how the
# lead-in half of the bug survived the bullets-only fix. LIST is three short
# items with a lead-in, so split_reply() now flattens and speaks the whole
# thing; the suppression assertions moved to a list that really is suppressed.
spoken, vs = stream_split(LIST)
ref, detail = voice.hud.split_reply(LIST)
check("list with a lead-in: same decision as split_reply",
      norm(spoken) == norm(ref), f"{spoken!r} vs {ref!r}")
check("short list with a lead-in: the items are actually said, not screened",
      "pip cache" in spoken and "CUDA" in spoken, f"{spoken!r}")
check("list: goes on screen", vs.truncated is True and detail == LIST)
check("list: reason is recorded", vs.reason == "list-spoken", vs.reason)

# The same shape, past the item cap: now the bullets ARE suppressed, the lead-in
# is spoken as the whole line, and the handoff is promised.
LONG_LIST = "Three things are worth doing.\n" + "\n".join(
    f"- {'word ' * 12}{i}" for i in range(6))
spoken, vs = stream_split(LONG_LIST)
ref, _ = voice.hud.split_reply(LONG_LIST)
check("list too long to say: same decision as split_reply",
      norm(spoken) == norm(ref), f"{spoken!r} vs {ref!r}")
check("list too long to say: no item is spoken, and the screen is promised",
      "word word" not in spoken and spoken.endswith(voice.hud.HUD_HANDOFF_PHRASE),
      f"{spoken!r}")
check("list too long to say: reason is recorded", vs.reason == "list", vs.reason)

spoken, vs = stream_split(NUMBERED)
ref, _ = voice.hud.split_reply(NUMBERED)
check("numbered list: same decision", norm(spoken) == norm(ref), f"{spoken!r} vs {ref!r}")

# A SHORT bullets-only reply is spoken in full, flattened — it is not handed to a
# monitor the owner may not be near. This assertion used to require the opposite,
# which is how the bug survived: the test encoded the behaviour, not the rule.
spoken, vs = stream_split(BULLETS_ONLY)
ref, detail = voice.hud.split_reply(BULLETS_ONLY)
check("bullets only, short: spoken in full, flattened", norm(spoken) == norm(ref),
      f"{spoken!r} vs {ref!r}")
check("bullets only, short: every item is actually said",
      "pip cache" in spoken and "CUDA" in spoken, f"{spoken!r}")
check("bullets only, short: still shows the panel", vs.truncated is True)
check("bullets only, short: promises no screen it did not need",
      voice.hud.HUD_HANDOFF_PHRASE not in spoken and "on the screen" not in spoken.lower(),
      f"{spoken!r}")

TINY = "- milk\n- eggs"
for fs in (1, 2, 7, 10_000):
    spoken, vs = stream_split(TINY, fs)
    ref, _ = voice.hud.split_reply(TINY)
    check(f"thirteen-char list is spoken, not screened (delta {fs})",
          norm(spoken) == norm(ref) and "milk" in spoken, f"{spoken!r} vs {ref!r}")

# Too many items: length alone would pass (well under the budget), the item cap
# is what catches it. The same list one item shorter is spoken.
MANY = "\n".join(f"- item {i}" for i in range(voice.hud.HUD_SPOKEN_MAX_ITEMS + 1))
FEW = "\n".join(f"- item {i}" for i in range(voice.hud.HUD_SPOKEN_MAX_ITEMS))
for name, text in (("over the item cap", MANY), ("at the item cap", FEW)):
    spoken, vs = stream_split(text)
    ref, _ = voice.hud.split_reply(text)
    check(f"{name}: same decision as split_reply", norm(spoken) == norm(ref),
          f"{spoken!r} vs {ref!r}")
check("over the item cap hands off rather than reciting",
      norm(stream_split(MANY)[0]) == voice.hud.HUD_SCREEN_ONLY_LINE)

# Long items, few of them: the char budget is what catches this one.
WORDY = "\n".join(f"- {'word ' * 20}{i}" for i in range(3))
spoken, vs = stream_split(WORDY)
ref, _ = voice.hud.split_reply(WORDY)
check("bullets only, too long to say: same decision", norm(spoken) == norm(ref),
      f"{spoken!r}")

# A reply that turns out longer than it first looked: the first items qualify,
# the seventh disqualifies it. Nothing may have been spoken by then.
GROWS = "\n".join(f"- item {i}" for i in range(12))
for fs in (1, 4, 10_000):
    spoken, vs = stream_split(GROWS, fs)
    check(f"a list that outgrows the cap mid-stream still hands off (delta {fs})",
          norm(spoken) == voice.hud.HUD_SCREEN_ONLY_LINE, f"{spoken!r}")

# Rule C: a colon must never be left dangling in speech — "you need:" and then
# silence. There are two honest endings and split_reply() picks between them by
# whether the list is worth saying: KEEP the colon when the items follow it
# ("You need: milk, eggs."), CLOSE it to a full stop when they do not
# ("You need. The rest is on the screen."). This asserted the second one
# unconditionally until 2026-08-18, which is the bug the owner hears: eleven
# characters of shopping list withheld and a monitor to walk to.
COLON = "You need:\n- milk\n- eggs"
for fs in (1, 3, 7, 10_000):
    spoken, vs = stream_split(COLON, fs)
    ref, _ = voice.hud.split_reply(COLON)
    check(f"a short list after a colon is spoken, not screened (delta {fs})",
          norm(spoken) == norm(ref) and "milk" in spoken and "eggs" in spoken,
          f"{spoken!r} vs {ref!r}")
check("a colon inside a sentence is left alone",
      "9:15" in stream_split("The meeting is at 9:15 in the morning.")[0])

# The other half of rule C: the same lead-in, a list too big to say. The colon
# is CLOSED here, because nothing follows it in speech.
COLON_LONG = "You need:\n" + "\n".join(f"- item {i}" for i in range(12))
for fs in (1, 3, 7, 10_000):
    spoken, vs = stream_split(COLON_LONG, fs)
    ref, _ = voice.hud.split_reply(COLON_LONG)
    check(f"a colon before a long list is completed, not left hanging (delta {fs})",
          norm(spoken) == norm(ref)
          and norm(spoken).lower().endswith("the rest is on the screen.")
          and "item 3" not in spoken, f"{spoken!r} vs {ref!r}")
check("the handoff after a colon is not then said a second time",
      stream_split(COLON_LONG)[0].lower().count("on the screen") == 1,
      f"{stream_split(COLON_LONG)[0]!r}")
check("no colon is ever left at the end of a spoken line",
      not any(stream_split(t, fs)[0].rstrip().endswith(":")
              for t in (COLON, COLON_LONG, LIST, LONG_LIST) for fs in (1, 5, 10_000)))

# The handoff was suppressed by the bare word "screen", so the vision path
# published a panel and never said so. Match the phrase, not the word.
VISION = ("Your screen shows a terminal on the left and a browser on the right. "
          "The terminal is running a long build. The browser is on a documentation "
          "page about audio devices. There is a chat window behind both of them. "
          "Nothing in view looks like it needs attention right now, sir. The build "
          "has been going for some minutes and is still compiling dependencies. "
          "I will keep an eye on it and tell you when it finishes, sir.")
spoken, vs = stream_split(VISION)
check("a reply that merely mentions a screen still gets the handoff",
      vs.truncated is True and spoken.endswith(voice.hud.HUD_HANDOFF_PHRASE),
      f"…{spoken[-70:]!r}")
check("the handoff is not doubled when the reply already says 'on the screen'",
      stream_split("It is on the screen. " + LONG)[0].count("on the screen") == 1)

spoken, vs = stream_split(LONG)
ref, detail = voice.hud.split_reply(LONG)
check("long reply is truncated", vs.truncated is True and detail == LONG)
check("long reply ends with the handoff line",
      spoken.endswith(voice.hud.HUD_HANDOFF_PHRASE), f"…{spoken[-60:]!r}")
check("long reply is cut at a sentence end, not mid-word",
      norm(spoken).replace(" " + voice.hud.HUD_HANDOFF_PHRASE, "").rstrip()[-1] in ".!?",
      f"…{spoken[-90:]!r}")
check("what is spoken is a real prefix of the reply",
      norm(LONG).startswith(norm(spoken).replace(" " + voice.hud.HUD_HANDOFF_PHRASE, "").strip()))
# The documented divergence: split_reply() stops before the sentence that would
# overrun, _VoiceSplit finishes it. Bounded by one sentence, and long rather than
# short on purpose. Measured 69 chars on this reply.
over = len(spoken) - len(ref)
check("overshoots split_reply by under one sentence, never falls short",
      0 <= over < 120, f"{over} chars")

for fs in (1, 3, 7, 40, 10_000):
    a, _ = stream_split(LIST, fs)
    b, _ = stream_split(LONG, fs)
    check(f"delta size {fs} does not change the decision",
          norm(a) == norm(stream_split(LIST, 7)[0]) and norm(b) == norm(stream_split(LONG, 7)[0]))

# ── the equivalence sweep ────────────────────────────────────────────────────
#
# The cases above pin the shapes that were actually broken. This pins the RULE:
# for every shape, at every delta size, _VoiceSplit says what split_reply()
# says. The one licensed difference is the documented latency trade — on a
# budget cut the stream finishes the sentence that split_reply() stops before —
# so that case is checked as "a prefix plus at most one more sentence", and
# everything else is checked as exact equality.
#
# feed_size=1 earns its place here rather than being tidied away: both bugs
# `voice-core` found (a whitespace-only delta swallowed, and the budget enforced
# only between deltas) were invisible at every other delta size.

TRAILING_PROSE = "Do these.\n- One.\n- Two.\nThat is all."
# The shape the lead asked about: it looks sayable right up until it isn't. Two
# short bullets qualify, and then a paragraph arrives that puts it over.
GROWS_LATE = ("You need:\n- milk\n- eggs\n"
              + "I would do the first one tonight, sir, because the disk is the "
                "thing actually holding you up and the rest can wait for the "
                "morning without costing you anything at all.")
LEAD_TWO_SENT = ("I checked the disk. Three things are worth doing.\n"
                 "- Clear the pip cache.\n- Move the models.")
SCREEN_LEAD = "I put it on the screen.\n- a\n- b"
SPACED = "The disk is ninety-eight percent full. That is worth fixing."
CORPUS = [("short prose", PROSE), ("bullets only, short", BULLETS_ONLY),
          ("bullets only, wordy", WORDY), ("bullets only, many", MANY),
          ("bullets only, at the cap", FEW), ("outgrows the cap", GROWS),
          ("lead-in, short list", LIST), ("lead-in, long list", LONG_LIST),
          ("numbered", NUMBERED), ("colon, short list", COLON),
          ("colon, long list", COLON_LONG), ("trailing prose line", TRAILING_PROSE),
          ("qualifies then outgrows it", GROWS_LATE),
          ("two-sentence lead-in", LEAD_TWO_SENT), ("already says screen", SCREEN_LEAD),
          ("long prose", LONG), ("vision reply", VISION), ("empty", ""),
          ("whitespace only", "   \n  \n"), ("one word", "Done, sir.")]

for label, text in CORPUS:
    ref, detail = voice.hud.split_reply(text)
    for fs in (1, 2, 3, 7, 40, 10_000):
        spoken, vs = stream_split(text, fs)
        if vs.reason == "budget":
            # The licensed divergence, and it is one-directional: never shorter
            # than split_reply(), never longer by more than the sentence that
            # crossed the budget.
            bare = norm(spoken).replace(" " + voice.hud.HUD_HANDOFF_PHRASE, "").strip()
            ok = (norm(text).startswith(bare)
                  and len(bare) >= len(norm(ref).replace(
                      " " + voice.hud.HUD_HANDOFF_PHRASE, "").strip())
                  and len(spoken) - len(ref) < 200)
            check(f"{label} @{fs}: overshoots by at most one sentence, never falls short",
                  ok, f"{spoken!r} vs {ref!r}")
        else:
            check(f"{label} @{fs}: identical decision to split_reply",
                  norm(spoken) == norm(ref), f"{spoken!r} vs {ref!r}")
        check(f"{label} @{fs}: panel published iff something is worth reading",
              vs.truncated == bool(detail), f"truncated={vs.truncated} detail={bool(detail)}")

# Ollama emits lone " " tokens, so a delta can be nothing but whitespace. Losing
# one gives "Thediskisninety-eightpercentfull." — caught only at feed_size=1.
sp = []
vs = voice._VoiceSplit(lambda d: sp.append(d) if d is not None else sp.clear())
for tok in re.findall(r"\S+|\s+", SPACED):
    vs.feed(tok)
vs.close()
check("whitespace-only deltas survive word-by-word feeding",
      norm("".join(sp)) == norm(SPACED), f"{''.join(sp)!r}")

# A retracted tool-call preamble must not spend the budget of the real reply.
spoken = []
vs = voice._VoiceSplit(lambda d: spoken.append(d) if d is not None else spoken.clear())
for i in range(0, 200, 7):
    vs.feed(LONG[i:i + 7])
vs.feed(None)                       # tool-call round: retract everything
for i in range(0, len(PROSE), 7):
    vs.feed(PROSE[i:i + 7])
vs.close()
check("a retracted preamble resets the budget",
      norm("".join(spoken)) == norm(PROSE), f"{''.join(spoken)!r}")
check("a retracted preamble leaves nothing on screen", vs.truncated is False)

spoken, vs = stream_split("")
check("empty reply speaks nothing and shows nothing", spoken == "" and not vs.truncated)

spoken, vs = stream_split(PROSE, budget=10)
check("a tiny budget still speaks one whole sentence",
      spoken.split(".")[0] == PROSE.split(".")[0], f"{spoken!r}")

# The failure this design exists to avoid: suppressed text coming back out of
# SpeechStream.finish() as `leftover` and being read aloud the slow way.
fed = []
st_spoken = []
vs = voice._VoiceSplit(lambda d: (st_spoken.clear() if d is None else fed.append(d)))
for i in range(0, len(LONG_LIST), 7):
    vs.feed(LONG_LIST[i:i + 7])
vs.close()
check("suppressed bullets are never handed to SpeechStream at all",
      "pip cache" not in "".join(fed))


# ── 6. the wake-word listener gates on the RECORDER, not the whole turn ──────

print("\n─ mic gating ─")

src = voice.inspect.getsource(voice._wake_word_loop)
check("listener no longer pauses for the whole turn (_busy.locked)",
      "_busy.locked()" not in src)
check("listener pauses for the recorder instead (_mic_busy)", "_mic_busy" in src)
check("listener publishes _mic_free so the recorder can wait for it",
      "_mic_free.set()" in src and "_mic_free.clear()" in src)

turn_src = voice.inspect.getsource(voice.voice_turn)
check("streaming TTS logging is still in voice_turn",
      "Streaming TTS: ON" in turn_src and "Streaming TTS: OK" in turn_src)
check("a barged reply is logged as BARGE-IN, not OK", "Streaming TTS: BARGE-IN" in turn_src)
check("the listen cue is played before the mic opens",
      turn_src.index('play_cue("listen")') < turn_src.index("record_until_silence"))
check("the HUD gets the transcribing state", 'hud.set_state("transcribing")' in turn_src)
check("the HUD gets the answering state", 'hud.set_state("answering")' in turn_src)
check("a retracted tool-call preamble puts the rail back to thinking",
      'answering.clear()' in turn_src and turn_src.index("answering.clear()")
      < turn_src.index('answering.append'))
check("the split flushes before the stream is closed",
      turn_src.index("split.close()") < turn_src.index("stream.finish()"))
check("the written half is published only when something was left out",
      "split.truncated" in turn_src and "hud.send_detail" in turn_src)


# ── 7. a summons heard MID-TURN is queued, not dropped ───────────────────────
#
# Approved by the owner 2026-08-18. Saying "Bastion" while it was thinking used to do
# nothing whatever, which is indistinguishable from a deaf wake word — the exact
# ambiguity that costs an evening of debugging the microphone.
#
# The accepted cost is that a FALSE trigger mid-turn now becomes one spurious
# turn instead of nothing. "One" is the load-bearing word, so it is tested.

print("\n─ queued summons ─")


class _Oww:
    """openWakeWord stands in for one method here: reset()."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


QCALLS = []
for _fn in ("queue_enter", "queue_leave", "clear_queue"):
    setattr(voice.hud, _fn, lambda *a, _f=_fn, **k: QCALLS.append(_f))


def queue_reset():
    QCALLS.clear()
    voice._queued_wake.clear()
    voice._trigger.clear()
    voice._speaking.clear()
    while voice._busy.locked():
        voice._busy.release()


queue_reset()
voice._busy.acquire()                      # a turn is running
oww = _Oww()
took_mic = voice._handle_wake(oww, 0.9)
check("a wake word mid-turn no longer releases the mic to a turn that is running",
      took_mic is False)
check("...but it IS remembered rather than dropped", voice._queued_wake.is_set())
check("...and the HUD is told something is waiting", QCALLS == ["queue_enter"], QCALLS)
check("...and the running turn is not interrupted", not voice._trigger.is_set())

for _ in range(4):
    voice._handle_wake(_Oww(), 0.9)
check("four more wake words are still ONE pending turn, never a pile-up",
      QCALLS == ["queue_enter"] and voice._queued_wake.is_set(), QCALLS)

voice._busy.release()
check("the queued summons becomes exactly one turn when the turn ends",
      voice._consume_queued_wake() is True and voice._trigger.is_set())
check("...and the waiting band is cleared", QCALLS == ["queue_enter", "queue_leave"], QCALLS)
check("consuming twice does not produce a second turn",
      voice._consume_queued_wake() is False)

# The race the briefed patch did not cover: the turn ends between the
# _busy.locked() check and the flag being set, so the release block has already
# run and nobody is left to consume it. Without the re-check it would fire at the
# end of the NEXT turn — an answer arriving one turn late.
queue_reset()
voice._queue_wake("test")                  # _busy NOT held: nobody to consume it
check("a summons queued just as the turn ends still runs immediately",
      voice._trigger.is_set() and not voice._queued_wake.is_set(),
      f"trigger={voice._trigger.is_set()} queued={voice._queued_wake.is_set()}")
check("...and the HUD band opens and closes rather than sticking",
      QCALLS == ["queue_enter", "queue_leave"], QCALLS)

# The hotkey half. He pressed a key, so it is acknowledged with the cue whether
# or not it can run yet — a silent key press reads as a broken hotkey.
queue_reset()
cues = []
_real_cue = voice.play_cue
voice.play_cue = lambda name, *a, **k: cues.append(name)
voice._busy.acquire()
try:
    voice._on_hotkey()
finally:
    voice.play_cue = _real_cue
    voice._busy.release()
check("the hotkey mid-turn is acknowledged out loud", cues == ["wake"], cues)
check("the hotkey mid-turn is queued, not dropped",
      voice._queued_wake.is_set() and QCALLS == ["queue_enter"], QCALLS)
queue_reset()

# The wake path must never raise, so the queue API is called through a guard: a
# Bastion running against an older hud_server has no queue_enter at all.
_saved = {f: getattr(voice.hud, f) for f in ("queue_enter", "queue_leave", "clear_queue")}
for _fn in _saved:
    delattr(voice.hud, _fn)
try:
    voice._busy.acquire()
    ok = voice._handle_wake(_Oww(), 0.9) is False and voice._queued_wake.is_set()
    voice._busy.release()
    check("an hud_server without the queue API does not break the wake path", ok)
except Exception as exc:
    check("an hud_server without the queue API does not break the wake path",
          False, repr(exc))
finally:
    for _fn, _v in _saved.items():
        setattr(voice.hud, _fn, _v)
    queue_reset()

main_src = voice.inspect.getsource(voice.main)
check("a crash mid-wait leaves no stale 'you are next' on the HUD",
      "clear_queue" in main_src and "_queued_wake.clear()" in main_src)
check("the queued summons is consumed after the turn's lock is released",
      "_consume_queued_wake()" in main_src)
check("a queued summons is a fresh turn, so the follow-up chain is zeroed",
      main_src.index("_consume_queued_wake()") < main_src.index("follow_up = False\n"
                                                               "            if _stop_listening"))
check("unprompted speech also releases a summons queued behind it",
      "_consume_queued_wake()" in voice.inspect.getsource(voice._deliver_unprompted))


# ── done ─────────────────────────────────────────────────────────────────────

print("\n─ alerting ─")
check("harness fired no real alerts", voice.alerts.send_alert is not _real_send_alert)
if SENT_ALERTS:
    print(f"       {len(SENT_ALERTS)} alert(s) captured by the stub: "
          f"{[(a[0], a[1]) for a, _ in SENT_ALERTS]}")

voice._stop_listening.set()
print(f"\n{PASS} passed, {FAIL} failed\n")
sys.exit(1 if FAIL else 0)
