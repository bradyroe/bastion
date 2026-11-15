"""
Bastion — Phase 2: Voice I/O

Activation flow:
  Press Ctrl+Shift+J  →  mic opens  →  speak  →  silence stops recording
  →  Whisper transcribes locally  →  Ollama responds  →  Windows TTS speaks
  →  back to idle (mic closed)

Security properties:
  - Microphone is ONLY active during sd.rec() calls — not always-on listening.
  - All audio is processed by Whisper on this machine — no cloud STT.
  - Transcribed text goes to local Ollama — never leaves the machine.
  - A threading.Lock prevents overlapping activations (no double-trigger race).
  - All Phase 1 tool security (path containment, whitelists, SQL params) is
    inherited via the imported chat() function.
"""

import logging
import sys
import textwrap
import threading

# Force UTF-8 on stdout/stderr so box-drawing chars don't crash on Windows
# (default console encoding cp1252 can't encode them). Guarded because under
# pythonw.exe (windowless autostart) these streams can be None.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import inspect
import io
import json
import os
import queue
import re
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from pynput import keyboard as kb

import alerts
import config
import hud_server as hud
import tools as T
from bastion import SYSTEM_PROMPT, check_ollama, chat, trim_history

log = logging.getLogger("bastion.voice")

# Alerting rule for this file, which is stricter than anywhere else in Bastion:
# voice.py handles the user's actual speech, so an alert from here may carry a
# component name, an exception class and a short fixed description — and nothing
# else. Never a transcript, never model output, never audio, never a voice-model
# path. See the module docstring of alerts.py.

# ─── Shared state ─────────────────────────────────────────────────────────────

_trigger  = threading.Event()   # set by hotkey, cleared after each turn begins
_busy     = threading.Lock()    # held during a full voice turn; blocks re-entry
# A summons heard mid-turn. Until 2026-08-18 it was dropped silently: saying
# "Bastion" while it was thinking did nothing at all, with no way to tell that
# from a deaf wake word. It is now remembered, shown on the HUD as waiting, and
# run when the turn ends (the owner's decision — it is how Bastion behaves toward
# him, not a technical call).
#
# An Event, not a counter, and that is the safety property: five wake words
# during one long answer are one pending turn, not five. the owner accepted that a
# false trigger mid-turn now costs one spurious turn instead of nothing; a
# counter would have made that unbounded.
_queued_wake = threading.Event()
_messages: list[dict] = []      # persistent conversation history

# One alert per unplug, not one per turn — see _resolve_input_device().
_mic_missing_alerted = False

# ── Mic ownership ────────────────────────────────────────────────────────────
#
# The wake-word listener and record_until_silence both want the microphone, and
# PortAudio does not want them to have it at the same time. Until 2026-08-17 the
# hand-off was "the listener pauses whenever _busy is locked" — i.e. for the
# WHOLE turn, thinking and speaking included. That is why barge-in could never
# have worked: while Bastion was talking, nothing was listening.
#
# The gate is now the recorder specifically, not the turn:
#   _mic_busy  set by the recorder while it owns the device
#   _mic_free  set by the listener whenever it is NOT holding an InputStream
# The recorder sets _mic_busy, waits for _mic_free, then records. The listener
# therefore stays live through thinking and speaking, which is exactly the
# window barge-in needs.
_mic_busy = threading.Event()
_mic_free = threading.Event()
_mic_free.set()                 # nobody holds the mic before the listener starts

# Set while Bastion is producing sound that the owner may talk over — the only state
# in which a wake word means "stop" rather than "start".
_speaking = threading.Event()
_speaking_since = 0.0           # perf_counter of the current utterance's start
_barged = threading.Event()     # last utterance was cut short by a barge-in


def _acquire_mic(timeout: float = 2.0) -> None:
    """Take the microphone from the wake-word listener."""
    _mic_busy.set()
    if not _mic_free.wait(timeout):
        # The listener should release within one 80 ms frame. If it has not, we
        # record anyway: a shared-mode WASAPI overlap is survivable, a turn that
        # silently never opens the mic is not.
        log.warning("Mic hand-off timed out after %.1fs — recording anyway.", timeout)


def _release_mic() -> None:
    _mic_busy.clear()

# ─── TTS ──────────────────────────────────────────────────────────────────────

def _resolve_kokoro_device() -> str:
    """
    Pick the device Kokoro synthesises on. `KOKORO_DEVICE` is "auto" (CUDA when
    the box has it, CPU otherwise), "cuda" or "cpu"; read through _cfg so this
    file still works before config.py carries the setting, and so a machine with
    no CUDA — or a torch build without it — quietly stays on the CPU path.
    """
    want = str(_cfg("KOKORO_DEVICE", "auto")).strip().lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        log.warning("Kokoro device: CUDA probe failed (%s) — synthesising on CPU.", exc)
        return "cpu"
    if not available:
        if want == "cuda":
            log.warning("Kokoro device: KOKORO_DEVICE='cuda' but torch reports no "
                        "CUDA device — synthesising on CPU.")
        return "cpu"
    return "cuda"


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for the VRAM-exhaustion family, whatever torch chose to raise."""
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


class _KokoroTTS:
    """Local Kokoro neural voice. Exposes .synth(text) -> float32 numpy audio."""

    sample_rate = 24000

    def __init__(self) -> None:
        from kokoro import KPipeline
        self._KPipeline = KPipeline
        self.device = _resolve_kokoro_device()
        self._pipeline = KPipeline(lang_code=config.KOKORO_LANG, device=self.device)
        self.voice = config.KOKORO_VOICE
        self.speed = config.KOKORO_SPEED

    @staticmethod
    def _run(pipeline, text: str, voice: str, speed):
        """Drive one pipeline, yielding host-side float32 samples.

        The `.cpu()` copy happens in here on purpose: on CUDA the generator
        yields while kernels are still queued, so an allocation failure can
        surface during the copy rather than during the call.
        """
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            yield arr.astype("float32")

    def _degrade_to_cpu(self, exc: BaseException) -> None:
        """Rebuild the pipeline on the CPU after CUDA ran out of memory."""
        log.error("Kokoro CUDA out of memory (%s) — falling back to CPU synthesis "
                  "for the rest of this session.", exc)
        self.device = "cpu"
        self._pipeline = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipeline = self._KPipeline(lang_code=config.KOKORO_LANG, device="cpu")
        log.info("Kokoro voice now on CPU (degraded, ~19x slower synthesis).")
        # Same reasoning as the Kokoro->Piper alert below: still speaking, but
        # noticeably slower, and the owner should know why without reading the log.
        alerts.send_alert("voice.tts", "CudaOutOfMemory",
                          "Kokoro ran out of VRAM; degraded to CPU synthesis for this session.")

    def generate(self, text: str):
        """
        Yield float32 numpy chunks for `text`.

        CUDA OOM degrades to the CPU pipeline rather than killing the voice —
        the same shape as the Kokoro->Piper fallback in _build_tts(). There is
        only ~300 MiB of headroom at peak, so a game or a video can take
        Kokoro's allocation out from under it; that must cost latency, not
        speech. If part of the utterance already played we re-raise instead of
        re-synthesising, so the caller's own fallback handles the remainder and
        nothing is spoken twice.
        """
        yielded = False
        try:
            for arr in self._run(self._pipeline, text, self.voice, self.speed):
                yielded = True
                yield arr
            return
        except Exception as exc:
            if not (self.device == "cuda" and _is_cuda_oom(exc)):
                raise
            self._degrade_to_cpu(exc)
            if yielded:
                raise
        for arr in self._run(self._pipeline, text, self.voice, self.speed):
            yield arr

    def synth(self, text: str) -> np.ndarray:
        chunks = [arr for arr in self.generate(text)]
        if not chunks:
            return np.zeros(1, dtype="float32")
        return np.concatenate(chunks)


def _build_tts():
    """
    Build the configured TTS engine. Tries Kokoro first (more natural); if it
    fails to load for any reason, falls back to the local Piper voice so speech
    never silently dies.
    """
    if config.TTS_ENGINE == "kokoro":
        try:
            tts = _KokoroTTS()
            log.info("Kokoro voice loaded: %s (lang %s, speed %s, device %s)",
                     tts.voice, config.KOKORO_LANG, tts.speed, tts.device)
            return tts
        except Exception as exc:
            log.error("Kokoro load failed (%s) — falling back to Piper.", exc)
            # Not fatal — Piper still speaks — but the voice has silently changed
            # from British Kokoro to the fallback, which is worth knowing about.
            alerts.send_alert("voice.tts", type(exc).__name__,
                              "Kokoro failed to load; fell back to the Piper voice.")

    onnx = config.PIPER_VOICE_DIR / f"{config.PIPER_VOICE}.onnx"
    cfg  = config.PIPER_VOICE_DIR / f"{config.PIPER_VOICE}.onnx.json"
    if not onnx.exists():
        # Both engines are now unavailable: Bastion starts mute, or not at all.
        # The path stays local — only the fact that the model is missing goes out.
        alerts.send_alert("voice.tts", "PiperVoiceMissing",
                          "Piper voice model file is missing; no TTS engine available.")
        raise FileNotFoundError(
            f"Piper voice model not found: {onnx}\n"
            f"Run gen_samples.py once to download voice models."
        )
    try:
        voice = PiperVoice.load(str(onnx), config_path=str(cfg))
    except Exception as exc:
        alerts.send_alert("voice.tts", type(exc).__name__,
                          "Piper fallback voice failed to load; Bastion has no TTS.")
        raise
    log.info("Piper voice loaded: %s", config.PIPER_VOICE)
    return voice


def speak(tts, text: str) -> None:
    """
    Synthesise text locally and play it on the default output device.

    Handles both the Kokoro engine (_KokoroTTS) and the Piper fallback. A fresh
    synthesis + playback happens every call, so there's no engine-state carryover.
    sd.play targets the current default device, so it follows speaker/headphone
    switching automatically.
    """
    if not text.strip():
        return

    engine = "kokoro" if isinstance(tts, _KokoroTTS) else "piper"
    try:
        if isinstance(tts, _KokoroTTS):
            audio = tts.synth(text)
            sd.play(audio, samplerate=tts.sample_rate)
            sd.wait()
            return

        # Piper fallback: synthesise into an in-memory WAV, read back as PCM samples.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            tts.synthesize_wav(text, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            frames      = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

        sd.play(audio, samplerate=sample_rate)
        sd.wait()   # block until playback finishes before returning to idle
    except Exception as exc:
        # Synthesis or playback died — Bastion is mute even though everything
        # upstream looks healthy. Engine name only; the text being spoken is
        # user content and must not leave the machine. Re-raised so the existing
        # callers handle it exactly as they did before.
        log.error("TTS playback failed (%s): %s", engine, exc)
        alerts.send_alert("voice.speak", type(exc).__name__,
                          f"TTS synthesis/playback failed ({engine}); Bastion cannot speak.")
        raise


# ─── Sentence-level streaming TTS ─────────────────────────────────────────────
#
# speak() above synthesises the WHOLE reply before a single sample is played, so
# time-to-first-audio is the full synthesis time. On CUDA that is ~0.25 s for a
# typical reply (65x realtime); on the CPU fallback it is ~5-6 s (3.6x realtime),
# and 16.4 s on a long 812-char one — which is what this path was built for.
#
# CUDA shrinks the win but does not remove it: whole-reply playback still waits
# for the model to finish generating, and streaming does not. Kokoro emits a
# single chunk under ~500 chars, so the sentence split has to happen here.
#
# The streaming path below cuts the reply into sentences as the model emits it,
# synthesises them on one worker thread and plays them on another, through a
# single continuous sounddevice OutputStream. Speech starts after sentence one
# instead of after sentence twelve; synthesis (3.6x realtime) outruns playback
# (1x) by a wide margin, so once the first chunk is out the rest stays ahead.
#
# Settings are read with getattr so this file keeps working before/without the
# matching entries in config.py (which is the single source of truth for them).

def _cfg(name: str, default):
    return getattr(config, name, default)


# Tokens that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset("""
mr mrs ms dr prof sr jr st vs etc al inc ltd co corp dept fig vol pp
gen col sgt capt lt cmdr rev hon sen rep gov pres
mt ave blvd rd apt no
a.m p.m e.g i.e u.s u.k ph.d b.sc m.sc approx est max min sec
""".split())

_TERMINALS = ".!?…"
_CLOSERS   = "\"')]}»”’"
_OPENERS   = "\"'“‘(["


def _is_sentence_break(buf: str, dot: int, run: str, rest: str) -> bool:
    """
    Decide whether the terminator run starting at buf[dot] really ends a
    sentence. The caller has already established it is followed by whitespace
    (or the end of a finished reply), which alone rules out "3.5 GB",
    "127.0.0.1" and "v2.1" — there is no space inside those.

    What is left is the ambiguous case: a period, then a space.
    """
    if run[0] in "!?":
        return True

    nxt = rest.lstrip()[:1]
    if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in _OPENERS):
        # A new sentence starts with a capital. A lowercase continuation means
        # the period belonged to the previous token: "e.g. this", "3.5 gb",
        # "Dr. smith". Conservative on purpose — a missed break costs a slightly
        # longer chunk, a false break costs an audibly chopped number.
        return False

    if len(run) > 1:
        return True   # "..." or "?!" followed by a capital

    # The token immediately before the period.
    k = dot - 1
    while k >= 0 and (buf[k].isalnum() or buf[k] in ".&"):
        k -= 1
    word = buf[k + 1:dot].lower()
    if word in _ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha():
        return False        # an initial: "J. Smith", "Harry S. Truman"
    if word.replace(".", "").isdigit():
        return False        # "1. Install it" — a list marker, not a sentence
    return True


def _boundaries(buf: str, final: bool):
    """Yield every index in buf that speech may be cut at (index is exclusive)."""
    i, n = 0, len(buf)
    while i < n:
        ch = buf[i]
        if ch == "\n":
            yield i + 1     # a line break is always a safe place to breathe
            i += 1
            continue
        if ch not in _TERMINALS:
            i += 1
            continue
        j = i + 1
        while j < n and buf[j] in _TERMINALS:
            j += 1          # swallow "..." / "?!"
        run = buf[i:j]
        while j < n and buf[j] in _CLOSERS:
            j += 1          # keep a closing quote with its sentence
        after = buf[j:j + 1]
        if after and not after.isspace():
            i = j           # mid-token: a decimal, an IP address, "e.g.,"
            continue
        if not after and not final:
            return          # more tokens may still be coming — do not guess
        if _is_sentence_break(buf, i, run, buf[j:]):
            yield j
        i = j


class _SentenceSplitter:
    """
    Turns a token stream into speakable chunks.

    The first chunk is kept short (latency is all that matters for it) and each
    later chunk may be at most TTS_CHUNK_GROWTH times the one before, up to a
    target length.

    The ramp is not cosmetic — it is what stops the reply gapping after sentence
    one. A 33-char opener is ~2 s of speech, and a 240-char second chunk needs
    ~3.4 s of generation plus ~4 s of synthesis to be ready, so playback runs
    dry. Measured: 574-2381 ms of silence after chunk one before the ramp, 0.0-
    0.2 ms after it. Chunk k+1 stays ready in time as long as it is under about
    1.9x chunk k, given ~80 chars/s of generation and ~3.4x-realtime synthesis.
    """

    def __init__(self) -> None:
        self.first_max  = int(_cfg("TTS_FIRST_CHUNK_CHARS", 120))
        self.target     = int(_cfg("TTS_CHUNK_CHARS", 240))
        self.min_chars  = int(_cfg("TTS_MIN_CHUNK_CHARS", 24))
        self.max_buffer = int(_cfg("TTS_MAX_BUFFER_CHARS", 420))
        self.growth     = float(_cfg("TTS_CHUNK_GROWTH", 1.8))
        self.word_cut_chunks = int(_cfg("TTS_WORD_CUT_MAX_CHUNKS", 3))
        self._buf = ""
        self._emitted = False
        self._last_len = 0
        self._count = 0

    @property
    def buffered(self) -> str:
        return self._buf

    def feed(self, text: str) -> list[str]:
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        out = self._drain(final=True)
        self._buf = ""
        return out

    def clear(self) -> None:
        self._buf = ""

    def _clause_cut(self, limit: int) -> int | None:
        """
        Largest comma/semicolon/dash break inside [min_chars, limit].

        If there is no punctuation to breathe at and the previous chunk was
        short, fall back to a word boundary: a hesitation mid-sentence is worth
        it only while the queue is genuinely at risk of running dry, which is
        the first two or three chunks and no further.

        That "and no further" has to be counted, not inferred from _last_len:
        during the ramp _last_len is under 100 almost by construction, so the
        length test alone let a word cut land at chunk 7 of 8 — where the queue
        is seconds ahead and the only thing a mid-word cut buys is a stutter.
        Past word_cut_chunks the caller takes the whole sentence instead.
        """
        best = None
        end = min(limit, len(self._buf) - 1)
        for i in range(self.min_chars, max(self.min_chars, end)):
            if self._buf[i] in ",;:—–" and self._buf[i + 1].isspace():
                best = i + 1
        if (best is None
                and self._count < self.word_cut_chunks
                and self._last_len < int(_cfg("TTS_WORD_CUT_BELOW_CHARS", 100))):
            sp = self._buf.rfind(" ", self.min_chars, end)
            if sp > 0:
                best = sp + 1
        return best

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while self._buf:
            limit = (self.first_max if not self._emitted
                     else min(self.target, max(self.min_chars,
                                               int(self._last_len * self.growth))))
            cuts = [b for b in _boundaries(self._buf, final)
                    if len(self._buf[:b].strip()) >= self.min_chars]
            cut = None
            over_budget = (cuts[0] > limit) if cuts else (len(self._buf) >= limit)
            if over_budget and limit < self.target:
                # Still ramping and the next sentence is longer than the budget:
                # breathe at a clause boundary instead of letting playback run
                # dry waiting for a full stop. Only during the ramp — once the
                # queue is ahead, cutting mid-sentence buys nothing and costs
                # prosody.
                clause = self._clause_cut(limit)
                if clause:
                    cut = clause
            if cut is None and cuts:
                within = [b for b in cuts if b <= limit]
                # Largest chunk that still fits the limit; if even the first
                # sentence is longer than the limit, take it whole rather than
                # cutting a sentence in half.
                cut = within[-1] if within else cuts[0]
            elif final and self._buf.strip():
                cut = len(self._buf)
            elif len(self._buf) >= self.max_buffer:
                # A run-on with no usable punctuation (a long list, a URL dump).
                # Break at whitespace so playback never stalls waiting for a
                # full stop that is not coming.
                sp = self._buf.rfind(" ", self.min_chars, self.max_buffer)
                cut = sp + 1 if sp > 0 else None
            if cut is None:
                break
            if cut > len(self._buf):
                break
            piece = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if piece:
                out.append(piece)
                self._emitted = True
                self._last_len = len(piece)
                self._count += 1
        return out


def _synth_chunks(tts, text: str):
    """
    Yield (float32-or-int16 samples, samplerate) for text, in as many pieces as
    the engine gives us. Kokoro yields per internal segment, so a long chunk
    starts playing before the whole chunk is synthesised.
    """
    if isinstance(tts, _KokoroTTS):
        for arr in tts.generate(text):
            yield arr, tts.sample_rate
        return

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        tts.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        frames      = wf.readframes(wf.getnframes())
    yield np.frombuffer(frames, dtype=np.int16), sample_rate


class SpeechStream:
    """
    Speak a reply while it is still being generated.

    Usage:
        st = SpeechStream(tts)
        st.feed(delta)            # for every content delta from the model
        st.feed(None)             # this round was a tool call — drop it unspoken
        leftover = st.finish()    # blocks until playback ends
        if leftover: speak(tts, leftover)   # degraded path, see below

    Failure policy: if splitting, synthesis or playback raises, the stream stops
    and finish() returns everything that was NOT spoken, so the caller can fall
    back to the old synthesise-then-play path for the remainder. A broken stream
    makes Bastion slow, never silent.
    """

    def __init__(self, tts, on_first_audio=None, sink=None) -> None:
        self._tts = tts
        self._on_first_audio = on_first_audio
        self._sink = sink                 # test hook; None = real audio device
        self._splitter = _SentenceSplitter()
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=int(_cfg("TTS_AUDIO_QUEUE", 6)))
        self._cancel  = threading.Event()
        self._lock    = threading.Lock()
        self._pending: list[str] = []     # handed to synthesis, not yet spoken
        self._threads: list[threading.Thread] = []
        self._out_stream = None
        self._unstreamed: list[str] = []  # deltas that arrived after a failure
        self._splitter_dead = False
        self.failed = False
        self.started = False              # any text was ever queued
        self.spoke = False                # any audio actually reached the device
        self.first_audio_at: float | None = None
        self.marks: list[dict] = []       # timing rows, for the benchmarks

    # ── producer side ────────────────────────────────────────────────────────

    def feed(self, delta) -> None:
        """Accept one content delta. delta=None discards everything unspoken."""
        if delta is None:
            self.discard()
            return
        if self._cancel.is_set() or not delta:
            return
        if self.failed:
            # The stream is dead but the model is still talking. Keep the text
            # so finish() can hand the whole remainder to the blocking path.
            with self._lock:
                self._unstreamed.append(delta)
            return
        try:
            for piece in self._splitter.feed(delta):
                self._enqueue(piece)
        except Exception as exc:
            # The splitter is the one component whose failure could swallow text
            # silently, so recover its buffer (and this delta, which it may not
            # have taken) before writing it off.
            log.error("Streaming TTS split failed: %s", exc)
            try:
                tail = self._splitter.buffered
            except Exception:
                tail = ""
            if not tail.endswith(delta):
                tail += delta
            self._fail("")
            with self._lock:
                self._splitter_dead = True
                self._unstreamed.append(tail)

    def _enqueue(self, text: str) -> None:
        with self._lock:
            if self._cancel.is_set() or self.failed:
                return
            self._pending.append(text)
            if not self.started:
                self.started = True
                self._start_threads()
        self._text_q.put((text, time.perf_counter()))

    def _start_threads(self) -> None:
        for target, name in ((self._synth_worker, "tts-synth"),
                             (self._play_worker,  "tts-play")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    # ── control ──────────────────────────────────────────────────────────────

    def discard(self) -> None:
        """Drop queued speech that has not started playing (tool-call round)."""
        with self._lock:
            self._splitter.clear()
            self._pending.clear()
        _drain_queue(self._text_q)
        _drain_queue(self._audio_q)

    def stop(self) -> None:
        """Barge-in: abandon everything, including audio already in the device."""
        self._cancel.set()
        self.discard()
        stream = self._out_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass

    def finish(self, timeout: float = 300.0) -> str:
        """
        Close the stream and block until playback ends. Returns any text that
        was never spoken — '' when everything was.
        """
        try:
            if not self.failed and not self._cancel.is_set():
                for piece in self._splitter.flush():
                    self._enqueue(piece)
        except Exception as exc:
            log.error("Streaming TTS flush failed: %s", exc)
            self._fail("")

        if not self.started:
            return self._leftover()

        self._text_q.put(None)
        deadline = time.perf_counter() + timeout
        for t in self._threads:
            t.join(max(0.1, deadline - time.perf_counter()))
        if any(t.is_alive() for t in self._threads):
            log.error("Streaming TTS did not finish within %.0fs — stopping.", timeout)
            self.stop()
        return self._leftover()

    def _leftover(self) -> str:
        if self._cancel.is_set():
            return ""      # barge-in: silence is what was asked for
        with self._lock:
            parts = list(self._pending)
            if not self._splitter_dead:
                tail = self._splitter.buffered.strip()
                if tail:
                    parts.append(tail)
            parts.append("".join(self._unstreamed).strip())
            self._pending.clear()
            self._unstreamed.clear()
        return " ".join(p for p in parts if p).strip()

    def _fail(self, current_text: str) -> None:
        """Give up on streaming. current_text is the chunk that died — it goes
        back on the pending list so the caller's fallback still speaks it."""
        with self._lock:
            self.failed = True
            if current_text and current_text not in self._pending:
                self._pending.insert(0, current_text)
        _drain_queue(self._audio_q)

    # ── workers ──────────────────────────────────────────────────────────────

    def _synth_worker(self) -> None:
        try:
            while True:
                item = self._text_q.get()
                if item is None:
                    break
                text, t_queued = item
                if self._cancel.is_set() or self.failed:
                    continue
                try:
                    got = False
                    for audio, sr in _synth_chunks(self._tts, text):
                        if self._cancel.is_set():
                            break
                        got = True
                        self._audio_q.put((audio, sr, text, t_queued, time.perf_counter()))
                except Exception as exc:
                    log.error("Streaming TTS synthesis failed: %s", exc)
                    alerts.send_alert("voice.speak", type(exc).__name__,
                                      "Streaming TTS synthesis failed; falling back to "
                                      "whole-reply playback.")
                    self._fail(text)
                    break
                else:
                    with self._lock:
                        if text in self._pending and got:
                            self._pending.remove(text)
        finally:
            self._audio_q.put(None)

    def _play_worker(self) -> None:
        try:
            while True:
                item = self._audio_q.get()
                if item is None:
                    break
                if self._cancel.is_set() or self.failed:
                    # Keep consuming: the synth worker blocks on a full queue,
                    # and a stalled producer would hang finish().
                    continue
                audio, sr, text, t_queued, t_synth = item
                t_start = time.perf_counter()
                try:
                    self._write(audio, sr)
                except Exception as exc:
                    log.error("Streaming TTS playback failed: %s", exc)
                    alerts.send_alert("voice.speak", type(exc).__name__,
                                      "Streaming TTS playback failed; Bastion may be mute "
                                      "for the rest of this reply.")
                    self._fail(text)
                    self._close_stream()
                    continue
                self.marks.append({
                    "chars": len(text), "queued": t_queued, "synth_done": t_synth,
                    "play_start": t_start, "play_end": time.perf_counter(),
                    "audio_s": len(audio) / float(sr),
                })
        finally:
            self._close_stream()

    def _write(self, audio: np.ndarray, sr: int) -> None:
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if self._sink is not None:
            self._mark_first()
            self._sink(audio, sr)
            return
        if self._out_stream is None:
            self._out_stream = sd.OutputStream(samplerate=sr, channels=audio.shape[1],
                                               dtype=audio.dtype)
            self._out_stream.start()
        # Write in slices so a barge-in during a long sentence is honoured in
        # ~250 ms instead of waiting for the whole chunk to buffer.
        step = max(1, int(sr * 0.25))
        for pos in range(0, len(audio), step):
            if self._cancel.is_set():
                return
            self._out_stream.write(audio[pos:pos + step])
            self._mark_first()

    def _mark_first(self) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
            self.spoke = True
            if self._on_first_audio:
                try:
                    self._on_first_audio()
                except Exception:
                    pass

    def _close_stream(self) -> None:
        stream, self._out_stream = self._out_stream, None
        if stream is None:
            return
        try:
            if not self._cancel.is_set():
                stream.stop()      # drains what is already buffered
            stream.close()
        except Exception as exc:
            log.debug("Output stream close: %s", exc)


def _drain_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


# ─── Voice / screen split, for the streaming path ─────────────────────────────
#
# hud_server.split_reply() decides which part of a reply is worth SAYING and
# which belongs on the HUD. It takes a finished string — and with TTS_STREAMING
# on, by the time voice_turn() holds the finished string the reply has already
# been read out loud. So on the streaming path the decision has to be made while
# the reply is still arriving, which is what this does.
#
# It sits between chat()'s content deltas and SpeechStream.feed, and SpeechStream
# itself is untouched: nothing about splitting, synthesis, playback or barge-in
# changes, this only stops handing it text. That is deliberate — the streaming
# path is the most fragile thing in this file and the most falsely reported.
#
# Rules, mirroring split_reply():
#   A. a bullet or numbered line ends speech at the start of that line. The
#      lead-in the model wrote to introduce the list is exactly what gets spoken.
#   A'. nothing before the first bullet — a reply that is ONLY a list — is the
#      case split_reply() got wrong until 2026-08-17 and the case the owner feels:
#      "- milk\n- eggs" is thirteen characters and used to be answered with
#      "It is on the screen." A short, few-enough list is FLATTENED and spoken
#      in full ("milk, eggs."); only a long or many-item one hands off.
#   A''. a lead-in AND a short list is the same bug one line further on, and it
#      survived the A' fix: "You need:\n- milk\n- eggs" spoke "You need." and
#      sent him to a monitor to read eleven characters. Rule A' therefore runs
#      whether or not there is a lead-in — the colon becomes the join rather
#      than a full stop, and the whole thing is one sentence: "You need: milk,
#      eggs." Only a long or many-item list still ends speech at the bullet.
#   B. past HUD_SPOKEN_MAX_CHARS, stop at the next sentence boundary.
#   C. a colon left at the end of the spoken line is a promise the voice never
#      keeps — "You need:" and then silence. When the list after it IS spoken
#      (A'') the colon simply stays, because the items keep the promise. When it
#      is not, the handoff completes the sentence instead: "You need: the rest
#      is on the screen." Closing it with a full stop was the older fix and
#      split_reply() moved off it — it spoke the bare fragment "You need."
#
# ONE DELIBERATE DIVERGENCE from split_reply(), and it is a latency trade: text
# is forwarded EAGERLY until the budget is reached, and only then held to find
# the end of the sentence that crosses it. So the _SentenceSplitter ramp (which
# is what stops playback gapping after chunk one — see its docstring) keeps
# working on the early part of the reply, where time-to-first-audio lives.
#
# The cost is that split_reply() stops BEFORE the sentence that would overrun
# and this one finishes it: measured 307 spoken chars against its 238 on the
# same 424-char reply. Erring long is the right direction — the overshoot is
# bounded by one sentence, and the failure it avoids is a reply cut shorter than
# the model's own point. Withholding whole sentences from the start would have
# matched exactly, at the price of delaying first audio to the first full stop,
# which is the one number this project has spent two days on.


def _list_line_re():
    """hud_server owns the pattern; don't grow a second copy of it here."""
    return getattr(hud, "_LIST_LINE_RE", None)


# A colon at the very end of what has arrived so far is ambiguous: it may be the
# middle of a sentence ("at 9:15", "note: it works") or the lead-in to a list we
# are about to stop speaking at. So it is held back rather than forwarded, and
# resolved once the next characters arrive. Cheap — a couple of characters, and
# only ever at the tip of the buffer.
_TRAIL_COLON_RE = re.compile(r":\s*$")


def _close_colon(text: str) -> str:
    """'You need:' → 'You need.' Mirrors the same step in split_reply()."""
    stripped = text.rstrip()
    if stripped.endswith(":"):
        return stripped[:-1].rstrip() + "."
    return text


def _list_items(text: str) -> list[str]:
    """The lines of a bullets-only reply, markers stripped — exactly the list
    split_reply() builds, so the two agree on the item count and the flattened
    length that decide whether it is spoken at all. Every non-blank line counts,
    bullet or not: a trailing 'That is all' is an item to split_reply()."""
    rx = _list_line_re()
    out = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        item = " ".join((rx.sub("", ln) if rx is not None else ln).split())
        if item:
            out.append(item)
    return out


class _VoiceSplit:
    """Forward the speakable part of a streaming reply; suppress the rest.

    Suppressed text is never handed to SpeechStream, so it cannot come back out
    of finish() as `leftover` and get read aloud the blocking way — which is the
    failure this would otherwise have: the bullets you just decided not to say,
    said anyway, slowly, after the reply ended.
    """

    def __init__(self, feed, budget: int | None = None,
                 handoff: str | None = None, screen_only: str | None = None,
                 max_items: int | None = None):
        self._feed = feed
        self._budget = int(budget if budget is not None
                           else getattr(hud, "HUD_SPOKEN_MAX_CHARS", 240))
        self._max_items = int(max_items if max_items is not None
                              else getattr(hud, "HUD_SPOKEN_MAX_ITEMS", 6))
        self._handoff = (handoff if handoff is not None
                         else getattr(hud, "HUD_HANDOFF_PHRASE", "The rest is on the screen."))
        self._screen_only = (screen_only if screen_only is not None
                             else getattr(hud, "HUD_SCREEN_ONLY_LINE", "It is on the screen."))
        self._re = _list_line_re()
        self.reset()

    def reset(self) -> None:
        """Back to the start of a reply. Called on a retracted tool-call round —
        the preamble is gone, so its characters must not count against the
        budget of the reply that replaces it."""
        self._buf = ""            # accepted, not yet judged
        self._taken = 0           # chars of .text consumed out of _buf
        self._sent = 0            # chars actually handed to speech
        self._cut = False
        self._at_line_start = True
        self._spoken_tail = ""
        self._said_screen = False
        self._listing = False    # rules A'/A'': buffering a list
        self._pending_lead = ""   # lead-in held back until the list is judged
        self._said_text = ""      # everything spoken, for the reference compare
        self._spoke_list = False  # the whole list was flattened and spoken
        self.text = ""            # everything the model produced
        self.truncated = False
        self.reason = ""

    # ── producer side ────────────────────────────────────────────────────────

    def feed(self, delta):
        if delta is None:
            self.reset()
            return self._feed(None)
        if not delta:
            return
        self.text += delta
        if self._cut:
            return                # decided already; the HUD carries the rest
        self._buf += delta
        self._emit(final=False)

    def close(self) -> None:
        """End of the reply. Flush what is left and say the handoff line."""
        if not self._cut:
            self._emit(final=True)
        # Truncated only if something was genuinely left out. A reply that
        # happens to end on the budget is not "the rest is on the screen" — and
        # promising a screen panel that has nothing on it is the one failure
        # mode of this feature that would be pure noise.
        self.truncated = bool(self._cut and self.text[self._taken:].strip())
        if self._spoke_list:
            # Said in full AND worth a glance: publish the panel, promise
            # nothing. `truncated` is what voice_turn() reads to decide whether
            # to publish, so it is true here even though nothing was withheld.
            self.truncated = True
            return
        if not self.truncated:
            return
        if self._sent == 0:
            # A list too long or too many to hear, with no lead-in to speak.
            # There is no honest sentence to extract, so say the one true thing
            # and let the panel do the work.
            self._forward(self._screen_only, from_buffer=False)
        elif not self._said_screen:
            # Matched the bare word "screen" until 2026-08-17, which ate the
            # handoff for any reply that merely mentioned one — worst on the
            # vision path, whose replies always get a panel. Match the phrase.
            self._forward(" " + self._handoff, from_buffer=False)

    # ── the two rules ────────────────────────────────────────────────────────

    def _decidable(self, final: bool) -> int:
        """How much of the buffer we can judge yet.

        A line is only knowable as a bullet once ~4 characters of it exist
        ("- ", "1. "), so a partial trailing line is held back that long. Four
        characters is not a latency problem; guessing is a correctness one."""
        if final:
            return len(self._buf)
        i = self._buf.rfind("\n")
        if i < 0:
            if self._at_line_start and len(self._buf) < 4:
                return 0          # the whole reply might BE a bullet
            return len(self._buf)
        tail = self._buf[i + 1:]
        return (i + 1) if len(tail) < 4 else len(self._buf)

    def _list_start(self, seg: str) -> int | None:
        if self._re is None:
            return None
        starts = [0] if self._at_line_start else []
        k = seg.find("\n")
        while k >= 0:
            starts.append(k + 1)
            k = seg.find("\n", k + 1)
        for st in starts:
            if self._re.match(seg[st:]):
                return st
        return None

    def _emit(self, final: bool) -> None:
        if self._listing:
            self._emit_list(final)
            return

        seg = self._buf[:self._decidable(final)]
        # Rule C, the streaming half: hold a colon sitting at the tip of the
        # buffer. Once it has been handed to SpeechStream it cannot be taken
        # back, and whether it should have been a full stop depends on what
        # comes next. `final` means nothing more is coming, so nothing to hold.
        if not final:
            m = _TRAIL_COLON_RE.search(seg)
            if m:
                seg = seg[:m.start()]
        if not seg:
            return

        st = self._list_start(seg)
        if st is not None:
            head = seg[:st]
            if self._sent == 0 and not head.strip():
                # Rule A'. Nothing was written before the first bullet, so there
                # is no lead-in to speak and the whole reply is the list. Do NOT
                # decide yet: a short one gets spoken in full, and how short it
                # is is not knowable until it ends. Drop the blank head and
                # start buffering.
                self._forward(head, say="")
                self._listing = True
                self._emit_list(final)
                return
            # Rule A''. There IS a lead-in, and whether it ends the speech or
            # merely introduces it is not knowable yet either — the same reason
            # as A'. So the un-spoken remainder of the lead-in (usually just the
            # held colon; the sentence before it has already streamed, so
            # first-audio is untouched) is held with the list and judged with it.
            self._forward(head, say="")
            self._pending_lead = head
            self._listing = True
            self._emit_list(final)
            return

        # Rule B. `need` is how much more may be spoken before the budget is
        # reached; the cut lands at the first sentence end at or after it, so a
        # sentence is never chopped in half. Checked against the segment rather
        # than only between deltas — a reply that arrives in one delta has to be
        # cut too.
        need = self._budget - self._sent
        if need <= len(seg):
            cuts = [b for b in _boundaries(seg, final) if b >= max(0, need)]
            if cuts:
                head = seg[:cuts[0]]
                # A line break is a boundary, so the budget can land on
                # "…you need:\n" just as the list rule can. Same fix — but only
                # promise the screen if there is really something left: at
                # `final` the reply may simply have ended on that colon.
                more = (not final) or bool(self._buf[len(head):].strip())
                self._forward(head, say=self._end_colon(head, more))
                self._stop("budget")
                return
            if final:
                self._forward(seg)
                self._stop("budget")
                return
            # HOLD. Nothing more is forwarded until the sentence that crosses
            # the budget ends. Holding rather than forwarding-and-looking-later
            # is what makes this independent of delta size: at one character per
            # delta, a "." and the space after it arrive in different calls, so
            # a boundary search inside a single delta never fires and the whole
            # reply gets spoken. Caught at feed_size=1; the queue is seconds
            # ahead by char 240, so the buffering costs nothing audible.
            return

        self._forward(seg)

    def _reference_tail(self) -> str | None:
        """What split_reply() would still say, given what has been said already.

        The DECISION is made locally and eagerly above — it has to be, because a
        streamed list disqualifies itself mid-flight and split_reply() only ever
        sees finished text. But once the reply has ended the whole of it is in
        hand, so the flattened WORDING — the join, the colon, the closing full
        stop — is taken from the spec itself rather than from a second copy of
        it here that drifts the next time the spec is tuned.

        Returns None if hud is a stub, if it raises, or if the two disagree
        (a mismatched prefix, or a handoff where we decided to speak). In every
        one of those cases the local wording is used, so this can improve the
        match but never break it."""
        fn = getattr(hud, "split_reply", None)
        if fn is None:
            return None
        try:
            ref = " ".join(str(fn(self.text)[0]).split())
        except Exception:
            return None
        if not ref or ref == self._screen_only or ref.endswith(self._handoff):
            return None
        said = " ".join(self._said_text.split())
        if not ref.startswith(said):
            return None
        return ref[len(said):]

    def _end_colon(self, text: str, handing_off: bool = True) -> str:
        """Rule C, mirroring the same step at the end of split_reply().

        A colon at the end of the spoken line is completed by the handoff clause
        rather than closed with a full stop, so it introduces something real
        instead of leaving the fragment "You need." The handoff is then already
        said — `_forward` notices the phrase and makes it sticky, so close() does
        not append it a second time. `handing_off` is false when nothing is
        actually being withheld, where promising a screen would be a lie."""
        stripped = text.rstrip()
        if not stripped.endswith(":"):
            return text
        if not handing_off or self._said_screen:
            return stripped[:-1].rstrip() + "."
        return (stripped[:-1].rstrip() + ": "
                + self._handoff[0].lower() + self._handoff[1:])

    def _flush_lead(self) -> None:
        """Speak the held lead-in as the END of the spoken line — rule C. Used
        when the list turns out to be too long or too many to say."""
        if not self._pending_lead:
            return
        lead, self._pending_lead = self._pending_lead, ""
        self._forward(lead, from_buffer=False,
                      say=self._end_colon(lead, bool(self._buf.strip())))

    def _emit_list(self, final: bool) -> None:
        """Rules A' and A'': the rest of the reply is a list.

        Everything stays in `_buf`, unspoken, until either it disqualifies
        itself or the reply ends. The cost is real and worth naming: for a
        bullets-only reply, first audio waits for the last token. It is the only
        honest option — item seven is what makes a six-item list too long, and
        it has not arrived yet — and it is bounded by the same test that
        disqualifies the list, so the wait is at most a few hundred characters.
        With a lead-in (A'') the cost is far smaller: the lead-in sentence has
        already streamed, and only its tail — normally the held colon — waits.
        Every other shape of reply still streams eagerly."""
        lead  = " ".join(self._pending_lead.split())
        items = _list_items(self._buf)
        flat  = ", ".join(items)
        # What would be added to the spoken line if the list were said. Compared
        # against the budget INCLUDING what has already been spoken, because
        # split_reply() measures the whole joined sentence, lead-in and all —
        # and the closing full stop is appended after its test, so not counted.
        candidate = (lead + " " + flat).strip() if lead else flat
        if len(items) > self._max_items or self._sent + len(candidate) > self._budget:
            # Too much to hear. Decided the moment it becomes true, so a long
            # list does not sit in the buffer waiting for the reply to end.
            self._flush_lead()
            self._stop("list")
            return
        if not final:
            return
        if not items:
            self._flush_lead()
            self._stop("list")
            return
        say = self._reference_tail()
        if say is None:
            say = candidate
            if say[-1] not in ".!?":
                say += "."
            # Joined onto a lead-in that is already out loud, so it needs the
            # space the collapse above removed — unless it opens with the colon,
            # which belongs tight against the word before it.
            if self._sent and not say.startswith(":"):
                say = " " + say
        self._pending_lead = ""
        # Nothing is withheld, so no handoff line — but the panel is still
        # published, because a list is worth a glance even when it was spoken.
        self._forward(self._buf, say=say)
        self._spoke_list = True
        self._stop("list-spoken")

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _forward(self, text: str, from_buffer: bool = True,
                 say: str | None = None) -> None:
        """Consume `text` from the buffer and speak `say` (default: `text`).

        They differ only where the spoken form is edited — a closed colon, a
        flattened list — and the split has to stay: the buffer arithmetic must
        count the ORIGINAL characters or `truncated` misjudges what was left."""
        if from_buffer:
            self._buf = self._buf[len(text):]
            self._taken += len(text)
        if text:
            self._at_line_start = text.endswith("\n")
        if say is None:
            say = text
        if not say:
            return
        self._sent += len(say)
        self._said_text += say
        self._spoken_tail = (self._spoken_tail + say)[-200:]
        # Sticky, because split_reply() tests the WHOLE spoken string and the
        # tail is a 200-char window: a reply that said "on the screen" early and
        # then ran on would otherwise be handed off twice. Checked here rather
        # than at close() so the phrase is still caught when it straddles two
        # deltas.
        if "on the screen" in self._spoken_tail.lower():
            self._said_screen = True
        self._feed(say)

    def _stop(self, reason: str) -> None:
        self._cut = True
        self._buf = ""      # whatever was in flight is the screen's problem now
        self.reason = reason

    @property
    def spoken_chars(self) -> int:
        return self._sent


# ─── "One moment, sir" — the wait acknowledgement ─────────────────────────────
#
# A turn that calls tools produces no content deltas until every round is done.
# Measured from the owner's real turns on 2026-08-18: first audio at 17.9 s, 18.5 s,
# 51.0 s and 105.8 s, and six turns where NO content delta ever arrived. During
# all of that the speakers are silent and there is nothing to tell him the thing
# is alive — which is what "slow and unresponsive" actually is. A HUD on another
# monitor does not fix a room you are listening in.
#
# So: if a turn has run WAIT_ACK_AFTER_S with no content, say something short.
# Three decisions worth not re-litigating:
#
# 1. It goes through the SAME SpeechStream as the reply, not a second sd.play().
#    Two output paths would race for the device and the ack could be talking
#    while the first sentence starts. One stream, one FIFO, no interleaving.
# 2. It stops the moment the first content delta arrives — never mid-reply, and
#    never after the answer has begun.
# 3. Speaking makes barge-in LIVE (on_first_audio marks _speaking), so an ack is
#    also the moment the owner gets the ability to cut the turn off. That is a
#    feature: before this, a 105 s tool turn could not be interrupted at all,
#    because nothing was making sound.

class _WaitAck:
    """Speaks a short holding phrase while a turn is generating nothing.

    Never raises: an acknowledgement failing must not take the turn with it."""

    def __init__(self, feed, phrases=None, after_s=None, repeat_s=None,
                 max_acks=None):
        self._feed = feed
        self._phrases = list(phrases if phrases is not None
                             else _cfg("WAIT_ACK_PHRASES",
                                       ("One moment, sir.",
                                        "Still working on it, sir.",
                                        "Nearly there, sir.")))
        # 4 s, not 6: the ack has to be SYNTHESISED before it is audible, and
        # `latency-profile` measured Kokoro on CPU at 2617 ms for a 54-char
        # sentence (bench_tts.py, warm, median of 5) — 2.6x what config.py's
        # comment claims. A 6 s trigger would put the first sound at ~8.6 s.
        self._after = float(after_s if after_s is not None
                            else _cfg("WAIT_ACK_AFTER_S", 4.0))
        self._repeat = float(repeat_s if repeat_s is not None
                             else _cfg("WAIT_ACK_REPEAT_S", 25.0))
        self._max = int(max_acks if max_acks is not None
                        else _cfg("WAIT_ACK_MAX", 3))
        self._done = threading.Event()   # content arrived, or the turn ended
        self._lock = threading.Lock()    # makes "still no content?" + feed atomic
        self._thread = None
        self.spoken = 0                  # how many acks were actually said

    def start(self) -> None:
        if not _cfg("WAIT_ACK_ENABLED", True) or not self._phrases or self._max <= 0:
            self._done.set()
            return
        self._thread = threading.Thread(target=self._run, name="tts-waitack",
                                        daemon=True)
        self._thread.start()

    def cancel(self, join: bool = True) -> None:
        """Content arrived (or the turn is over). No further acks, ever.

        Takes the lock so a `feed` already in flight completes first — an ack
        half-fed into the splitter would land inside the reply's first
        sentence."""
        self._done.set()
        if join:
            with self._lock:
                pass
            t = self._thread
            if t is not None and t is not threading.current_thread():
                t.join(timeout=2.0)

    def _run(self) -> None:
        wait = self._after
        while self.spoken < self._max:
            if self._done.wait(timeout=max(0.05, wait)):
                return
            with self._lock:
                if self._done.is_set() or _barged.is_set():
                    return
                try:
                    self._feed(self._phrases[min(self.spoken,
                                                 len(self._phrases) - 1)] + " ")
                    self.spoken += 1
                except Exception as exc:
                    log.debug("Wait acknowledgement failed: %s", exc)
                    return
            wait = self._repeat
        log.info("Wait acknowledgement: %d spoken; going quiet for the rest of "
                 "the turn.", self.spoken)


# ─── Sound cues ───────────────────────────────────────────────────────────────
#
# Two short tones, synthesised here with numpy. Nothing is downloaded and no file
# is read, so a cue cannot fail because an asset is missing — which also keeps
# this inside the local-first rule with nothing to argue about.
#
# The two are deliberately different SHAPES, not just different pitches, because
# on a wake-word turn they play a fifth of a second apart: "wake" is a rising
# two-note blip (heard you), "listen" is a single softer note that fades (mic is
# open). On a follow-up turn only "listen" plays, which is what tells the owner the
# window is still open without Bastion having said anything.

_CUE_RATE = 24000
_CUES: dict = {}


def _tone(freq: float, ms: int, decay: float = 3.2) -> np.ndarray:
    """One note: sine plus a quiet octave, raised-cosine attack, exponential tail.

    The attack matters more than it sounds like it should — a hard start on a
    24 kHz stream is an audible click, and a click is exactly the kind of small
    ugly noise that makes someone turn cues off."""
    n = max(1, int(_CUE_RATE * ms / 1000.0))
    t = np.arange(n, dtype="float32") / _CUE_RATE
    w = np.sin(2 * np.pi * freq * t).astype("float32")
    w += (0.22 * np.sin(2 * np.pi * freq * 2 * t)).astype("float32")
    env = np.exp(-np.linspace(0.0, decay, n, dtype="float32"))
    atk = max(1, int(_CUE_RATE * 0.006))
    env[:atk] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, atk, dtype="float32"))
    return (w * env).astype("float32")


def _cue_audio(name: str) -> np.ndarray:
    if name not in _CUES:
        if name == "wake":
            _CUES[name] = np.concatenate([_tone(784.0, 70, 4.5),
                                          _tone(1174.7, 120, 3.0)])
        else:   # "listen"
            _CUES[name] = _tone(987.8, 110, 4.0)
    return _CUES[name]


def play_cue(name: str) -> None:
    """
    Play a short interface tone. Never raises: a cue is feedback, and feedback
    failing must never be the thing that stops a turn.

    Blocking on purpose, and short (~0.2 s). The listen cue is played BEFORE the
    recorder opens the mic precisely so its own tail cannot be recorded as the
    first half-second of speech.
    """
    if not _cfg("SOUND_CUES_ENABLED", True):
        return
    try:
        vol = max(0.0, min(1.0, float(_cfg("SOUND_CUE_VOLUME", 0.18))))
        if vol <= 0.0:
            return
        sd.play(_cue_audio(name) * vol, samplerate=_CUE_RATE, blocking=True)
    except Exception as exc:
        log.debug("Sound cue %r failed: %s", name, exc)


# The stream currently speaking, so an interrupt from another thread (wake word,
# hotkey, stop command) can cut it off. Assigned under _speaking_lock.
_active_stream: SpeechStream | None = None
_speaking_lock = threading.Lock()


def stop_speaking() -> None:
    """Barge-in hook: abandon whatever Bastion is saying right now."""
    with _speaking_lock:
        st = _active_stream
    if st is not None:
        st.stop()
    else:
        try:
            sd.stop()
        except Exception:
            pass


def _mark_speaking_started() -> None:
    """Bastion has started making sound. Barge-in is live from here."""
    global _speaking_since
    _speaking_since = time.perf_counter()
    _barged.clear()
    _speaking.set()


def _mark_speaking_ended() -> None:
    _speaking.clear()


def _speak_interruptible(tts, text: str) -> None:
    """Whole-reply playback that the owner can actually stop.

    speak() is a blocking sd.play/sd.wait, and stop_speaking() DOES abort it
    (via sd.stop, since _active_stream is None on this path). What was missing
    is _speaking: both _on_hotkey and _handle_wake test it before treating a
    sound as something to interrupt. Every fallback playback in voice_turn ran
    with _speaking CLEAR — after the streaming path's own _mark_speaking_ended()
    — so the wake word logged "mid-turn with no audio playing" and queued, and
    the hotkey queued a summons, while a 40-second reply played on regardless.
    That is the "I can't interrupt it" the owner reported, on exactly the turns
    (NO DELTAS, DEGRADED) he reported it on.
    """
    if _barged.is_set():
        return          # already interrupted; do not start a new utterance
    _mark_speaking_started()
    try:
        speak(tts, text)
    finally:
        _mark_speaking_ended()


# --- Unprompted speech -------------------------------------------------------
#
# Every other path to the speakers in this file is downstream of a wake word or
# the hotkey: the owner says something, Bastion answers. Nothing could make Bastion
# speak on its own except _startup_briefing(), which cheats by running before the
# loop starts and cannot be called again safely.
#
# speak_unprompted() is the one way in. Timers, reminders, alert read-back,
# presence greetings and HUD-to-voice all go through it rather than each growing
# its own path to sd.play - one queue, one set of rules, one place to mute.
#
# THE FOUR DECISIONS, made rather than deferred:
#
# 1. Already speaking, or mid-turn?  It WAITS. The worker takes _busy, which is
#    the same lock a voice turn holds, so an announcement can never land in the
#    middle of a reply or while the owner is being recorded. It is a queue, not an
#    interrupt.
# 2. Mid-turn while the owner is TALKING?  Same answer, and this is the case the
#    design is really for: _busy is held from before the mic opens until after
#    playback ends, so there is no window in which an announcement can talk over
#    him or corrupt an in-flight turn.
# 3. Suppressible?  Yes, two ways: UNPROMPTED_SPEECH_ENABLED in config kills it
#    outright, and set_unprompted_muted(True) silences it at runtime (for a call,
#    a film, a meeting) without a restart. Muting drops what is queued rather
#    than banking it - a stack of announcements arriving the moment you unmute is
#    the behaviour that makes people turn a feature off for good.
# 4. Does it ever interrupt?  NO. Not on any priority, and there is deliberately
#    no flag to make it. A Bastion that cuts you off mid-sentence to say a timer
#    finished is worse than one that says it eleven seconds later. What it does
#    instead is EXPIRE: an item that has waited longer than UNPROMPTED_MAX_WAIT_S
#    is dropped with a log line, because a timer announcement four minutes late
#    is not information, it is noise.

_unprompted_q: queue.Queue = queue.Queue()
_unprompted_muted = threading.Event()
_unprompted_worker_started = threading.Event()
_tts_engine = None              # set by main(); the worker's handle on the voice

# Test hook. When set, the worker calls this instead of speak() - so a harness
# can prove the whole queue/defer/expire path without a sound card and without
# loading Kokoro. Production leaves it None.
_unprompted_sink = None


def set_unprompted_muted(muted: bool) -> None:
    """Silence (or re-enable) unprompted speech for the rest of the session."""
    if muted:
        _unprompted_muted.set()
        dropped = 0
        while True:
            try:
                _unprompted_q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        log.info("Unprompted speech muted (%d queued item(s) dropped).", dropped)
    else:
        _unprompted_muted.clear()
        log.info("Unprompted speech unmuted.")


def unprompted_muted() -> bool:
    return _unprompted_muted.is_set()


def speak_unprompted(text: str, source: str = "system") -> bool:
    """
    Have Bastion say something nobody asked for, at the next safe moment.

    Returns True if it was accepted onto the queue - NOT that it was spoken. It
    will be spoken once no voice turn is in flight, unless it is muted or waits
    past UNPROMPTED_MAX_WAIT_S first. Safe to call from any thread; never blocks
    on speech and never raises.

    `source` is a short caller label for the log ("timer", "reminder", ...). It
    is not spoken and never leaves the machine.
    """
    text = (text or "").strip()
    if not text:
        return False
    if not _cfg("UNPROMPTED_SPEECH_ENABLED", True):
        log.info("Unprompted speech from %s dropped: disabled in config.", source)
        return False
    if _unprompted_muted.is_set():
        log.info("Unprompted speech from %s dropped: muted.", source)
        return False

    limit = int(_cfg("UNPROMPTED_QUEUE_MAX", 8))
    if _unprompted_q.qsize() >= limit:
        # Backed up this far means something is generating announcements in a
        # loop. Drop the OLDEST - it is also the one closest to expiring - and
        # say so, rather than silently swallowing the new one.
        try:
            stale = _unprompted_q.get_nowait()
            log.warning("Unprompted queue full (%d) - dropped the oldest item from %s.",
                        limit, stale[2])
        except queue.Empty:
            pass

    _ensure_unprompted_worker()
    _unprompted_q.put((text, time.time(), source))
    log.info("Unprompted speech queued from %s (%d char(s)).", source, len(text))
    return True


def _ensure_unprompted_worker() -> None:
    """Start the delivery thread once, lazily, so callers need not care."""
    if _unprompted_worker_started.is_set():
        return
    _unprompted_worker_started.set()
    threading.Thread(target=_unprompted_worker, name="unprompted",
                     daemon=True).start()


def _unprompted_worker() -> None:
    while not _stop_listening.is_set():
        try:
            item = _unprompted_q.get(timeout=0.5)
        except queue.Empty:
            continue
        text, queued_at, source = item
        max_wait = float(_cfg("UNPROMPTED_MAX_WAIT_S", 300.0))
        try:
            _deliver_unprompted(text, queued_at, source, max_wait)
        except Exception as exc:
            # Nothing upstream of here has a caller to report to - a timer thread
            # is long gone by now - so this is the last place the failure can be
            # noticed at all.
            log.error("Unprompted speech from %s failed: %s", source, exc, exc_info=True)
            alerts.send_alert("voice.unprompted", type(exc).__name__,
                              "Unprompted speech failed; an announcement was not spoken.")


def _deliver_unprompted(text: str, queued_at: float, source: str,
                        max_wait: float) -> None:
    """Wait for a quiet moment, then say it. Gives up if the moment never comes."""
    deadline = queued_at + max_wait
    while True:
        if _stop_listening.is_set():
            return
        if _unprompted_muted.is_set():
            log.info("Unprompted speech from %s dropped while waiting: muted.", source)
            return
        if time.time() > deadline:
            # Deliberately not spoken. See decision 4 above.
            log.warning("Unprompted speech from %s expired after %.0fs waiting for a "
                        "quiet moment - not speaking it.", source, max_wait)
            return
        # _trigger set means a turn is about to start; jumping in front of it
        # would answer a question the owner has not asked yet.
        if not _trigger.is_set() and _busy.acquire(timeout=0.25):
            break
        time.sleep(0.05)

    try:
        if _trigger.is_set():
            # He reached for the hotkey in the last 250 ms. His turn wins; put
            # this back and try again in a moment.
            _unprompted_q.put((text, queued_at, source))
            return
        if _unprompted_muted.is_set() or not _cfg("UNPROMPTED_SPEECH_ENABLED", True):
            # Re-checked after taking the lock, not only before: the acquire
            # above blocks for up to 250 ms, and muting inside that window has
            # to win. Without this, "mute" could still be followed by one more
            # announcement — which is exactly the moment the mute was for.
            log.info("Unprompted speech from %s dropped at the last moment: muted.",
                     source)
            return
        log.info("Unprompted speech (%s): %s", source, text[:200])
        print(f"\nBastion > {textwrap.fill(text, width=68, subsequent_indent='         ')}\n")
        hud.send_transcript("bastion", text)
        hud.set_state("speaking")
        _mark_speaking_started()
        try:
            if _unprompted_sink is not None:
                _unprompted_sink(text, source)
            else:
                speak(_tts_engine, text)
        finally:
            _mark_speaking_ended()
            hud.set_state("idle")
    finally:
        _busy.release()
        # Unprompted speech holds _busy too, so a summons can queue behind an
        # announcement rather than behind a turn. Nothing else would consume it
        # until the END of the next real turn — a summons answered one turn
        # late. Not in the briefed patch; found by reading for other _busy
        # holders, and reported.
        _consume_queued_wake()


def _chat_supports_streaming() -> bool:
    """True once bastion.chat() accepts on_content (the call-site patch)."""
    try:
        return "on_content" in inspect.signature(chat).parameters
    except (TypeError, ValueError):
        return False


# ─── Audio device selection ───────────────────────────────────────────────────

def _resolve_input_device() -> int | None:
    """
    Find the best input device to use:
      1. If AUDIO_DEVICE_NAME is set, search for a device whose name contains
         that fragment (case-insensitive). Yeti plugged in → found; not plugged
         in → fall back to Windows default with a warning.
      2. If AUDIO_DEVICE_NAME is empty, use whatever Windows has as default.

    Returns a sounddevice device index, or None for the system default.
    """
    global _mic_missing_alerted
    name_pref = config.AUDIO_DEVICE_NAME.strip().lower()
    if not name_pref:
        return None

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_pref in d["name"].lower():
            log.info("Audio device selected: [%d] %s", i, d["name"])
            _mic_missing_alerted = False   # re-arm: a later unplug alerts again
            return i

    # Preferred device not found — warn but continue with system default
    if not _mic_missing_alerted:
        # Once per process, not once per dedupe window: this function runs on
        # every turn and inside the wake-word loop, and a Yeti that is simply
        # unplugged would otherwise alert forever.
        _mic_missing_alerted = True
        alerts.send_alert("voice.mic", "PreferredMicMissing",
                          "Preferred microphone not found; using the system default device.")
    log.warning("Device matching %r not found; using system default.", name_pref)
    print(
        f"  Warning: mic matching '{config.AUDIO_DEVICE_NAME}' not found "
        f"(is it plugged in?). Using system default instead.\n"
    )
    return None


# ─── Audio recording ──────────────────────────────────────────────────────────

def record_until_silence(device: int | None,
                         open_window_s: float | None = None) -> tuple[np.ndarray, bool]:
    """
    Record 500 ms chunks until SILENCE_DURATION_S of quiet follows speech,
    or MAX_RECORDING_S total. Returns (audio, speech_was_detected).

    `open_window_s` is how long to wait for the owner to START talking before giving
    up. It was hard-coded at 4.0 s; a conversational follow-up wants it shorter,
    which is the whole difference between "still listening" and "hanging on your
    every word". Defaults to LISTEN_WINDOW_S.

    The microphone is open only for the duration of this call, and is taken from
    the wake-word listener for exactly that long — see _acquire_mic().
    Background noise below SILENCE_THRESHOLD does not register as speech.
    """
    chunk_samples        = int(config.SAMPLE_RATE * 0.5)
    silence_chunks_needed = max(1, int(config.SILENCE_DURATION_S / 0.5))
    max_chunks           = int(config.MAX_RECORDING_S / 0.5)
    if open_window_s is None:
        open_window_s = float(_cfg("LISTEN_WINDOW_S", 4.0))
    # If no speech begins within this window, give up — so an auto-listen
    # follow-up (or a mis-fire) ends in a few seconds instead of recording 30s.
    initial_chunks       = max(1, int(max(0.5, float(open_window_s)) / 0.5))

    chunks:        list[np.ndarray] = []
    silence_count: int  = 0
    speech_started: bool = False

    _acquire_mic()
    try:
        for i in range(max_chunks):
            chunk = sd.rec(
                chunk_samples,
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device,       # None = system default; int = specific device index
                blocking=True,       # synchronous — mic is open only while recording
            )
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            chunks.append(chunk)

            if rms > config.SILENCE_THRESHOLD:
                speech_started = True
                silence_count  = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break
            elif i >= initial_chunks:
                break   # nobody started talking — stop waiting
    finally:
        _release_mic()

    if not chunks:
        return np.zeros(config.SAMPLE_RATE, dtype="float32"), False

    audio = np.concatenate(chunks).flatten()
    return audio, speech_started


# ─── Whisper STT ──────────────────────────────────────────────────────────────

_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        sizes = {"tiny": 75, "base": 145, "small": 460, "medium": 1500,
                 "tiny.en": 75, "base.en": 145, "small.en": 460, "medium.en": 1500}
        mb = sizes.get(config.WHISPER_MODEL_SIZE, "?")
        print(f"\n  Loading Whisper '{config.WHISPER_MODEL_SIZE}' on {config.WHISPER_DEVICE} "
              f"(first run downloads ~{mb} MB)…")
        try:
            _whisper_model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:
            # No STT means Bastion can hear the wake word and then understand
            # nothing. Model size/device only — both are config, not user data.
            log.error("Whisper failed to load: %s", exc)
            alerts.send_alert("voice.stt", type(exc).__name__,
                              f"Whisper '{config.WHISPER_MODEL_SIZE}' failed to load "
                              f"on {config.WHISPER_DEVICE}.")
            raise
        print("  Whisper ready.\n")
    return _whisper_model


# Prime Whisper with the proper nouns / commands it tends to mangle, so it stops
# inventing things like "Open-BS-Code" for "VS Code" or "Springfield B.A.".
_STT_PROMPT = (
    "Bastion. Open or close apps: Spotify, Chrome, Discord, Visual Studio Code, "
    "VS Code, Notepad, Calculator, File Explorer, Edge. Check the weather, system "
    "status, CPU, GPU, RAM, disk. Volume up, mute, play, pause, next track. "
    f"Places: {config.DEFAULT_LOCATION}, Pennsylvania, Philadelphia."
)


def transcribe(audio: np.ndarray) -> str:
    model = _get_whisper()
    try:
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,                           # skip truly silent segments
            vad_parameters={"min_silence_duration_ms": 400},
            initial_prompt=_STT_PROMPT,                # bias toward our vocabulary
        )
        # The generator is lazy — decoding actually happens here, so the join
        # stays inside the try or a mid-decode failure escapes unreported.
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        # Exception class only. Not the transcript, not the audio, and not its
        # duration — how long the owner spoke is still information about what he
        # said. A model label is config, so that much is safe to send.
        log.error("Transcription failed: %s", exc)
        alerts.send_alert("voice.stt", type(exc).__name__,
                          f"Whisper '{config.WHISPER_MODEL_SIZE}' transcription failed; "
                          f"speech was not understood.")
        raise


# ─── One voice turn ───────────────────────────────────────────────────────────

def _split_for_speech(response: str) -> str:
    """Whole-reply version of the voice/screen split, for the two paths that
    reach speak() with a finished string. Returns `response` unchanged — and
    shows nothing — when the reply is short enough to simply say."""
    if not _cfg("HUD_VOICE_SPLIT", True):
        return response
    send_reply = getattr(hud, "send_reply", None)
    if send_reply is None:
        return response
    try:
        spoken = send_reply(response, title="ANSWER")
    except Exception as exc:
        log.error("Voice/screen split failed (%s) — speaking the whole reply.", exc)
        return response
    if spoken != response:
        log.info("Voice/screen split: speaking %d of %d chars — the rest is on "
                 "the HUD.", len(spoken), len(response))
    return spoken or response


def voice_turn(tts, device: int | None, follow_up: bool = False) -> str:
    """
    Run one voice turn. Returns Bastion's spoken response ('' if nothing).

    `follow_up` marks a turn that the owner did not summon — the short listening
    window left open after a reply. It gets a shorter open-window and no wake
    cue, because nothing woke.
    """
    global _messages, _active_stream

    if follow_up:
        print("\n  ◌  Still listening…  (say nothing to end)\n")
    else:
        print("\n  ◉  Listening…  speak now,  silence to finish\n")
    hud.set_state("listening")
    hud.send_thinking("")   # wipe any reasoning left on the HUD from the last turn

    # Before the mic opens, never after: the cue's own tail would otherwise be
    # the first thing recorded, and a 988 Hz tone is well above SILENCE_THRESHOLD.
    play_cue("listen")

    window = float(_cfg("FOLLOW_UP_WINDOW_S", 2.5)) if follow_up else None
    audio, speech_found = record_until_silence(device, open_window_s=window)

    if not speech_found:
        print("  —  No speech detected. Ready.\n")
        hud.set_state("idle")
        return ""

    print("  ·  Transcribing…")
    hud.set_state("transcribing")
    text = transcribe(audio)

    if not text:
        print("  —  Could not make out any words. Ready.\n")
        hud.set_state("idle")
        return ""

    print(f"\nYou    > {textwrap.fill(text, width=68, subsequent_indent='         ')}")
    log.info("Voice input: %s", text[:200])
    hud.send_transcript("user", text)

    print("  ·  Thinking…")
    hud.set_state("thinking")

    # Sentence-level streaming: speak sentence one while the rest is still being
    # generated. Only available once bastion.chat() accepts on_content; without
    # it (or with TTS_STREAMING off) this falls through to the original
    # generate-everything-then-speak path, unchanged.
    #
    # The fallback below is correct behaviour and stays. But a silent fallback
    # is indistinguishable from success from the outside — audio plays, nothing
    # raises, and none of the latency win arrives. So the decision is logged
    # every turn, with the reason, and the outcome is logged after.
    stream = None
    split = None
    if not _cfg("TTS_STREAMING", True):
        log.info("Streaming TTS: OFF (config.TTS_STREAMING is false) — whole-reply playback.")
    elif not _chat_supports_streaming():
        log.warning("Streaming TTS: UNAVAILABLE — bastion.chat() does not accept "
                    "on_content, so there is nothing to stream from. Falling back "
                    "to whole-reply playback.")
    else:
        log.info("Streaming TTS: ON — speaking sentence by sentence.")
        t_stream0 = time.perf_counter()

        # Tracked separately from stream.started, which is no longer the same
        # question: the wait acknowledgement starts the stream without a single
        # content delta having arrived. This is what "NO DELTAS" means.
        saw_content = threading.Event()

        def _first_audio():
            # A wait acknowledgement makes sound while the turn is still
            # thinking, so "there is audio" no longer implies "it is answering".
            # The rail must not read SPEAKING for the next 90 seconds of tool
            # calls; _content() moves it to ANSWERING when the reply starts.
            if saw_content.is_set():
                hud.set_state("speaking")
            _mark_speaking_started()   # barge-in is live from the first sample

        stream = SpeechStream(tts, on_first_audio=_first_audio)
        with _speaking_lock:
            _active_stream = stream
        if _cfg("HUD_VOICE_SPLIT", True) and hasattr(hud, "send_detail"):
            split = _VoiceSplit(stream.feed)

        # The HUD's turn rail wants to know when generation stops being thinking
        # and starts being an answer. The retraction guard is load-bearing: a
        # tool-call preamble is streamed and then withdrawn (on_content(None)),
        # and bastion.log has caught that happening for real — without it the rail
        # would sit on ANSWERING for the rest of the turn.
        answering = []

        # Feeds the stream DIRECTLY, past _VoiceSplit: an ack is not part of the
        # reply and must not count against the split's character budget or be
        # mistaken for a lead-in.
        wait_ack = _WaitAck(stream.feed)

        def _content(delta):
            if delta is None:
                answering.clear()
                hud.set_state("thinking")
            elif not answering:
                answering.append(1)
                hud.set_state("answering")
            if delta:
                if not saw_content.is_set():
                    saw_content.set()
                    wait_ack.cancel()
            return split.feed(delta) if split is not None else stream.feed(delta)

        wait_ack.start()

    try:
        # on_thinking streams the model's live reasoning to the HUD ("watch it think").
        if stream is not None:
            response, _messages = chat(_messages, text, on_thinking=hud.send_thinking,
                                       on_content=_content)
        else:
            response, _messages = chat(_messages, text, on_thinking=hud.send_thinking)
    except Exception:
        if stream is not None:
            wait_ack.cancel(join=False)
            stream.stop()
            _mark_speaking_ended()
            with _speaking_lock:
                _active_stream = None
        raise
    finally:
        if stream is not None:
            # Generation is over either way: nothing may say "one moment" now.
            wait_ack.cancel()

    if split is not None:
        # Flush the withheld tail (and the handoff line) before finish() closes
        # the stream, or the last sentence never gets spoken.
        split.close()

    _messages = trim_history(_messages)

    log.info("Bastion: %s", response[:200])
    print(f"\nBastion > {textwrap.fill(response, width=68, subsequent_indent='         ')}\n")
    hud.send_transcript("bastion", response)

    if split is not None and split.truncated:
        # Streaming already stopped speaking at the right place; the screen half
        # is published here, once the whole reply exists.
        hud.send_detail(response, title="ANSWER")
        log.info("Voice/screen split: spoke %d of %d chars (%s) — the rest is on "
                 "the HUD.", split.spoken_chars, len(response), split.reason)

    hud.set_state("speaking")
    if stream is None:
        # Non-streaming fallback: hud.send_reply() does the same job on the
        # finished string, which is all this path needs.
        _speak_interruptible(tts, _split_for_speech(response))
    else:
        try:
            leftover = stream.finish()
        finally:
            _mark_speaking_ended()
            with _speaking_lock:
                _active_stream = None
        if _barged.is_set():
            # Cut off on purpose. Say so rather than logging OK, which would
            # claim a complete reply was spoken when most of it never was.
            log.info("Streaming TTS: BARGE-IN — %d chunk(s) spoken before the owner "
                     "interrupted; the rest was dropped.", len(stream.marks))
        elif not saw_content.is_set():
            # Nothing ever reached the streamer (no content deltas) — speak it
            # the old way rather than saying nothing at all. Tested on
            # saw_content, not stream.started: a wait acknowledgement starts the
            # stream, and testing started() would make an acked turn skip the
            # fallback and say nothing but "one moment, sir".
            log.warning("Streaming TTS: NO DELTAS — chat() accepted on_content but "
                        "never called it. Whole-reply playback%s.",
                        " (after %d wait ack(s))" % wait_ack.spoken
                        if wait_ack.spoken else "")
            _speak_interruptible(tts, _split_for_speech(response))
        elif leftover:
            log.warning("Streaming TTS: DEGRADED after %d chunk(s) — speaking the "
                        "remaining %d chars the blocking way.",
                        len(stream.marks), len(leftover))
            _speak_interruptible(tts, leftover)
        elif (len(stream.marks) - wait_ack.spoken) <= 0 and not (
                split is not None and split.truncated):
            # Deltas arrived, nothing was held back for the screen, and yet no
            # reply audio was ever played. The old code caught this case by
            # accident (`not stream.started`); saw_content no longer does, so it
            # is caught deliberately. Silence is the worst outcome available
            # here — speak it rather than logging OK over a mute turn.
            log.warning("Streaming TTS: NO AUDIO — content arrived but no chunk "
                        "was played. Whole-reply playback.")
            _speak_interruptible(tts, _split_for_speech(response))
        else:
            first = (stream.first_audio_at - t_stream0) if stream.first_audio_at else -1
            log.info("Streaming TTS: OK — %d chunk(s), first audio %.2fs after the "
                     "request.", len(stream.marks), first)
    hud.set_state("idle")
    return response


# ─── Hotkey callback (runs on pynput's listener thread) ──────────────────────

def _hud_queue(action: str, *args) -> None:
    """Call hud_server's queue API if this build has it, and never raise.

    `queue_enter`/`queue_leave`/`clear_queue` are new (2026-08-17), so a Bastion
    running against an older hud_server does not have them. This is called from
    the wake path, which is the one path in this file that must never throw: a
    summons lost to an AttributeError is a far worse failure than a HUD band
    that does not appear."""
    fn = getattr(hud, action, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception as exc:
        log.debug("HUD queue call %s failed: %s", action, exc)


def _prewarm_backend() -> None:
    """Start the answering backend's process NOW, while the owner is still talking.

    A summons — wake word, hotkey or barge-in — is the earliest moment Bastion
    knows a turn is coming, and it then spends ~3.1 s median (measured over 12
    real turns, 2026-08-18) capturing and transcribing before the model is
    called at all. A Claude subprocess costs ~1.4 s to launch, and the FIRST one
    after startup cost 4.75 s measured. Started here, that launch overlaps the
    user speaking and he never experiences it.

    Three properties this must have, in order of how badly they'd hurt:

    * **It must never raise.** This is called from the wake path, the one path
      in this file that must never throw — the same reason _hud_queue() exists.
      A summons lost to an AttributeError is far worse than a cold launch.
    * **It must never block.** The whole point is overlap with capture; run
      inline before recording starts and it has achieved exactly nothing. So it
      goes on a daemon thread and the caller returns immediately, regardless of
      what prewarm() itself does.
    * **It must not fire on the local backend.** Spawning a Claude process on
      every wake word when qwen3 is answering is pure waste on a machine whose
      VRAM and CPU are already the binding constraint.

    Everything is resolved lazily and defensively, so a tree without backends.py
    — or with one that predates prewarm() — still runs.

    The honest cost: a FALSE wake trigger now also launches a process. It is
    idempotent and short-lived, but it is not free."""
    if not _cfg("PREWARM_BACKEND", True):
        return
    try:
        threading.Thread(target=_prewarm_worker, name="bastion-prewarm",
                         daemon=True).start()
    except Exception as exc:                                # pragma: no cover
        log.debug("Backend pre-warm could not start: %s", exc)


def _prewarm_worker() -> None:
    """The pre-warm itself, off the wake path's thread. Swallows everything."""
    try:
        import backends
    except Exception as exc:
        log.debug("Backend pre-warm skipped — no backends module (%s).", exc)
        return
    try:
        name = backends.active_name()
        # "ollama" is the local model; "local" is accepted too in case the
        # backend names ever change under us.
        if name in (getattr(backends, "OLLAMA", "ollama"), "local", ""):
            return
        fn = getattr(backends, "prewarm", None)
        if fn is None:
            log.debug("Backend pre-warm skipped — %s has no prewarm().", name)
            return
        t0 = time.perf_counter()
        fn()
        log.info("Backend pre-warm (%s) returned in %.0f ms.",
                 name, (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        log.debug("Backend pre-warm failed: %s", exc)


def _queue_wake(reason: str) -> None:
    """Remember a summons heard mid-turn, and show it as waiting on the HUD."""
    if not _queued_wake.is_set():
        _queued_wake.set()
        _hud_queue("queue_enter", "")
        log.info("Summons queued (%s) — it will run when this turn ends.", reason)
    # The turn may have ENDED between the caller's _busy.locked() check and the
    # set above. The release block would then never see this flag, and it would
    # instead fire at the end of the NEXT turn — a reply arriving a turn late,
    # which is worse than the drop this replaces. So re-check and consume it
    # here if nobody is left to.
    if not _busy.locked():
        _consume_queued_wake()


def _consume_queued_wake() -> bool:
    """Turn a queued summons into the next turn. True if one was waiting.

    Safe to call from two threads at once: `_trigger` is an Event rather than a
    counter, so even if both get past the check, the result is one turn."""
    if not _queued_wake.is_set():
        return False
    _queued_wake.clear()
    _hud_queue("queue_leave")
    _prewarm_backend()
    _trigger.set()
    return True


def _on_hotkey() -> None:
    # Pressed while Bastion is talking: that is a barge-in. No guard window and no
    # threshold — a deliberate key press is never a false positive, which is also
    # what makes the hotkey the reliable way to shut it up if the wake word turns
    # out to trigger on Bastion's own voice.
    if _speaking.is_set() and _cfg("BARGE_IN_ENABLED", True):
        _barge_in("hotkey")
        return
    # Otherwise only set the trigger if we're not already mid-turn.
    # _busy.locked() is a non-blocking check — avoids queuing a second activation.
    if not _busy.locked():
        # Before the cue, not after: play_cue() blocks for ~0.2 s on purpose, and
        # the pre-warm may as well have those milliseconds too. The hotkey is the
        # documented backup for the wake word and starts a turn exactly the same
        # way, so it gets the same head start on the backend.
        _prewarm_backend()
        play_cue("wake")
        _trigger.set()
    else:
        # Mid-turn. A deliberate key press is never a false positive, so it is
        # acknowledged with the cue he already expects and remembered, rather
        # than swallowed — pressing the hotkey and getting nothing at all is
        # indistinguishable from a dead hotkey.
        play_cue("wake")
        _queue_wake("hotkey")


# ─── Wake word ("Bastion") ─────────────────────────────────────────────────────

_stop_listening = threading.Event()


def _wake_word_listener() -> None:
    """
    Always-on LOCAL wake-word detection. Streams 16 kHz mono mic audio to
    openWakeWord; when "Bastion" is heard it fires the same _trigger the hotkey
    uses. Audio is processed entirely on-device and never leaves the machine.

    The mic is shared with record_until_silence, so this pauses whenever a turn
    is running (_busy locked) or pending (_trigger set) and resumes afterward —
    that hand-off avoids two streams fighting over the device.
    """
    try:
        import openwakeword
        from openwakeword.model import Model
        model_path = os.path.join(
            os.path.dirname(openwakeword.__file__),
            "resources", "models", "hey_jarvis_v0.1.onnx",
        )
        oww = Model(wakeword_models=[model_path], inference_framework="onnx")
    except Exception as exc:
        log.error("Wake word disabled — failed to load model: %s", exc)
        # Bastion now looks completely normal but cannot hear its own name. The
        # hotkey still works, which is exactly why this needs saying out loud.
        alerts.send_alert("voice.wakeword", type(exc).__name__,
                          "Wake-word model failed to load; Bastion will not respond to "
                          "'Bastion'. The hotkey still works.")
        return

    block = 1280  # 80 ms at 16 kHz — openWakeWord's expected frame size
    fails = 0     # consecutive audio failures, for backoff
    log.info("Wake word listener ready (say 'Bastion').")

    try:
        _wake_word_loop(oww, block, fails)
    except BaseException as exc:
        # This thread dying is the quietest failure in the whole pipeline: the
        # HUD sits at idle, nothing is logged by anyone else, and Bastion is deaf
        # until the owner walks over and says its name to no effect.
        log.error("Wake word listener died: %s", exc, exc_info=True)
        alerts.send_alert("voice.wakeword", type(exc).__name__,
                          "Wake-word listener thread stopped; Bastion is deaf until "
                          "restart. The hotkey still works.")
        raise
    if not _stop_listening.is_set():
        # Fell out of the loop without being asked to stop — same deafness, no
        # exception to report.
        alerts.send_alert("voice.wakeword", "ListenerExited",
                          "Wake-word listener exited unexpectedly; Bastion is deaf until "
                          "restart. The hotkey still works.")


def _barge_in(how: str) -> None:
    """
    Cut Bastion off mid-sentence and hand the mic straight back to the owner.

    The machinery this calls (stop_speaking, SpeechStream._cancel, the slice-wise
    writes in _write) was written months ago and, until now, was reachable only
    from two error handlers — nothing let the owner talk over a reply. This is the
    caller that was missing.
    """
    t0 = time.perf_counter()
    _barged.set()
    stop_speaking()
    _mark_speaking_ended()
    log.info("BARGE-IN (%s): playback stopped in %.0f ms.",
             how, (time.perf_counter() - t0) * 1000.0)
    _prewarm_backend()
    play_cue("wake")
    _trigger.set()


def _wake_threshold() -> float:
    """
    How confident the wake word must be right now.

    Higher while Bastion is speaking, because that is the one moment its own voice
    is coming out of the speakers and back into the Yeti.

    Measured, not assumed — bench_selfhear.py, Kokoro bm_george straight into
    openWakeWord: "Hey Bastion" scores 1.00 (the positive control, so the probe
    works), but Bastion talking ABOUT itself peaks at 0.31 and an ordinary reply
    at 0.00. The model wants "hey" before the name, and Bastion never says that.
    0.6 sits above every self-reference measured and below the 1.00 control.

    That path has no room, no speaker colouration and no volume knob, so it
    bounds the risk rather than eliminating it. If it ever does loop,
    BARGE_IN_THRESHOLD is the dial and BARGE_IN_ENABLED = False is the off
    switch — and the hotkey barge-in is unaffected either way.
    """
    if _speaking.is_set():
        return float(_cfg("BARGE_IN_THRESHOLD", 0.6))
    return config.WAKEWORD_THRESHOLD


def _handle_wake(oww, score: float) -> bool:
    """Act on a detection. Returns True when the listener should release the mic."""
    if _speaking.is_set():
        if not _cfg("BARGE_IN_ENABLED", True):
            return False
        since = time.perf_counter() - _speaking_since
        guard = float(_cfg("BARGE_IN_GUARD_S", 0.7))
        if since < guard:
            # The opening moments are when a false trigger is likeliest (the tail
            # of the wake word that started this very turn is still in the ring
            # buffer) and least useful — nobody interrupts a reply before it has
            # said a word.
            log.debug("Barge-in ignored %.2fs into speech (score %.2f).", since, score)
            return False
        log.info("Wake word during playback (%.2f, %.2fs in) — barging in.",
                 score, since)
        oww.reset()
        _barge_in("wake word")
        return True

    if _busy.locked():
        # A turn is running but nothing is audible yet: it is thinking. There is
        # no audio to cut, and chat() has no cancel — interrupting here would
        # mean throwing away a reply that is seconds from arriving.
        #
        # It is no longer DROPPED, though (2026-08-18, the owner's decision): the
        # summons is remembered and run when the turn ends, and the HUD shows it
        # waiting meanwhile. The turn in flight still finishes untouched — this
        # queues, it does not interrupt.
        log.info("Wake word mid-turn with no audio playing (%.2f) — queued.", score)
        oww.reset()
        _queue_wake("wake word")
        return False

    log.info("Wake word detected (%.2f).", score)
    oww.reset()
    _prewarm_backend()
    play_cue("wake")
    _trigger.set()
    return True


def _wake_word_loop(oww, block: int, fails: int) -> None:
    """The listener's steady-state loop, split out so its caller can alert if it
    ever returns or raises.

    Changed 2026-08-17: the gate is _mic_busy (the recorder actually holding the
    device) rather than _busy (the whole turn). The listener now stays live while
    Bastion thinks and speaks, which is what makes barge-in possible at all."""
    while not _stop_listening.is_set():
        # While the recorder owns the mic — or a turn is pending and about to
        # take it — stay off the device.
        if _mic_busy.is_set() or _trigger.is_set():
            oww.reset()
            time.sleep(0.15)
            continue
        try:
            dev = _resolve_input_device()
            _mic_free.clear()
            try:
                with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                    dtype="int16", device=dev, blocksize=block) as stream:
                    while (not _stop_listening.is_set()
                           and not _mic_busy.is_set() and not _trigger.is_set()):
                        frame, _ = stream.read(block)
                        fails = 0   # a healthy read clears the failure backoff
                        scores = oww.predict(frame[:, 0])
                        score = max(scores.values()) if scores else 0.0
                        if score >= _wake_threshold() and _handle_wake(oww, score):
                            break   # release the stream so the turn can take the mic
            finally:
                _mic_free.set()
        except Exception as exc:
            # Mic briefly unavailable (driver hiccup, unplug, another app grabbed
            # it). Back off instead of hammering — sleep grows with consecutive
            # failures (capped), and we log sparsely to avoid flooding.
            fails += 1
            if fails == 1 or fails % 20 == 0:
                log.warning("Wake word audio unavailable (x%d): %s", fails, exc)
            if fails == 5:
                # Five in a row (~15s of backoff) is past a driver hiccup — the
                # mic is gone and the wake word has stopped working. Alerting on
                # the count, not every failure, keeps a knocked cable to one
                # message. The exception text can name a device, not user data.
                alerts.send_alert("voice.wakeword", type(exc).__name__,
                                  "Microphone unavailable for 5 consecutive attempts; "
                                  "wake word is not listening.")
            if isinstance(exc, sd.PortAudioError):
                _reset_audio()
            time.sleep(min(1.0 * fails, 8.0))


# ─── Audio recovery ───────────────────────────────────────────────────────────

def _reset_audio() -> None:
    """
    Force PortAudio to re-enumerate audio devices.

    PortAudio caches its device list at init and never refreshes it, so if the
    mic is unplugged and plugged back in (a knocked Yeti cable), the stale list
    keeps causing 'device -1' errors. Re-initializing makes the next
    _resolve_input_device() see the device again — no full restart needed.
    """
    try:
        sd._terminate()
        sd._initialize()
        log.info("Audio subsystem re-initialized (device list refreshed).")
    except Exception as exc:
        # The recovery path itself is broken, so a re-plugged mic will no longer
        # be picked up — this one needs a restart, and the owner should hear that now.
        log.error("Audio re-init failed: %s", exc)
        alerts.send_alert("voice.audio", type(exc).__name__,
                          "PortAudio re-init failed; a re-plugged mic will not be "
                          "detected without a restart.")


# ─── Proactive startup briefing ─────────────────────────────────────────────

def _startup_briefing(tts) -> None:
    """Speak a short greeting with time, weather, and status when Bastion starts."""
    if not getattr(config, "STARTUP_BRIEFING", True):
        return
    try:
        now  = datetime.now()
        part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
        h12  = now.hour % 12 or 12
        clock = f"{h12}:{now.minute:02d}"

        bits = [f"Good {part}, sir.", f"It's {clock}."]
        try:
            wx = json.loads(T.get_weather(config.DEFAULT_LOCATION))
            if "temp_f" in wx:
                city = str(wx.get("location", "")).split(",")[0] or "your area"
                bits.append(f"It's {round(wx['temp_f'])} degrees and "
                            f"{wx.get('condition', 'clear')} in {city}.")
        except Exception as exc:
            log.debug("Briefing weather skipped: %s", exc)
        bits.append("All systems nominal.")

        text = " ".join(bits)
        # Routed through the unprompted queue rather than calling speak()
        # directly: it is the same kind of speech, and going through one path
        # means the mute switch and the barge-in bookkeeping apply to it too.
        speak_unprompted(text, source="briefing")
    except Exception as exc:
        log.error("Startup briefing failed: %s", exc)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    global _messages

    if not check_ollama():
        sys.exit(1)

    T.init_db()

    # Start the HUD WebSocket bridge so the wallpaper can show live state
    hud.start()

    # State which brain is live, straight away. The band is a PRIVACY indicator
    # — "is my speech leaving this machine" — and it only ever published on a
    # CHANGE, so a session that starts local and never toggles showed nothing at
    # all. A blank space cannot be read as an assurance: it means "local", "the
    # HUD died" and "nothing published yet" identically. Wrapped because a
    # cosmetic band is never worth failing startup for, and lazily imported so a
    # tree without backends.py still runs.
    try:
        import backends
        backends.announce()
    except Exception as exc:
        log.debug("Could not announce the active backend to the HUD: %s", exc)

    # Resolve mic once at startup — re-checks on each turn if Yeti was unplugged
    device = _resolve_input_device()
    mic_label = (
        f"[{device}] {sd.query_devices(device)['name']}"
        if device is not None
        else f"system default ('{config.AUDIO_DEVICE_NAME}' not found)"
    )

    # Init TTS before Whisper so any voice-selection errors surface fast
    global _tts_engine
    tts = _build_tts()
    _tts_engine = tts          # the unprompted worker speaks through this
    _ensure_unprompted_worker()

    # Load Whisper — downloads model on first run, then cached in ~/.cache/huggingface
    _get_whisper()

    # Warm the OCR engine in the background so the first screen-read is snappy.
    threading.Thread(target=T.preload_ocr, daemon=True, name="ocr-preload").start()

    _messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    model_label = config.MODEL.split(":")[0]
    key_label   = config.HOTKEY.replace("<", "").replace(">", "").replace("+", " + ")
    print(f"""
╔══════════════════════════════════════════════════════╗
║  Bastion  ·  VOICE MODE  ·  {model_label:<23} ║
║  Hotkey : {key_label:<43} ║
║  Whisper: {config.WHISPER_MODEL_SIZE:<5} on {config.WHISPER_DEVICE:<37} ║
║  Mic is OFF between turns — no always-on listening. ║
╚══════════════════════════════════════════════════════╝
  Mic    : {mic_label}

Waiting…  Say "Bastion"  or press  {key_label} .  Ctrl+C to exit.
""")

    # Register global hotkey (works from any window, no admin required)
    hotkey_listener = kb.GlobalHotKeys({config.HOTKEY: _on_hotkey})
    try:
        hotkey_listener.start()
    except Exception as exc:
        # The hotkey is the fallback for a dead wake word. If it fails too there
        # is no way left to reach Bastion, so this is worth a message on its own.
        log.error("Hotkey listener failed to start: %s", exc)
        alerts.send_alert("voice.hotkey", type(exc).__name__,
                          f"Hotkey {config.HOTKEY} failed to register; only the wake "
                          f"word can start a turn.")

    # Start always-on local wake-word listener ("Bastion"). Guarded: if it fails
    # to start, the hotkey still works.
    if getattr(config, "WAKEWORD_ENABLED", False):
        threading.Thread(target=_wake_word_listener, daemon=True, name="wakeword").start()

    # Clear any stale HUD state (e.g. a 'listening' left over from a prior crash
    # or mid-turn unplug) so the wallpaper starts clean.
    hud.set_state("idle")

    # Proactive greeting so Bastion speaks first when you sit down.
    _startup_briefing(tts)

    follow_up = False   # is the next turn an auto-opened window, or a summons?
    chain     = 0       # consecutive follow-up turns, capped below
    try:
        while True:
            _trigger.wait()     # sleep until hotkey fires
            _trigger.clear()

            # Re-resolve device each turn so plugging in the Yeti mid-session works
            device = _resolve_input_device()

            response = ""
            _barged.clear()
            with _busy:         # lock prevents a second hotkey from re-entering
                try:
                    response = voice_turn(tts, device, follow_up=follow_up)
                except sd.PortAudioError as exc:
                    stop_speaking()   # never leave a queued reply half-playing
                    _mark_speaking_ended()
                    log.error("Audio device error: %s", exc)
                    alerts.send_alert("voice.turn", "PortAudioError",
                                      "Mic error during a voice turn; the turn was lost "
                                      "and the device list was refreshed.")
                    print(f"  Audio error: {exc}")
                    print("  Mic issue — recovering. Re-plug it if needed, then press the hotkey again.\n")
                    _reset_audio()   # refresh device list so a re-plugged mic is found next turn
                except Exception as exc:
                    stop_speaking()
                    _mark_speaking_ended()
                    log.error("Voice turn error: %s", exc, exc_info=True)
                    # Class name only — the exception message on this path can
                    # quote what was said or what the model replied.
                    alerts.send_alert("voice.turn", type(exc).__name__,
                                      "Voice turn failed; Bastion did not answer.")
                    print(f"  Unexpected error: {exc}\n")
                finally:
                    # Never leave the HUD frozen on listening/thinking/speaking.
                    hud.set_state("idle")

            # A summons heard while that turn was running is run now. It is a
            # FRESH turn, not a follow-up: the block below sees _trigger already
            # set and zeroes the chain, which is what we want — he interrupted
            # to ask something new, so the auto-listening window is not owed.
            _consume_queued_wake()

            # Stay in the conversation: leave a short listening window open after
            # EVERY reply, not only after a question. It ends on silence, so the
            # cost of being wrong is FOLLOW_UP_WINDOW_S of an open mic and one
            # quiet cue — and the alternative is saying "Bastion" before every
            # sentence of a back-and-forth.
            #
            # Three things end the chain rather than extending it: silence (the
            # turn returns ''), a barge-in (the owner is already talking, the trigger
            # is set for him), and FOLLOW_UP_MAX_TURNS. The cap exists because a
            # television in the room can hold the window open indefinitely; after
            # it, the wake word is required again.
            follow_up = False
            if _stop_listening.is_set() or _barged.is_set() or _trigger.is_set():
                chain = 0
            elif not (isinstance(response, str) and response.strip()):
                chain = 0       # nothing was said, or nothing was heard
            elif not _cfg("FOLLOW_UP_ALWAYS", True):
                # Legacy behaviour, kept behind the flag: question marks only.
                chain = 0
                if (getattr(config, "FOLLOW_UP_ON_QUESTION", False)
                        and response.rstrip().endswith("?")):
                    log.info("Follow-up: Bastion asked a question — auto-listening.")
                    follow_up = True
                    _trigger.set()
            elif chain >= int(_cfg("FOLLOW_UP_MAX_TURNS", 6)):
                log.info("Follow-up chain hit %d turns — wake word required again.", chain)
                chain = 0
            else:
                chain += 1
                log.info("Follow-up: listening for %.1fs (chain %d).",
                         float(_cfg("FOLLOW_UP_WINDOW_S", 2.5)), chain)
                follow_up = True
                _trigger.set()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # Ctrl+C above is the normal exit; anything reaching here has ended the
        # voice session for good. Re-raised so the traceback still prints.
        log.error("Voice session crashed: %s", exc, exc_info=True)
        alerts.send_alert("voice.main", type(exc).__name__,
                          "Voice session crashed; Bastion is no longer listening.")
        raise
    finally:
        _stop_listening.set()
        # A crash or Ctrl+C while something was waiting would otherwise leave
        # the HUD promising "you're next" for a Bastion that is gone.
        _queued_wake.clear()
        _hud_queue("clear_queue")
        try:
            hotkey_listener.stop()
        except Exception:
            # Never started, or already stopped — nothing to do on the way out.
            pass
        print("\nBastion offline.")
        log.info("Voice session ended.")


if __name__ == "__main__":
    main()
