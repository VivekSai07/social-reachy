# Reachy Mini conversation template — architecture reference

Condensed from direct investigation of the reference repo the
`reachy-mini-app-assistant create --template conversation` scaffold forks
from: [`pollen-robotics/reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app)
(the `reachy_mini_conversation_demo` name mentioned in the docs 301-redirects
here — same repo). Kept local for the same reason as
`reachy_mini_notes.md`: so phase 2 ("AI brain," see `apps/social_app/plan.md`)
doesn't need to re-derive this, including on a different machine.

## Backend: Hugging Face Realtime, not a generic chat API

This is **not** built on OpenAI/Anthropic chat-completions or Ollama. It's
built on Hugging Face's hosted **Realtime speech-to-speech** backend — a
websocket protocol shaped like OpenAI's Realtime API (`session.update`,
`conversation.item.create`). STT, LLM, and TTS are bundled server-side into
one streaming session; the app doesn't manage a message list itself — turn
state lives in the realtime session.

Config (`.env.example`): `REALTIME_TRANSCRIPTION_LANGUAGE`,
`HF_REALTIME_CONNECTION_MODE` (`deployed` = HF-hosted, no API key needed;
`local` = point `HF_REALTIME_WS_URL` at a self-hosted `speech-to-speech`
server), `HF_TOKEN`, `REACHY_MINI_APP_TIMEOUT_MINUTES`.

Implementation: `src/reachy_mini_conversation_app/huggingface_realtime.py`.

## Persona / system prompt

Plain text file: `profiles/default/instructions.txt`. Sections: IDENTITY /
CRITICAL RESPONSE RULES / CORE TRAITS / RESPONSE EXAMPLES / BEHAVIOR RULES /
TOOL & MOVEMENT RULES. Enforces short (1-2 sentence) replies, "warm,
efficient, light humor, no sarcasm." This is already close to the "friendly
buddy" tone wanted for phase 2 — likely needs rewording, not new
infrastructure.

14 alternate personas exist as sibling folders under `profiles/` (e.g.
`mad_scientist_assistant`, `victorian_butler`) — each just its own
`instructions.txt` / `voice.txt` / `greeting.txt`. Loaded by
`get_session_instructions()` in `src/reachy_mini_conversation_app/prompts.py`.

## Existing memory system (fact-store, no timestamps)

- `src/reachy_mini_conversation_app/memory.py` — JSON file at
  `~/.local/share/reachy_mini_conversation_app/memory.v1.json` (XDG path,
  already outside any git repo — a privacy default worth keeping). One
  `MemoryFact{id, text, created_at}` per fact, deduplicated
  case-insensitively, capped at 60 facts / 280 chars each, atomic
  tmp-file writes guarded by a `threading.Lock`.
- `tools/remember.py` / `tools/forget.py` — LLM-callable tools; the model
  itself decides what's worth persisting (save one atomic third-person
  fact) or removing (by substring match). No summarization, no embeddings.
- **Injection point** — where any new memory/context should plug in —
  is `get_session_instructions()` in `prompts.py`:
  ```python
  memory_prompt = format_memory_for_prompt(instance_path)
  if memory_prompt:
      return f"{memory_prompt}\n\n{instructions}"
  return instructions
  ```
  Remembered facts are prepended to the persona instructions as a bullet
  list every session start ("Things you remember about the user... do not
  recite the list verbatim").
- **What's missing**: `created_at` isn't surfaced to the prompt, and
  there's no session-level log at all — no "you last spoke 2 days ago," no
  per-conversation summary. This is the actual gap phase 2 needs to fill
  (see `apps/social_app/plan.md`'s Phase 2 section for the planned
  `sessions.db` extension) — a memory system already exists, temporal
  awareness on top of it doesn't yet.

## `tools/` — the extension pattern

Directory: `src/reachy_mini_conversation_app/tools/`. Base `Tool` class +
`ToolDependencies` in `core_tools.py`; concrete tools subclass it. Existing
tools: `dance.py`, `stop_dance.py`, `play_emotion.py`, `stop_emotion.py`,
`camera.py`, `move_head.py`, `head_tracking.py`, `go_to_sleep.py`,
`idle_do_nothing.py`, `task_status.py` / `task_cancel.py`, `remember.py`,
`forget.py`. Custom tools resolve first from the active profile folder, then
this core library. Per the SDK's `ai-integration.md`: tool calls enqueue a
move/action; a separate control loop drains the queue each tick so motion
stays smooth regardless of LLM latency — the tool itself never touches
`set_target()` directly.

`tools/go_to_sleep.py` is the natural hook for "conversation is ending,
persist a session summary now" — it's already the code path that runs when
a conversation winds down.

## Not found in this repo (confirmed, not just unchecked)

No turn-by-turn transcript log, no summarization step, no non-HF LLM
provider path (no OpenAI/Anthropic key handling exists here — using a
different provider means rebuilding STT/LLM/TTS/tool-calling largely from
scratch, not a config change).

## Source

- `ai-integration.md` (upstream `reachy_mini` skills doc):
  https://raw.githubusercontent.com/pollen-robotics/reachy_mini/main/skills/ai-integration.md
- Reference app: https://github.com/pollen-robotics/reachy_mini_conversation_app
