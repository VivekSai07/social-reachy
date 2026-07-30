# plan.md — companion

Phase 2 of the project (see `apps/social_app/plan.md`): friendly conversation
with persistent, temporally-aware memory, layered on the Reachy Mini
`conversation` template. Full research/design history:
`docs/ai_brain_notes.md` and the plan-mode session that produced this app
(`C:\Users\Vivek Sai\.claude\plans\answers-to-your-questions-synchronous-dewdrop.md`).

## Decisions made before implementing

1. Backend: Hugging Face Realtime speech-to-speech (the template's default),
   not a different LLM provider.
2. Gaze mode: LLM-toggleable via a `gaze_tracking` tool, not always-on
   ambient.
3. Gaze priority: pauses during other moves (dance/emotion/goto), exactly
   like the template's existing `BreathingMove` idle behavior.
4. Lives in its own directory (`apps/companion/`), not retrofitted into
   `apps/social_app/` — phase 1 stays untouched and working.
5. Session summary source: derived from facts saved via the existing
   `remember` tool during that session (not a dedicated transcript-summary
   LLM call — no local transcript exists to summarize from; the realtime
   session's turn state lives server-side). Honest limitation: a rich
   conversation where nothing was explicitly remembered produces an empty
   recap.
6. Code layout: flat files in `src/companion/` (matching `social_app`'s
   convention), not a dedicated `gaze/` subpackage.

## What was built

- `perception.py`, `gaze.py` — ported verbatim from `apps/social_app/social_app/`
  (one import line fixed in `gaze.py`).
- `gaze_move.py` — `GazeMove(Move)`, modeled on the template's own
  `BreathingMove`: infinite duration, evicted by the same idle-management
  logic. Uses a long-lived, `MovementManager`-owned `GazeController`/
  `WebcamFaceTracker` (constructed once in `MovementManager.__init__`) —
  constructing fresh ones per activation would reset EMA smoothing state and
  break the stale-tracker timeout logic (see `gaze_move.py`'s docstring).
- `sessions.py` — SQLite (`sessions.v1.db`, stdlib `sqlite3`, same XDG
  location convention as the existing `memory.v1.json`): `start_session()`,
  `end_session()` (idempotent), `format_session_context_for_prompt()`.
- `moves.py` — `MovementManager` gained `set_gaze_tracking()` (mirrors the
  existing `set_head_tracking()` shape), and its idle-move selection now
  picks `GazeMove` instead of `BreathingMove` when gaze tracking is enabled.
  Toggling proactively evicts the current idle move (both have infinite
  duration, so without this the switch would sit inert until an unrelated
  real move happened to interrupt it).
- `tools/gaze_tracking.py` — new tool, replaces `tools/head_tracking.py`
  (deleted, along with its stale test — the daemon-side tracking it drove is
  a different mechanism, useless in sim for the same reason phase 1 exists).
- `prompts.py` — `get_session_instructions()` now also injects
  `format_session_context_for_prompt()` alongside the existing
  `format_memory_for_prompt()`, additively.
- `main.py` — starts a session record near the top of `run()`, ends it (with
  a fact-derived summary) inside `go_to_sleep_and_stop_app()`, and has a
  defensive idempotent fallback in the outer `finally:` for exit paths that
  bypass that (dashboard stop, crash).
- `profiles/_companion_locked_profile/instructions.txt` — persona ported
  from the template's real `profiles/default/instructions.txt` (warm,
  concise, light humor), extended with a "buddy" framing and a note on using
  session/memory context naturally rather than reciting it.

## Known upstream bugs found and worked around

`reachy-mini-app-assistant create --template conversation` was completely
broken before this work (fails outright) and needed four fixes, all applied
to the installed `.venv` (gitignored, not part of this repo — will need
reapplying after any `reachy-mini` upgrade until fixed upstream):

1. `fork_conversation.py` hardcoded cloning a `develop` branch that no
   longer exists upstream (`main` is now the default branch) — patched to
   `main`.
2. The post-clone `.git` cleanup (`shutil.rmtree`) fails on Windows because
   git marks object files read-only and `shutil.rmtree` doesn't clear that
   attribute — patched with an `onerror` handler.
3. File renaming during the fork (`_rename_package`) uses `Path.read_text()`/
   `write_text()` without an explicit encoding, which defaults to the
   Windows system codepage (cp1252) and crashes on a UTF-8 byte sequence in
   one of the cloned files — worked around by running with `PYTHONUTF8=1`
   rather than patching every call site.
4. **Structural bug, not just Windows-specific**: `_create_profile()` writes
   the locked profile to `src/<app_name>/profiles/<locked_profile>/`, but
   `config.py`'s `DEFAULT_PROFILES_DIRECTORY` resolves to the top-level
   `<app_name>/profiles/` for a source checkout — a real mismatch, not a
   platform quirk. `_cleanup()` also only prunes sibling profiles under the
   `src/` copy, leaving all 14 original template profiles orphaned at the
   top level. Fixed by hand for this app: moved the locked profile to the
   top-level `profiles/` directory (where `config.py` actually looks) and
   removed the stale sibling profiles from there.

## Quality review findings (fixed)

Three parallel code-reviewer passes (simplicity/DRY, bugs/correctness,
project-convention adherence) caught three real issues, all fixed:

1. **Critical**: `set_gaze_tracking`'s command handler called
   `WebcamFaceTracker.stop()` synchronously from `moves.py`'s real-time
   worker thread — the same thread that must call `set_target()` every
   ~16.7ms. `stop()` joins its capture thread (up to 2s), so toggling gaze
   off would freeze the robot mid-motion for up to 2 seconds. Fixed: the
   tracker is now only ever `.start()`ed on enable (idempotent) and never
   explicitly stopped on disable — it keeps running harmlessly in the
   background once started (nothing reads its output while disabled), and
   is only actually stopped in `MovementManager.stop()`, which runs outside
   the hot loop at real shutdown.
2. `sessions.py` had a dead, unused `session_has_ended()` function —
   removed (`end_session()`'s own `WHERE end_ts IS NULL` guard already made
   it idempotent without needing this).
3. `sessions.py` lacked `memory.py`'s graceful-degradation contract (a
   broken/locked `sessions.v1.db` would crash the app at startup instead of
   degrading). Fixed: `start_session()`, `end_session()`, and
   `format_session_context_for_prompt()` now catch `sqlite3.Error`/`OSError`
   and log-and-degrade, matching `memory.py`'s `_read_memory_file()` pattern.

Two new regression tests added to `tests/test_moves.py` covering the
gaze-selection/eviction logic these fixes touch (not previously covered by
the upstream test suite): `test_idle_move_is_gaze_when_gaze_tracking_enabled`,
`test_toggling_gaze_tracking_evicts_current_idle_move`.

## Open items for whenever this gets tested/extended

- Not yet run against the sim — needs a human on a machine with a
  microphone/speakers and a webcam (same as phase 1's verification).
- `reachy-mini` got upgraded 1.9.0 → 1.10.0rc2 in the shared `.venv` as a
  side effect of this app's `pyproject.toml` requiring it — verify phase 1
  (`apps/social_app`) still works under the new version before merging.
- No real transcript-based summary (see decision #5 above) — worth
  revisiting if `huggingface_realtime.py` turns out to expose a transcript
  hook and the fact-derived summary feels too thin in practice.
- The upstream CLI bugs above are only patched locally; if `reachy-mini`
  gets reinstalled/upgraded, check whether upstream has fixed
  `--template conversation` before re-patching.
