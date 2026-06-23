# P1 Kakao Session/Registry Store Design

## Context

`src/chatbot/kakao_colab_bot.py` still owns many responsibilities. The previous P1 PR extracted pure message formatting. The next small extraction should remove session and JSON registry persistence from the bot without changing chatbot behavior.

This is a maintenance-only change. It must not affect `predicted_return`, recommendations, rankings, signal scores, or news/disclosure display-only handling.

## Goals

- Extract JSON registry load/save and user session access into a focused helper module.
- Preserve the existing runtime file layout and legacy migration behavior.
- Keep the current JSON schema stable:
  - job registry: `dict[str, dict[str, Any]]`
  - session registry: `dict[str, dict[str, Any]]`
- Keep secret redaction and atomic writes.
- Keep bot compatibility helpers for tests and incremental follow-up PRs.

## Non-goals

- Do not refactor subprocess prediction-job execution.
- Do not refactor live news/disclosure context collection.
- Do not change Kakao response text, quick replies, cached prediction lookup, or pipeline outputs.
- Do not add new config keys or external dependencies.

## Proposed Architecture

Add `src/chatbot/session_store.py` with a small persistence layer:

- `load_registry(path: Path) -> dict[str, dict[str, Any]]`
  - Returns `{}` for missing files, invalid JSON, non-dict top-level JSON, or non-dict entries.
- `save_registry(path: Path, data: dict[str, dict[str, Any]], secret_values: Iterable[str]) -> None`
  - Redacts secrets using existing `redact_value`.
  - Writes with existing `atomic_write_text`.
- `ChatbotSessionStore`
  - Owns `session_path`, in-memory `registry`, and a lock supplied by the bot.
  - Provides:
    - `update(user_id, symbol, display_code, intent)`
    - `symbol_for(user_id)`
    - `intent_for(user_id)`
    - `data` property for compatibility/tests.

`KakaoColabPredictionBot` keeps the same public/private method names for now, but delegates:

- `_load_registry(...)` -> `load_registry(...)`
- `_save_registry(...)` -> `save_registry(...)`
- `_update_session(...)` -> `ChatbotSessionStore.update(...)`
- `_symbol_from_session(...)` -> `ChatbotSessionStore.symbol_for(...)`
- `_session_intent(...)` -> `ChatbotSessionStore.intent_for(...)`

The bot will continue to expose `_job_registry` and `_session_registry` so existing tests and follow-up PRs remain low-risk.

## Data Flow

Startup remains unchanged:

1. Resolve runtime paths.
2. Load job registry from runtime path or legacy path.
3. Load session registry from runtime path or legacy path.
4. Save migrated registries to runtime paths.
5. Create `ChatbotSessionStore` around the loaded session registry.

Runtime session updates:

1. Bot extracts `user_id`.
2. Bot determines symbol/intent.
3. Bot calls `_update_session(...)`.
4. The wrapper delegates to the store.
5. Store writes redacted JSON atomically.

## Error Handling

- Missing or malformed registry files remain non-fatal and produce empty registries.
- Non-dict registry entries are ignored, matching current behavior.
- Save uses existing atomic write helper.
- Locking remains controlled by the bot's existing `_state_lock`.

## Testing

Add deterministic tests for the new store:

- Missing/invalid/non-dict registries load as empty.
- Non-dict entries are filtered.
- Save redacts configured secrets.
- Session update persists symbol, display code, and intent.
- Session lookup returns stable defaults for missing users.

Update bot tests only where needed:

- Confirm bot session wrappers delegate to `ChatbotSessionStore`.
- Confirm default runtime path and legacy migration behavior remains unchanged.

Required verification:

- Impacted chatbot tests.
- `pytest tests/test_pipeline_smoke.py -q`.
- Sample pipeline:
  `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
- Full `pytest -q` before PR.

## Risks and Mitigations

- Risk: test coupling to `_session_registry`.
  - Mitigation: keep `_session_registry` as an alias to store data.
- Risk: subtle lock behavior change.
  - Mitigation: store uses the bot-provided lock; wrappers preserve call sites.
- Risk: accidental schema change.
  - Mitigation: add store tests around persisted JSON shape and keep dataclass fields unchanged.
