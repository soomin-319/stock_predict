# P1 Kakao Job Store Design

## Context

`src/chatbot/kakao_colab_bot.py` still owns prediction job state, runtime process tracking,
JSON persistence, stale-running cleanup, and user-facing bot behavior. Formatter and session
state are already separated. The next low-risk P1 slice is to move only persistent job registry
state handling into a small store while keeping subprocess/thread orchestration in the bot.

## Goals

- Extract persistent Kakao prediction job state operations from `KakaoColabPredictionBot`.
- Preserve existing runtime behavior, user messages, cooldown checks, concurrency checks, and logs.
- Keep buy/sell/hold decisions based only on next-day `predicted_return`.
- Keep news/disclosures display-only; this change must not affect expected returns, rankings,
  recommendations, or signals.
- Preserve JSON load/save behavior, atomic writes, and secret redaction.

## Non-goals

- Do not extract subprocess launch, stream reading, monitor threads, or bootstrap worker logic.
- Do not change pipeline commands, result paths, cooldown values, concurrency rules, or output schema.
- Do not change recommendation, ranking, signal, news, or disclosure logic.

## Proposed architecture

Add `src/chatbot/job_store.py`:

- `PredictionJobState`: the existing dataclass moved from `kakao_colab_bot.py`.
- `ChatbotJobStore`: owns the job registry dictionary, its path, lock, and secret values.
- Methods:
  - `data`: compatibility access to the underlying registry.
  - `save()`: persist with existing redaction/atomic JSON behavior.
  - `get(symbol)`: read one job state.
  - `set(symbol, state)`: write one job state and save.
  - `running_prediction_count(bootstrap_key)`: count running non-bootstrap jobs.
  - `elapsed_seconds(job_state)`: preserve current completion-time parsing.
  - `mark_failed(symbol, exit_code, note="")`: current failed-state update.
  - `mark_completed(symbol, exit_code)`: current completed/failed terminal update used after process exit.
  - `mark_stale_running_on_startup()`: convert persisted running jobs to failed with
    `exit_code=-2`, `failure_note="stale_running_on_startup"`, and no command/pid.

`KakaoColabPredictionBot` will create `_job_store` during init and keep `_job_registry =
_job_store.data` as a temporary compatibility alias for existing tests and callers. Existing
bot wrappers such as `_mark_job_failed()` and `_job_elapsed_seconds()` will delegate to the store.

## Data flow

1. Init loads the persisted registry with existing `load_registry()`.
2. Bot constructs `ChatbotJobStore(path, registry, lock, secret_values)`.
3. Job start remains in bot:
   - check existing/running/concurrency/cooldown via store helpers;
   - launch subprocess and threads in bot;
   - write running `PredictionJobState` through the store.
4. Process completion remains in bot:
   - close log resources and run user-facing side effects in bot;
   - update terminal state through store.
5. Startup stale-running cleanup delegates the state mutation to store.

## Error handling

- JSON load/save semantics remain identical by reusing existing registry helpers from
  `session_store.py`.
- Terminal job states remove `command` and `pid` before persistence as they do today.
- Invalid timestamps in `elapsed_seconds()` still return `None`.
- Store methods hold the injected lock while reading/mutating/persisting registry state.

## Testing

- Add focused unit tests for `ChatbotJobStore`:
  - terminal failed/completed updates strip `command` and `pid`;
  - elapsed seconds handles valid, missing, and invalid timestamps;
  - stale running startup cleanup mutates only running jobs and persists changes;
  - save redacts secrets.
- Add bot delegation tests for `_mark_job_failed()`, `_job_elapsed_seconds()`, and startup cleanup.
- Keep existing job tests unchanged:
  - duplicate running job;
  - concurrency limit;
  - finalize process success/failure.
- Run impacted tests, sample pipeline smoke, then full `pytest`.

## Rollout

This is a stacked PR based on `p1-kakao-session-store`. It is a refactor-only slice. Runtime
process orchestration stays in `kakao_colab_bot.py` to avoid combining state extraction with
thread/subprocess behavior changes.
