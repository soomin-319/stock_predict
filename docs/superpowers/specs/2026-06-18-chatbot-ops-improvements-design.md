# Chatbot Operations Improvements Design

Date: 2026-06-18

## Goal

Improve the Kakao/Colab chatbot in the order requested by the user:

1. Persist runtime state/logs under a configurable Colab Drive-friendly directory and clean stale running jobs on boot.
2. Add a Kakao rich list response for recommendation lists without changing recommendation logic.
3. Add optional webhook source IP/CIDR allowlisting for operations security.

The chatbot remains research and operations support only. Buy/sell/hold decisions must continue to use `predicted_return` and existing recommendation policy only. News, disclosures, and response presentation must not change expected returns, rankings, recommendations, or signals.

## Approach

Use a small, low-risk extension of the existing chatbot modules instead of a broad refactor. Keep `kakao_colab_bot.py` as the integration point, add response-format helpers in `responses.py`, and update deterministic tests in `tests/test_chatbot_helpers.py`.

## 1. Runtime Directory and Stale Job Cleanup

Add `runtime_dir` to `PipelineRuntimeConfig`, defaulting to `result/runtime`. Expose it through `--runtime-dir` and `CHATBOT_RUNTIME_DIR`.

When explicit `state_path` or `session_path` are not supplied, derive these paths from `runtime_dir`:

- `chatbot_jobs.json`
- `chatbot_sessions.json`
- `prewarm_cache_meta.json`
- `logs/`

This lets Colab users point runtime files at a Google Drive mount while preserving current defaults.

On bot startup, inspect loaded job state. Any job with `status == "running"` but no active local process is stale after restart, so mark it failed with `exit_code = -2`, `completed_at = now`, and `note = "stale_after_restart"`. This avoids indefinite running states after Colab session loss.

Keep the existing legacy migration from `result/chatbot_jobs.json` and `result/chatbot_sessions.json`.

## 2. Recommendation Rich Format

Add a Kakao `listCard` response helper in `src/chatbot/responses.py`. For recommendation requests, keep the same recommendation service, ordering, filters, and text formatter. Only change display format when a non-empty recommendation list is available.

Recommended response behavior:

- Use `listCard` for multiple recommendation items.
- Include rank, stock name, symbol, and final score if available.
- Keep quick replies such as `다시 추천` and `도움말`.
- Fall back to existing `simpleText` response if recommendation data is empty, malformed, or rich formatting fails.

This preserves the guardrail that presentation must not influence recommendations.

## 3. Webhook Source IP/CIDR Allowlist

Add `allowed_webhook_cidrs` to `PipelineRuntimeConfig`. Expose it through `--allowed-webhook-cidrs` and `KAKAO_ALLOWED_WEBHOOK_CIDRS` as comma-separated CIDR/IP entries.

In `/kakao/webhook`, apply checks in this order:

1. If `allowed_webhook_cidrs` is configured, verify `request.remote_addr` belongs to at least one configured network.
2. If `kakao_webhook_secret` is configured, verify `X-Webhook-Secret` using constant-time comparison.
3. Process the Kakao payload.

If the allowlist is unset, keep current behavior. Do not trust `X-Forwarded-For` in this change, because trusting proxy headers safely requires explicit proxy configuration. Kakao request signature verification is not added in this pass because its current contract must be verified against external documentation before implementation.

## Testing

Add deterministic tests for:

- `runtime_dir` path derivation.
- startup cleanup of stale `running` jobs.
- `listCard` response shape and quick replies.
- recommendation rich response fallback safety.
- webhook CIDR allow and deny behavior.

Run at minimum:

```bash
pytest tests/test_chatbot_helpers.py
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

## Non-Goals

- Do not change prediction, ranking, scoring, or buy/sell/hold logic.
- Do not let news/disclosures influence expected returns or signals.
- Do not refactor the whole chatbot server.
- Do not trust forwarded IP headers without a dedicated trusted-proxy design.
