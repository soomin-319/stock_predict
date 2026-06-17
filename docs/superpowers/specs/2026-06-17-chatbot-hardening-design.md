# Chatbot Hardening Design

## Goal
Improve Kakao chatbot safety and input reliability without changing prediction policy: recommendations remain based on `predicted_return`; news/disclosures stay display-only.

## Scope
Implement P0/P1 items from `docs/09_chatbot.md`: optional webhook shared-secret verification, background job throttle/deduplication, stricter stock-code extraction, more tolerant help/status intent matching, and centralized Kakao response truncation.

## Design
- Webhook auth is opt-in. If `KAKAO_WEBHOOK_SECRET` or CLI `--kakao-webhook-secret` is unset, existing local/Colab behavior remains unchanged. If set, `/kakao/webhook` requires `X-Webhook-Secret` matching via constant-time comparison.
- Job spawning is guarded in `KakaoColabPredictionBot._start_prediction_job`: same-symbol running jobs are reused, a small global concurrency cap rejects excess new jobs, and refresh requests are rate-limited per symbol.
- Intent parsing in `src/chatbot/intent.py` normalizes whitespace/punctuation and accepts keyword containment for Korean/English help/status phrases.
- Stock-code extraction only accepts a valid six-digit code with optional `.KS`/`.KQ`; noisy numeric tokens fall through to name search/help behavior instead of pipeline execution.
- `src/chatbot/responses.py` owns `simpleText` length limiting and quick reply formatting, while bot-level code keeps response assembly simple.

## Tests
Add deterministic pytest coverage in `tests/test_chatbot_helpers.py` for auth success/failure, job dedup/throttle/cooldown, intent containment, strict symbol parsing, and response truncation.
