# P1 Kakao bot message formatter decomposition design

## Goal

Reduce the `KakaoColabPredictionBot` god-class surface by extracting cached prediction message formatting into a small, testable formatter module.

This is the first small PR in the broader P1 chatbot decomposition. It must not change recommendation, ranking, expected return, or signal behavior.

## Scope

In scope:

- Add `src/chatbot/message_formatter.py`.
- Move prediction message formatting helpers from `KakaoColabPredictionBot` into a formatter object.
- Keep existing `KakaoColabPredictionBot` helper method names as compatibility wrappers.
- Preserve current Kakao message text, fallback behavior, and tests.

Out of scope:

- Flask route changes.
- Prediction job/subprocess changes.
- Session/state registry changes.
- Live news/disclosure collection changes.
- Any change to buy/sell/hold logic.

## Architecture

Add `PredictionMessageFormatter` with methods for:

- full prediction message rendering;
- reason-line rendering;
- issue summary block rendering;
- news/disclosure impact block rendering;
- percent, price, and confidence formatting;
- text cleanup and bullet splitting.

`KakaoColabPredictionBot` owns one formatter instance and delegates current formatting methods to it. This preserves the bot's public/test-facing method surface while removing the formatting implementation from the class.

## Data flow

Current:

`KakaoColabPredictionBot` cached row -> bot formatting helpers -> Kakao response.

New:

`KakaoColabPredictionBot` cached row -> `PredictionMessageFormatter` -> Kakao response.

The input remains a `pandas.Series`. The output remains a plain string.

## Guardrails

- Recommendation text remains whatever is already present in the prediction row.
- No news/disclosure field can change expected return, recommendation, ranking, or signal.
- News/disclosure and impact fields remain display-only message context.
- Encoding-sensitive Korean expected values in tests should use existing fixtures or Unicode-safe literals.

## Error handling

Existing fallback behavior remains:

- `_format_cached_prediction_message()` first uses the normal formatter path.
- If the legacy rationale bug patch applies, it retries the canonical formatter.
- Otherwise it logs and falls back to the same canonical formatter.

The formatter itself should not access bot state or external services.

## Testing

Add or update tests to prove:

- formatter output matches the existing bot helper output for representative cached prediction rows;
- bot compatibility wrappers delegate to the formatter;
- issue summary/news impact display still appears unchanged;
- display-only context still has no influence on recommendation fields.

Run:

- impacted Kakao/chatbot tests;
- `pytest tests/test_pipeline_smoke.py`;
- sample pipeline smoke command;
- full `pytest -q` before PR.

## Rollout

Ship as a stacked draft PR on top of the current P0/P2 improvement stack. Later P1 PRs can extract session/job/context responsibilities independently.
