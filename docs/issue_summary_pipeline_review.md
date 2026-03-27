# Issue Summary / Pipeline Runtime Review (2026-03-27)

This document captures a lightweight architecture review of the current prediction + issue-summary runtime.
It is intended as a practical checklist for incremental refactoring without a full rewrite.

## Current flow (high level)

1. `run_pipeline()` handles end-to-end prediction generation and can append issue-summary columns.
2. The Kakao bot request handler checks bootstrap/prediction status, reads cache, and triggers summary generation when needed.
3. Live summary attachment can collect events/news and generate fallback summaries when LLM output is missing.

## Main risks observed

### 1) Coupled orchestration logic
- Prediction and summary orchestration are still strongly coupled in runtime paths.
- The chatbot request path currently mixes status control, cache resolution, and summary trigger decisions.

### 2) Duplicate event-fetch opportunities
- Batch pipeline context collection and chatbot live enrichment can both fetch similar source data.
- Shared caching policy is not explicit enough to prevent repeated upstream calls.

### 3) Date policy ambiguity
- Effective date selection can diverge between prediction date and "today" depending on path.
- A single utility-level date policy should be reused across pipeline and chatbot.

### 4) Background concurrency limits
- Summary jobs are launched in ad-hoc background threads.
- Under burst traffic this can become difficult to reason about vs bounded queue/worker model.

### 5) Failure observability
- Several broad exception handlers reduce precision in diagnosing API/network/model failures.
- Error classes/codes and metrics would improve operational debugging.

## Recommended staged plan

### P0 (immediate)
1. Introduce a single summary service interface used by chatbot handlers.
2. Centralize date-policy selection into a reusable helper.
3. Add a bounded worker queue for summary jobs (with symbol dedup key).

### P1 (short term)
1. Add source-event cache keying (`symbol`, `date`, `source`) with TTL.
2. Add structured error categories (API timeout, auth, parse, model, I/O).
3. Emit lightweight metrics (cache hit rate, summary latency, failure rate).

### P2 (mid term)
1. Split chatbot runtime concerns into orchestration/service/storage modules.
2. Separate bootstrap prewarm responsibilities from summary serving responsibilities.

## Acceptance checks for next implementation PR

- No duplicate summary jobs for the same symbol/date while one is in progress.
- Date chosen for summary generation is deterministic and logged.
- Cache-hit path avoids external API calls.
- Failures include machine-readable error category.
