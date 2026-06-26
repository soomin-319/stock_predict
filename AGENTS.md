# Repository Guidelines

## Core Guardrails

This Python pipeline predicts next-day stock returns. Treat all outputs as research and operations support, not investment advice or an automated trading system.

Buy/sell/hold decisions must be anchored in next-day expected return (`predicted_return`). News and disclosures may be used as leak-safe, deterministic model/scoring inputs when explicitly configured, so they may affect expected returns, rankings, recommendations, or signals through the documented pipeline. Do not manually override model outputs from narrative context alone.

## Project Layout

Core code lives in `src/`, tests in `tests/`, presets in `configs/`, inputs in `data/`, generated outputs in `result/`, Colab helpers in `colab/`, and notes in `docs/`.

## Development Guidelines

- Use Python 3.10+ and the same environment for installation, CLI runs, and tests.
- Follow existing PEP 8 style. Prefer typed, small functions and minimal diffs.
- Reuse existing helpers before adding abstractions.
- Add or update deterministic pytest tests for behavior changes. Mock or disable live integrations unless testing them directly.
- Run `pytest` before submitting, or at minimum impacted tests plus `pytest tests/test_pipeline_smoke.py`.
- Run the sample pipeline with `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`.

## Agent Execution Guidelines

Do not use subagents. Execute all commands and tool calls sequentially; do not run them in parallel. (Ordering matters for determinism given the pipeline, the single-GPU gemma server on port 8001, and shared `result/` artifacts.)

## Data and Outputs

Keep generated CSV and JSON files under `result/`. CSV outputs must use `utf-8-sig`. Keep sample and universe inputs under `data/`; avoid adding large or private market data.

The vendored `src.news_impact` package should collect Korean news first. Use non-Korean or overseas media only when explicitly needed. When its outputs are used for calculations, keep the transformation deterministic, auditable, and leakage-safe; otherwise treat them as review context.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, often with PR references, for example `Refresh stale cached predictions from detail date in bot handler (#207)`. Keep commits focused. If any repository change is made, the agent MUST commit, push, and create a pull request before final handoff. Do not stop after local changes only. Pull requests should include a summary, test results, linked issues when relevant, and artifact paths when user-facing outputs change. Note new config keys, data files, or external API requirements.

## Branch Handling

- Do not switch the working branch, or create a branch from a base other than the current `HEAD` (for example, off `origin/main`), without an explicit user request. The working branch usually diverges from `main`, so reparenting silently rewrites the working tree: files appear or disappear and look like lost work.
- Add incidental or independent changes (such as a new doc) as commits on the current branch. When the user does ask for a new branch, branch from the current `HEAD`, not from `origin/main`.
- Do not modify or "reconcile" the local `main` branch. Treat `origin/main` as the source of truth and leave local `main` untouched.

## Security & Configuration Tips

Do not commit API keys, ngrok tokens, or private market data. Pass secrets such as `OPENAI_API_KEY`, `DART_API_KEY`, `NAVER_CLIENT_ID`, and `NAVER_CLIENT_SECRET` through environment-specific tooling or local arguments (`--openai-api-key`, `--openai-model`, `--naver-client-id`, `--naver-client-secret`). Treat `result/` as generated output and avoid checking in large or stale artifacts unless intentional.
