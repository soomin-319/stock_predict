# Repository Guidelines

## Core Guardrails

This Python pipeline predicts next-day and multi-horizon stock returns. Treat all outputs as research and operations support, not investment advice or an automated trading system.

Buy/sell/hold decisions must use next-day expected return (`predicted_return`). News and disclosures are display-only context and must never change expected returns, rankings, recommendations, or signals.

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

Do not use subagents. Execute all commands and tool calls sequentially; do not run them in parallel.

## Data and Outputs

Keep generated CSV and JSON files under `result/`. CSV outputs must use `utf-8-sig`. Keep sample and universe inputs under `data/`; avoid adding large or private market data.

The vendored `src.news_impact` package should collect Korean news first. Use non-Korean or overseas media only when explicitly needed. Its output remains display/review context only.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, often with PR references, for example `Refresh stale cached predictions from detail date in bot handler (#207)`. Keep commits focused. If any repository change is made, create a pull request before final handoff unless the user explicitly says not to. Pull requests should include a summary, test results, linked issues when relevant, and artifact paths when user-facing outputs change. Note new config keys, data files, or external API requirements.

## Security & Configuration Tips

Do not commit API keys, ngrok tokens, or private market data. Pass secrets such as `OPENAI_API_KEY`, `DART_API_KEY`, `NAVER_CLIENT_ID`, and `NAVER_CLIENT_SECRET` through environment-specific tooling or local arguments (`--openai-api-key`, `--openai-model`, `--naver-client-id`, `--naver-client-secret`). Treat `result/` as generated output and avoid checking in large or stale artifacts unless intentional.
