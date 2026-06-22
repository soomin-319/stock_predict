# Operations

## Local Run

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Artifacts are written under `result/` even when the CLI receives a path outside that directory.

## News Impact Scoring

`src/news_impact/` is vendored as a standalone package and installs the
`stock-news-impact` console entry point.

Prepare local config from examples:

```powershell
Copy-Item configs/news_impact.example.json configs/news_impact.json
Copy-Item data/news_impact/watchlist.example.csv data/news_impact/watchlist.csv
Copy-Item data/news_impact/company_master.example.csv data/news_impact/company_master.csv
```

`configs/news_impact.example.json` defaults to a local Gemma/llama.cpp runtime
(`gemma-4-26b-a4b` at `http://localhost:8001/v1`), matching the `LLMConfig.default()`
code default. To use OpenAI instead, copy `configs/news_impact.openai.example.json`
over `configs/news_impact.json` and provide `OPENAI_API_KEY` (for example, a Colab
environment variable).

Inspect available options:

```powershell
stock-news-impact --help
```

After `stock-news-impact` writes a JSON report, attach it to normal prediction output
as display-only context:

```powershell
python src/pipeline.py --input data/real_ohlcv.csv --news-impact-report result/news_impact_report.json
```

Operational rule: use Korean company names, Korean industry/search keywords, and
Korean news first. Use non-Korean or overseas media only when explicitly needed.
The appended `news_impact_*` columns are review metadata; they must not change
`predicted_return`, recommendation labels, ranking, or automated signal policy.

## Real Data Refresh

Full refresh:

```powershell
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv --real-symbols 005930.KS 000660.KS
```

Incremental refresh:

```powershell
python src/pipeline.py --input data/real_ohlcv.csv --auto-refresh-real
```

Append selected symbols:

```powershell
python src/pipeline.py --input data/real_ohlcv.csv --add-symbols 005930 000660.KS --real-start 2024-01-01
```

## Daily Publish

Publish the default 200-symbol baseline to `published/` and push it to GitHub by running `stock-predict-publish` once, locally, on demand (no cron). It incrementally refreshes real data, runs the pipeline, copies the operational run into `published/latest/` and `published/history/<trading-date>/`, updates `published/index.json`, then commits and pushes. Use `--no-push` to commit only, `--dry-run` to write the published files without committing, `--full-refresh` to refetch from scratch, and `--news-mode rule` to force rule-based news impact (otherwise gemma at `localhost:8001` with automatic rule-based fallback). News/disclosure scores stay display-only and do not change `predicted_return`, recommendation, or signal policy. See the README "Daily Publish" section for the command/output reference.

## Kakao/Colab Bot

Operational deployment is GitHub -> Google Colab -> KakaoTalk. Keep the code and the published baseline (`published/`) in GitHub, load or clone it inside Colab, start the chatbot entry point from that runtime, and expose the Flask webhook to KakaoTalk, typically through ngrok. In Colab, call `load_published_predictions()` to display the GitHub baseline without running the pipeline.

Start the chatbot entry point:

```powershell
stock-predict-kakao
```

The bot:

- serves the `published/latest/` baseline and overlays any session predictions from `result/` (session rows override the baseline per 종목코드)
- maps user symbols with suffix-insensitive matching
- resolves the latest prediction date from the session detail first, then the published baseline detail
- predicts only the specific symbol a user requests (or asks to refresh) within the session; automatic bootstrap/prewarm of the default symbols is off, and session predictions are not pushed to GitHub
- formats cached rows through one safe formatter path and falls back to the same row-based formatter on legacy formatter errors
- treats news and disclosure summaries as display-only context; they are shown in responses but do not change the next-day expected return or buy/sell/hold signal

## Decision Scope

The output is one reference material for a user's own investment review. The operational buy/sell/hold label is based only on the next-day expected return (`predicted_return`). News, disclosures, and issue summaries are presented so the user can read the surrounding context, not so they can influence the model's expected return.

## Environment Variables

- `OPENAI_API_KEY`, `OPENAI_MODEL`: issue/news summaries; `src.news_impact` uses
  `OPENAI_API_KEY` by default
- `DART_API_KEY`, `DART_CORP_MAP_CSV`: DART disclosure context
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: Naver news context
- `NEWS_IMPACT_CONFIG` or local `configs/news_impact.json`: optional standalone news-impact runtime config, if used by local tooling

Keep secrets in local environment-specific tooling. Do not commit credentials.

## Troubleshooting

- If CSV outputs are open in Excel, the writer falls back to a `_fallback` filename.
- If OOF predictions are empty, increase the input data length or relax the training window.
- If external downloads fail, rerun with `--disable-external` to verify the local pipeline.
- If Kakao returns a stale job state, request a refresh or remove the stale local state file under `result/` after confirming no process is running.
- If `--news-impact-report` adds no columns, confirm the JSON has `rows` with `date` and `ticker` fields and that tickers match pipeline symbols such as `005930.KS`.
