# Operations

## Local Run

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

Artifacts are written under `result/` even when the CLI receives a path outside that directory.

## News Impact Scoring

`news_impact/` is vendored as a standalone package and installs the
`stock-news-impact` console entry point.

Prepare local config from examples:

```powershell
Copy-Item configs/news_impact.example.json configs/news_impact.json
Copy-Item data/news_impact/watchlist.example.csv data/news_impact/watchlist.csv
Copy-Item data/news_impact/company_master.example.csv data/news_impact/company_master.csv
```

`configs/news_impact.example.json` uses OpenAI by default and reads the key from
`OPENAI_API_KEY` (for example, a Colab environment variable). To use the optional
local Gemma/llama.cpp runtime instead, copy `configs/news_impact.gemma.example.json`
over `configs/news_impact.json`.

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

## Kakao/Colab Bot

Operational deployment is GitHub -> Google Colab -> KakaoTalk. Keep the code in GitHub, load or clone it inside Colab, start the chatbot entry point from that runtime, and expose the Flask webhook to KakaoTalk, typically through ngrok.

Start the chatbot entry point:

```powershell
stock-predict-kakao
```

The bot:

- reads cached prediction rows from `result/result_simple.csv`
- maps user symbols with suffix-insensitive matching
- uses `result/result_detail.csv` for the latest prediction date when available
- starts a background prediction job if a symbol is missing
- formats cached rows through one safe formatter path and falls back to the same row-based formatter on legacy formatter errors
- treats news and disclosure summaries as display-only context; they are shown in responses but do not change the next-day expected return or buy/sell/hold signal

## Decision Scope

The output is one reference material for a user's own investment review. The operational buy/sell/hold label is based only on the next-day expected return (`predicted_return`). News, disclosures, and issue summaries are presented so the user can read the surrounding context, not so they can influence the model's expected return.

## Environment Variables

- `OPENAI_API_KEY`, `OPENAI_MODEL`: issue/news summaries; `news_impact` uses
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
