# Operations

## Local Run

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

Artifacts are written under `result/` even when the CLI receives a path outside that directory.

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

## Environment Variables

- `OPENAI_API_KEY`, `OPENAI_MODEL`: issue/news summaries
- `DART_API_KEY`, `DART_CORP_MAP_CSV`: DART disclosure context
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: Naver news context

Keep secrets in local environment-specific tooling. Do not commit credentials.

## Troubleshooting

- If CSV outputs are open in Excel, the writer falls back to a `_fallback` filename.
- If OOF predictions are empty, increase the input data length or relax the training window.
- If external downloads fail, rerun with `--disable-external` to verify the local pipeline.
- If Kakao returns a stale job state, request a refresh or remove the stale local state file under `result/` after confirming no process is running.
