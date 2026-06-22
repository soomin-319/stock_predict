# Stock Predict

Python stock prediction pipeline for next-day and multi-horizon return signals. It builds price, market, and investor-flow features, validates with walk-forward OOF predictions, runs a long-only top-k backtest, and writes CSV/JSON artifacts under `result/`.

This project is for research and operations support. The client uses its outputs as one reference material for investment decisions; it is not an investment-advice or automated trading system. Buy/sell/hold signals are based only on the next-day expected return (`predicted_return`). News and disclosures are collected and summarized only for user display, and must not change the expected return or the buy/sell decision.

## Install

Use the same Python environment for install, CLI runs, and tests. In this workspace, the dependency-installed interpreter is Python 3.12 under the user-local Python installation; the global `python` on `PATH` may point elsewhere.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Run

Sample data, no live market downloads:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Installed console entry point:

```powershell
stock-predict --input data/sample_ohlcv.csv --disable-external
```

Real OHLCV refresh before running:

```powershell
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv --real-symbols 005930.KS 000660.KS
```

Append symbols to an existing real-data CSV:

```powershell
python src/pipeline.py --input data/real_ohlcv.csv --add-symbols 005930 000660.KS --real-start 2024-01-01
```

## Main CLI Options

- `--input`: OHLCV input CSV. Default: `data/real_ohlcv.csv`.
- `--output`: legacy compatibility option. CSV outputs are always normalized under `result/`.
- `--universe-csv`: optional CSV with a `Symbol` column.
- `--report-json`: pipeline report JSON filename. Default: `pipeline_report.json`.
- `--news-impact-report`: optional pre-generated `stock-news-impact` JSON report to attach as display-only context.
- `--fetch-real`: download OHLCV before the run. Uses all 200 bundled KOSPI200 symbols when no symbols or universe CSV are specified.
- `--real-symbols`: explicit symbols for `--fetch-real`.
- `--real-start`: start date for real-data fetch. Default: `2020-01-01`.
- `--auto-refresh-real`: incrementally append latest OHLCV when not doing a full refresh. Uses all 200 bundled KOSPI200 symbols by default.
- `--add-symbols`: normalize and append user-entered stock codes/symbols to the input CSV.
- `--disable-external`: skip external market features.
- `--fetch-investor-context`: enable investor flow/disclosure/news context.
- `--disable-disclosure-context`: skip DART disclosure context.
- `--openai-api-key`, `--openai-model`: optional issue/news summary settings.
- `--naver-client-id`, `--naver-client-secret`: optional Naver News Search credentials.
- `--config-json`: nested `AppConfig` override file.
- `--min-value-traded`, `--turnover-limit`, `--min-up-probability`, `--min-signal-score`: backtest/report overrides.
- `--min-external-coverage-ratio`, `--min-investor-coverage-ratio`: coverage gate overrides.
- `--portfolio-value`, `--max-daily-participation`, `--max-positions-per-market-type`: liquidity and portfolio capacity controls.
- `--issue-summary-symbols`: restrict issue-summary generation to selected symbols.
- `--walk-forward-n-jobs`, `--model-n-jobs`, `--model-head-n-jobs`, `--context-raw-event-n-jobs`, `--issue-summary-n-jobs`: parallel worker counts for walk-forward folds, per-model LightGBM training, model heads, raw context-event collection, and issue-summary generation.

Deprecated but still accepted for compatibility: `--dart-api-key`, `--dart-corp-map-csv`.

## Inputs

Required columns:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

Optional but recommended:

- `Symbol`
- investor/context columns such as `foreign_net_buy`, `institution_net_buy`
- display-only issue context such as disclosure/news summaries; these are shown to users but are not inputs to the expected-return decision policy
- market structure fields such as `market_type`, `venue`, `session`, `listing_date`

Use sample data for deterministic local testing. Avoid live network calls in tests.

## Outputs

All generated outputs are written under `result/`:

- `result_detail.csv`: full latest prediction rows and feature context.
- `result_simple.csv`: user-facing summary used by Kakao chatbot.
- `result_news.csv`, `result_disclosure.csv`: live/raw context or generated issue-summary snapshots.
- `pm_report.json`: portfolio-manager style summary.
- `pipeline_report.json` or the `--report-json` filename.

CSV files are saved with `utf-8-sig` for Excel/Windows compatibility.

## Tests

```powershell
pytest
pytest tests/test_pipeline_smoke.py
pytest tests/test_kakao_colab_bot.py
```

Pytest cache and temporary files are configured under ignored `result/` paths.

## Daily Publish (기본 200종목 → GitHub)

로컬에서 수동 1회 실행해 기본 200종목 예측을 `published/`에 게시하고 GitHub에 push한다.

```powershell
# gemma 서버(localhost:8001)가 떠 있으면 gemma 뉴스 임팩트, 아니면 규칙기반 폴백
stock-predict-publish                 # 증분 갱신 + 게시 + commit/push
stock-predict-publish --no-push       # 커밋까지만(푸시 안 함)
stock-predict-publish --dry-run       # 게시 파일만 만들고 commit/push 안 함
stock-predict-publish --news-mode rule --full-refresh
```

산출물:

- `published/latest/` — 최신 게시본(Colab 기본 읽기 대상)
- `published/history/<거래일>/` — 거래일별 스냅샷
- `published/index.json` — 가용 날짜·메타 인덱스

각 폴더는 `csv/result_*.csv`, `manifest.json`, `pipeline_report.json`, `publish_meta.json`을 포함한다.
뉴스/공시 점수는 표시용이며 `predicted_return`·추천·신호 정책에 영향을 주지 않는다.

## Kakao Bot

The production usage pattern is GitHub -> Google Colab -> KakaoTalk:

1. Store the project code (and the published baseline under `published/`) in GitHub.
2. Load the repository in a Colab runtime; by default it serves the GitHub-published baseline without re-running the pipeline.
3. Expose the Colab Flask app, typically through ngrok, so the user can communicate with the service through a KakaoTalk chatbot.

The Kakao/Colab integration lives in `src/chatbot/kakao_colab_bot.py` and can be started locally through:

```powershell
stock-predict-kakao
```

Colab 기본 흐름(기준데이터 서빙):

```python
# 1) 최신 코드/기준데이터 받기
!git pull
# 2) GitHub 기준데이터 표시 (파이프라인 미실행)
from colab.stock_predict_colab import load_published_predictions
load_published_predictions()           # 최신; 특정일은 load_published_predictions("2026-06-17")
# 3) 봇 실행 (자동 부트스트랩 OFF, 기준데이터 베이스라인 서빙)
from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PyngrokTunnelConfig
launch_colab_kakao_bot(tunnel_config=PyngrokTunnelConfig(auth_token="..."), prewarm_cache=False)
```

봇은 `published/latest/`를 기준으로 응답하며, 사용자가 종목코드/이름을 입력하거나 '최신화'를 요청할 때만 해당 종목을 세션에서 예측해 기준데이터 위에 덮어 보여준다(세션 한정, GitHub push 없음). 기본 200종목 재예측은 사용자가 명시적으로 `run_colab_pipeline(...)`을 호출할 때만 수행한다. Chatbot responses may include news/disclosure summaries, but those summaries are display-only context separate from the expected-return signal.


## News Impact Scoring Module

This repository now vendors the full `src.news_impact` package from `stock-news-impact` under the main `src` package. It can run independently through the `stock-news-impact` console entry point after editable install.

Example files:

- `configs/news_impact.example.json`: default LLM config template (local Gemma/llama.cpp `gemma-4-26b-a4b`), matching the `LLMConfig.default()` code default.
- `configs/news_impact.gemma.example.json`: same local Gemma/llama.cpp template, used as the explicit path in the chatbot/CLI integration.
- `configs/news_impact.openai.example.json`: optional OpenAI template; reads the API key from `OPENAI_API_KEY`.
- `data/news_impact/watchlist.example.csv`: watchlist template.
- `data/news_impact/company_master.example.csv`: company master template.

Use Korean company names, industry keywords, and Korean search queries by default. Keep Korean news first; include English or overseas media only when explicitly needed. News-impact outputs remain display/review context in this project and must not change `predicted_return`, recommendation, or automated signal policy.

### Two execution paths

```text
Standalone (research report)              Integrated (display context)
────────────────────────────             ─────────────────────────────
stock-news-impact CLI                     run_pipeline (main prediction)
  -> src.news_impact.pipeline               |
  -> report.json / report.csv               +- --news-impact-report given:
     + audit.json                           |    -> append_news_impact_context
     (reproducibility metadata:             |       (attach the external report)
      llm_model, llm_temperature,           +- otherwise:
      llm_prompt_hash; per-article          |    -> append_generated_news_impact_context
      hashes live in the LLM cache)         |       (from collected raw context events)
                                            -> result_detail.csv `news_impact_*`
                                               (display-only; dropped from
                                                model features and signals)
```

Both paths are review-only. The integrated path attaches `news_impact_*` columns for display; `select_feature_columns()` drops every `news_impact_` column so they never become model inputs, and `predicted_return` stays the sole signal.

```powershell
Copy-Item configs/news_impact.example.json configs/news_impact.json
stock-news-impact --help
```

### On-demand gemma news-impact (챗봇)

요청 종목 예측 시 뉴스/공시 임팩트를 로컬 gemma로 판정하도록 연결할 수 있다.

1. 로컬 llama.cpp로 `gemma-4-26b-a4b`를 `http://localhost:8001/v1`에 서빙.
2. 연결 확인:
   ```bash
   python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json
   ```
   `{"status": "ok", ...}` 이면 정상.
3. 챗봇 구성에서 `PipelineRuntimeConfig(news_impact_llm_config="configs/news_impact.gemma.example.json")` 설정.
4. 이후 단일 종목 예측 / "최신화" 시 gemma로 임팩트를 판정한다. 서버 무응답·오류 시 규칙 기반으로 자동 폴백하며, 부트스트랩(전 종목 prewarm)은 규칙 기반을 유지한다. 산출 점수는 표시용이며 `predicted_return`·추천·신호 정책을 바꾸지 않는다.

## Environment Variables

- `OPENAI_API_KEY`, `OPENAI_MODEL`: issue/news summary generation; `src.news_impact` also uses `OPENAI_API_KEY` by default.
- `DART_API_KEY`, `DART_CORP_MAP_CSV`: disclosure context.
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: Naver news context.

Do not commit API keys, credentials, or private market data.

## Result Lifecycle

공식 최신 운영 결과는 `result/latest/`, 실행별 원본은 `result/runs/<run_id>/`에 있다.
샘플 smoke 실행은 운영 `latest/`를 승격하거나 기존 최상위 호환 CSV를 덮어쓰지 않는다.

```powershell
Get-Content -Encoding utf8 result/latest/manifest.json
```

뉴스와 공시는 표시 전용이며 예상수익률·순위·권고에 영향을 주지 않는다.

## Documentation

문서는 `docs/`에 정리되어 있다. 현재 기준 종합 레퍼런스는 코드베이스 분석 문서이며, `docs/` 전체 구성은 문서 인덱스를 참조한다.

- [문서 인덱스](docs/README.md) — `docs/` 구성 안내
- [코드베이스 분석 (2026-06-22)](docs/CODEBASE_ANALYSIS_2026-06-22.md) — 전체 아키텍처·모듈·파이프라인·가드레일 종합 분석
- [TIMA 벤치마크 업그레이드](docs/TIMA_BENCHMARK_UPGRADE.md) · [TIMA 예측 피처 후보](docs/TIMA_PREDICTION_FEATURE_CANDIDATES.md)

과거 분석·제안·리뷰 및 구 가이드는 [`docs/archive/`](docs/archive/)에, 구현 계획·설계 기록은 [`docs/superpowers/`](docs/superpowers/)에 보관한다.
