# 코드베이스 종합 분석 및 개선 제안

작성일: 2026-06-23  
대상: `C:\Users\카운\Desktop\stock_predict`

## 1. 프로젝트 목적과 핵심 원칙

이 저장소는 OHLCV, 외부 시장 지표, 수급/공시/뉴스 컨텍스트를 조합해 다음 거래일 예상 수익률과 운용 보조용 신호를 산출하는 Python 파이프라인이다. 산출물은 리서치/운영 참고 자료이며 투자 조언 또는 자동매매 시스템이 아니다.

핵심 guardrail:

- 매수/매도/관망 판단은 `predicted_return` 중심 정책으로 결정한다.
- 뉴스, 공시, LLM 요약, `news_impact_*` 컬럼은 표시용 컨텍스트다.
- 표시용 컨텍스트는 모델 feature, 랭킹, 추천, 신호 정책을 바꾸면 안 된다.
- CSV 산출물은 Windows/Excel 호환을 위해 `utf-8-sig`를 쓴다.
- 생성 산출물은 `result/` 아래에 둔다.

## 2. 전체 규모

로컬 소스 기준 정량 요약:

- Python 파일: 152개 (`src/`, `tests/` 합산)
- 비공백/비주석 Python LOC: 약 22,984줄
- 문서 파일: `docs/` 아래 97개
- 설정 파일: `configs/` 5개
- 샘플/기준 데이터: `data/` 9개

큰 파일:

| 파일 | 대략 LOC | 성격 |
| --- | ---: | --- |
| `src/chatbot/kakao_colab_bot.py` | 2,100 | Kakao/Colab Flask 앱, 캐시, 백그라운드 실행, ngrok |
| `src/pipeline.py` | 1,451 | 메인 CLI orchestration, 검증, 예측, 산출물 작성 |
| `src/reports/issue_summary.py` | 730 | 공시/뉴스 이슈 요약, LLM fallback |
| `src/news_impact/pipeline.py` | 626 | vendored news-impact standalone pipeline |
| `src/reports/news_impact_context.py` | 449 | 예측 결과에 news-impact 표시 컨텍스트 부착 |
| `src/data/investor_context.py` | 435 | 수급/공시/뉴스 raw context 수집 |
| `src/models/lgbm_heads.py` | 422 | LightGBM/sklearn fallback multi-head 모델 |
| `src/domain/signal_policy.py` | 437 | 추천/리스크/포트폴리오 표시 정책 |

## 3. 디렉터리와 책임

| 경로 | 책임 |
| --- | --- |
| `src/pipeline.py` | `run_pipeline()`, CLI parser, 전체 실행 순서, artifact 작성 orchestration |
| `src/config/` | dataclass 기반 `AppConfig`, JSON override, 값 검증 |
| `src/data/` | CSV 로딩/정제, yfinance real OHLCV, KRX universe, 수급/공시/뉴스 context 수집 |
| `src/features/` | 가격/기술지표/외부시장/투자신호 feature 생성, 표시용 context feature 제외 |
| `src/models/` | LightGBM 우선, sklearn fallback 모델 학습/예측/저장 |
| `src/validation/` | walk-forward OOF, calibration, backtest, baseline, signal tuning |
| `src/inference/` | 예측 배열을 결과 frame으로 변환, percentile/scoring 계산 |
| `src/domain/` | `predicted_return` 기반 추천/리스크/행동 정책 |
| `src/reports/` | result CSV/JSON, PM report, issue/news context, artifact manifest |
| `src/news_impact/` | Korean-first 뉴스/공시 영향도 분석 패키지. 메인 파이프라인에서는 표시용만 허용 |
| `src/chatbot/` | KakaoTalk/Colab 응답, session/job store, message formatting |
| `src/ops/` | daily publish, published artifact store |
| `colab/` | Colab helper/runner |
| `configs/` | 보수적 운영/리서치 설정 및 news-impact LLM 예시 |
| `data/` | sample OHLCV, KRX/KOSPI symbol map, news-impact 예시 master/watchlist |
| `tests/` | pytest 단위/통합/회귀 테스트 |

## 4. 실행 엔트리포인트

`pyproject.toml` 기준 console scripts:

- `stock-predict = src.pipeline:main`
- `stock-predict-kakao = src.chatbot.kakao_colab_bot:main`
- `stock-predict-publish = src.ops.publish_predictions:main`
- `stock-news-impact = src.news_impact.run:main`

대표 실행:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

주요 CLI 옵션:

- 입력/출력: `--input`, `--output`, `--report-json`, `--universe-csv`
- 실데이터: `--fetch-real`, `--real-symbols`, `--real-start`, `--auto-refresh-real`, `--add-symbols`
- 외부/컨텍스트: `--disable-external`, `--fetch-investor-context`, `--disable-disclosure-context`
- LLM/API: `--openai-api-key`, `--openai-model`, `--naver-client-id`, `--naver-client-secret`, `--news-impact-llm-config`
- 검증/운영: `--walk-forward-n-jobs`, `--model-n-jobs`, `--model-head-n-jobs`, `--portfolio-value`, `--min-up-probability`, `--min-signal-score`

## 5. 메인 파이프라인 흐름

`src/pipeline.py::run_pipeline()` 기준 요약:

1. 설정 로드: 기본 `AppConfig` + `--config-json` + CLI override 병합.
2. seed 고정: 재현 가능한 학습/검증을 위해 random seed 적용.
3. 선택적 실데이터 갱신: yfinance 기반 full/incremental fetch.
4. 입력 로드/정제: OHLCV 필수 컬럼 표준화, 중복/정렬 처리.
5. universe 적용: CSV 또는 기본 KOSPI200 계열 symbol 필터.
6. 선택적 investor context 수집: 수급, DART, Naver raw event.
7. price/technical feature 생성: return, MA, volatility, RSI, MACD, ATR, CCI, OBV 등.
8. 외부시장 feature 병합: KOSPI/KOSDAQ/S&P/Nasdaq/VIX/환율/금리 등. 실패 시 coverage/fallback 기록.
9. 시장 regime 및 투자신호 feature 추가.
10. 표시용 context column 제외 후 모델 feature 선택.
11. walk-forward OOF 검증, probability calibration, signal weight tuning, long-only top-k backtest.
12. 최종 모델 학습 후 최신 종목 row 예측.
13. issue/news/news-impact 표시 컨텍스트 부착 후 result CSV/JSON/manifest 작성.

## 6. 데이터와 feature 정책

필수 입력 컬럼:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

권장 컬럼:

- `Symbol`
- `market_type`, `venue`, `session`, `listing_date`
- `foreign_net_buy`, `institution_net_buy`
- 표시용 뉴스/공시 context 컬럼

feature 선택 핵심 파일:

- `src/features/price_features.py`: feature/target 생성.
- `src/features/feature_selection.py`: 모델 입력으로 쓸 컬럼을 제한.

중요 정책:

- `is_display_only_context_column()`은 `news_`, `disclosure_`, `_news_`, `_impact_` 패턴과 명시된 display-only 컬럼을 모델 입력에서 제외한다.
- `tests/test_display_only_feature_guard.py`, `tests/test_signal_policy_contract.py`, `tests/test_backtest_and_calibration.py`가 표시용 context가 신호/랭킹에 영향을 주지 않는지 방어한다.

## 7. 모델/검증 구조

모델:

- `src/models/lgbm_heads.py::MultiHeadStockModel`
- LightGBM 설치 시 LightGBM 사용.
- LightGBM 미사용 가능 환경에서는 sklearn `GradientBoosting*` fallback.
- head 구성: next-day return regression, up/down classification, quantile regression.
- feature imputation 값과 metadata/hash를 저장해 schema drift를 줄인다.

검증:

- `src/validation/walk_forward.py`: 시간 순서 보존 fold, purge/embargo, 병렬 fold 실행.
- `src/validation/support.py`: OOF split, probability calibration, diagnostics.
- `src/validation/signal_tuning.py`: OOF train split에서 signal weight 조정.
- `src/validation/backtest.py`: liquidity/capacity/turnover/market type cap을 반영한 long-only top-k backtest.
- `src/validation/result_validity.py`: backtest 결과 validity 판단.

주의:

- 현재 테스트는 `predicted_return_5d`, `predicted_return_20d`가 없음을 검증한다. 문서/README의 multi-horizon 표현은 현 코드와 불일치 가능성이 있다.

## 8. 산출물 구조

`src/reports/run_artifacts.py::RunArtifactManager`가 run별 artifact를 관리한다.

주요 산출물:

- `result/runs/<run_id>/csv/result_detail.csv`
- `result/runs/<run_id>/csv/result_simple.csv`
- `result/runs/<run_id>/csv/result_news.csv`
- `result/runs/<run_id>/csv/result_disclosure.csv`
- `result/runs/<run_id>/pm_report.json`
- `result/runs/<run_id>/pipeline_report.json`
- `result/runs/<run_id>/manifest.json`
- production + real data + pass/warning일 때 `result/latest/`와 compatibility copy를 갱신.

안전장치:

- artifact path가 run directory 밖으로 나가지 못하게 검증.
- 필수 artifact 누락 시 manifest status를 fail로 설정.
- CSV는 `safe_to_csv(..., encoding="utf-8-sig")` 경로를 사용.

## 9. News Impact 패키지

`src/news_impact/`는 별도 `stock-news-impact` 기능을 vendoring한 구조다.

구성:

- collectors: Naver/DART 등 수집.
- deduper/semantic_clusterer: 중복 및 클러스터링.
- impact_judge/llm_client/llm_config: LLM 판정과 cache.
- scorer/ranking/report: 영향도 점수와 보고서.
- stock_factors: 업종/시장/시간 horizon 중심 factor 분석.

메인 `stock-predict` 통합 위치:

- `src/reports/news_impact_context.py`
- `src/pipeline.py::_predict_pipeline_latest()`

통합 정책:

- `--news-impact-report`가 있으면 외부 report를 표시용으로 attach.
- `--news-impact-llm-config`가 있으면 on-demand LLM 판정 후 표시용 컬럼 생성.
- 둘 다 없으면 raw context 기반 rule mode를 사용.
- 실패 시 fallback metadata를 기록하고 예측 자체는 유지.
- 어떤 경로도 `predicted_return`, 추천, ranking을 바꾸면 안 된다.

## 10. Kakao/Colab 운영 흐름

핵심 파일:

- `src/chatbot/kakao_colab_bot.py`
- `src/chatbot/message_formatter.py`
- `src/chatbot/job_store.py`
- `src/chatbot/session_store.py`
- `colab/stock_predict_colab.py`

동작 요약:

1. Kakao utterance를 intent로 분류한다.
2. 종목코드/종목명/상태조회/최신화 요청을 처리한다.
3. `published/latest/` 또는 `result/latest/` 계열 baseline 예측을 우선 읽는다.
4. 필요 시 background pipeline job을 만들고 job/session JSON store에 기록한다.
5. Colab에서는 Flask + pyngrok 조합으로 webhook을 노출한다.
6. 응답 메시지는 simple/detail/news/disclosure context를 조합하지만 신호 기준은 `predicted_return`이다.

## 11. 테스트 전략

주요 테스트 축:

- `tests/test_pipeline_smoke.py`: pipeline 실행, artifact, CLI, report, display-only guard.
- `tests/test_display_only_feature_guard.py`: 표시용 context feature 제외.
- `tests/test_signal_policy*.py`: 추천/리스크/정책 contract.
- `tests/test_backtest_and_calibration.py`: predicted_return 기반 ranking, calibration/backtest.
- `tests/test_walk_forward.py`: fold, purge/embargo, parallel execution.
- `tests/test_lgbm_heads_persistence.py`: 모델 저장/로드/metadata.
- `tests/test_news_impact_*.py`: news-impact config/cache/prompt/context.
- `tests/test_kakao_colab_bot.py`, `tests/test_chatbot_*.py`, `tests/test_colab_runner.py`: 챗봇/Colab 운영 경로.
- `tests/test_*hardening.py`: 출력/캐시/데이터/feature/report 운영 hardening.

좋은 점:

- live integration은 대부분 monkeypatch/sample data 중심이다.
- 표시용 뉴스/공시가 예측/추천을 바꾸지 않는 contract test가 이미 있다.
- result/output 경로와 encoding 회귀 테스트가 있다.

## 12. 강점

- 연구/운영 guardrail이 README, AGENTS, 테스트에 반복 반영되어 있다.
- 외부 API 실패를 pipeline 실패로 곧장 전파하지 않고 coverage/fallback metadata로 보존한다.
- walk-forward OOF와 calibration을 분리해 단순 in-sample 예측보다 안전하다.
- artifact manager가 run별 manifest, latest promotion, compatibility copy를 관리한다.
- config dataclass 검증이 비교적 촘촘하다.
- Kakao/Colab 운영 경로까지 테스트가 존재한다.

## 13. 개선 및 수정 제안

### P0: guardrail 유지/강화

1. `predicted_return` 단독 추천 원칙을 계속 contract test로 잠근다.
   - 유지 파일: `tests/test_signal_policy_contract.py`, `tests/test_backtest_and_calibration.py`.
   - 추가 권장: `news_impact_*` 값만 바뀌는 fixture를 pipeline output level에서도 비교.
2. `select_feature_columns()`의 표시용 제외 규칙을 중앙화한다.
   - 현재 `FEATURE_COLUMN_BASE`에는 일부 news/disclosure 계열 이름이 남아 있고, display-only set/prefix에서 다시 제외한다.
   - 동작은 안전하지만 유지보수자가 오해하기 쉽다.
   - 권장: display-only 후보는 base 목록에서 제거하거나 주석으로 “명시적 제외 검증용”이라고 분리.

### P1: 문서/인코딩 정리

1. `README.md`, `docs/README.md`, 일부 source comment/string에 mojibake가 보인다.
   - 원인 가능성: 과거 CP949/UTF-8 변환 혼재.
   - 권장: 별도 PR에서 문서/주석/사용자 표시 문자열만 UTF-8로 재작성.
   - 주의: 테스트 snapshot 또는 사용자-facing 컬럼명과 연결된 문자열은 한 번에 바꾸지 말고 fixture 업데이트와 함께 처리.
2. 문서의 “multi-horizon” 표현을 현 코드와 맞춘다.
   - 테스트는 `predicted_return_5d`, `predicted_return_20d` 미존재를 검증한다.
   - README/문서가 next-day 중심으로 정리되어야 한다.
3. `docs/README.md`가 `CODEBASE_OVERVIEW.md`를 링크하지만 파일이 없었다.
   - 이 문서 추가로 링크 공백을 복구한다.

### P1: 큰 모듈 분리

1. `src/pipeline.py` 분리.
   - 후보: CLI parser, config override, validation orchestration, prediction orchestration, artifact writing.
   - 이미 `src/pipeline_support.py`, `src/reports/output.py`, `src/data/cli_refresh.py`로 일부 분리되어 있으니 같은 패턴 유지.
2. `src/chatbot/kakao_colab_bot.py` 분리.
   - 후보: Flask route, runtime config, cache loading, pipeline command builder, ngrok bootstrap, bot response service.
   - 현재 2,000줄 이상이라 회귀 위험과 리뷰 비용이 높다.
3. `tests/test_kakao_colab_bot.py`, `tests/test_pipeline_smoke.py`도 fixture/helper를 분리해 가독성을 높인다.

### P1: 병렬성/성능 안전장치

1. 기본값 `walk_forward_n_jobs=-1`, `model_n_jobs=-1`, `model_head_n_jobs=1` 조합은 환경에 따라 CPU oversubscription을 만들 수 있다.
2. Colab/Windows/CI 기본 preset은 보수적으로 제한하고, 고성능 서버 preset만 `-1`을 허용하는 방향이 안전하다.
3. pipeline report에 실제 사용 worker 수와 elapsed time breakdown을 더 자세히 남기면 운영 디버깅이 쉽다.

### P1: 외부 의존성 명확화

1. yfinance, DART, Naver, OpenAI/LLM, ngrok은 모두 선택적 실패 가능 경로다.
2. 현재 fallback metadata는 좋지만, 운영 문서에 “어떤 실패가 예측 품질/표시 컨텍스트에 어떤 영향을 주는지”를 표로 분리하면 좋다.
3. API key는 CLI로 받을 수 있으나 운영에서는 환경변수 또는 secret manager 사용을 표준으로 고정하는 편이 안전하다.

### P2: artifact lifecycle 개선

1. `result/`와 `published/` 계열 산출물의 retention 정책을 한 문서로 합친다.
2. `result/latest/manifest.json`과 `latest_manifest.json`의 차이를 운영자용으로 명확히 설명한다.
3. compatibility root copy(`result_detail.csv` 등)는 legacy surface로 표시하고 신규 코드는 run manifest 기반으로 읽게 한다.

### P2: 데이터 품질/누수 방어

1. `feature_selection.py`와 target 생성 간 누수 가능 컬럼을 더 명시적으로 blacklist한다.
2. 입력 컬럼 alias 처리에서 한국어 컬럼명이 mojibake되어 보이는 영역은 실제 운영 CSV와 대조해 정리한다.
3. `Date` timezone/거래일 calendar 처리를 KRX calendar 모듈로 더 일관화한다.

### P2: CI/문서 품질

1. docs link checker를 추가한다.
2. README/docs mojibake 회귀를 잡는 간단한 grep 기반 테스트를 고려한다.
3. sample smoke command를 CI에 고정한다.
4. `pytest tests/test_pipeline_smoke.py`와 `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`를 PR checklist에 유지한다.

## 14. 권장 작업 순서

1. 문서/인코딩 정리 PR
   - README, docs/README, 사용자-facing 한국어 문자열 점검.
   - multi-horizon 표현을 next-day 중심으로 수정.
2. display-only feature policy 정리 PR
   - `feature_selection.py`의 base/display-only 중복 제거 또는 명시화.
   - 기존 contract tests 유지/확장.
3. pipeline 분리 PR
   - behavior 변경 없이 함수 이동만 수행.
   - smoke + full pytest 필수.
4. chatbot 분리 PR
   - formatter/session/job/runtime/Flask route 경계 강화.
   - Kakao/Colab 테스트 유지.
5. 운영 preset/병렬성 PR
   - Colab/CI/prod worker 기본값과 report metadata 정리.

## 15. 유지보수 체크리스트

변경 전 확인:

- 이 변경이 `predicted_return` 산출, 추천, ranking, signal policy에 영향을 주는가?
- 뉴스/공시/LLM context가 표시용 경계를 넘는가?
- output path가 `result/` 밖으로 나가는가?
- CSV가 `utf-8-sig`로 저장되는가?
- live API 없이 deterministic test가 가능한가?

권장 검증:

```powershell
pytest
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

최소 검증:

```powershell
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```
