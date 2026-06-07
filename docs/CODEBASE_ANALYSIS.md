# 코드베이스 분석

이 문서는 현재 코드 기준으로 주식 예측 파이프라인의 구조와 운영 포인트를 빠르게 파악하기 위한 분석 문서다. 기존 `README.md`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md`는 수정하지 않고, 실제 구현 파일을 기준으로 저장소의 큰 흐름과 책임 경계를 정리한다.

## 프로젝트 목적

이 저장소는 OHLCV 가격 데이터와 시장/수급 맥락을 결합해 다음 거래일 및 5일/20일 수익률 신호를 만드는 Python 파이프라인이다. 공시/뉴스는 예상 수익률에 영향을 주는 입력이 아니라 사용자에게 보여주는 표시용 컨텍스트로 다룬다. 주요 결과는 최신 종목별 예측, 사용자용 요약 CSV, 포트폴리오 매니저용 JSON, 백테스트/진단 그림으로 `result/` 아래에 저장된다.

이 프로젝트는 리서치와 운영 보조용이며 투자 조언 시스템이 아니다. 클라이언트는 결과물을 투자 판단에 참고할 하나의 근거자료로 사용한다. 운영상 매수/매도/관망 신호는 다음날 예상 수익률(`predicted_return`)만 기준으로 삼고, 뉴스/공시는 사용자가 읽는 참고 정보로만 표시한다.

## 전체 구조

- `src/pipeline.py`: 메인 CLI와 `run_pipeline(...)` 오케스트레이션.
- `src/config/`: dataclass 기반 설정과 JSON override 병합.
- `src/data/`: 입력 CSV 로딩, 정제, 실데이터 다운로드, 유니버스 필터, 투자자 컨텍스트와 표시용 공시/뉴스 컨텍스트 수집.
- `src/features/`: 가격/기술적 지표, 외부 시장 지표, 시장 국면, 투자 신호 피처 생성.
- `src/models/`: LightGBM 또는 sklearn fallback 기반 멀티 헤드 모델.
- `src/validation/`: walk-forward 검증, OOF 예측, 백테스트, 기준 모델, 보정/진단 지표.
- `src/inference/`: 최신 행에 대한 예측 프레임과 신호 점수 계산.
- `src/domain/`: 추천, 리스크 플래그, 이벤트 부스트, 포트폴리오 정책 필드.
- `src/reports/`: CSV/JSON/그림 산출물 생성과 콘솔 출력.
- `src/chatbot/`: Kakao/Colab 웹훅, 캐시 조회, 백그라운드 예측 작업.
- `colab/`: Colab 친화적 실행 래퍼.
- `configs/`: 운영/리서치용 설정 프리셋.
- `data/`: 샘플 OHLCV, 기본 유니버스, KRX 종목명 매핑.
- `tests/`: pytest 기반 단위/통합/스모크 테스트.

## 실행 엔트리포인트

메인 엔트리포인트는 세 가지다.

- `python src/pipeline.py ...`: 직접 스크립트 실행. `build_cli_parser()`가 CLI 옵션을 정의하고 `main()`이 데이터 갱신 옵션을 처리한 뒤 `run_pipeline(...)`을 호출한다.
- `stock-predict`: `pyproject.toml`의 console script이며 `src.pipeline:main`으로 연결된다.
- `stock-predict-kakao`: Kakao/Colab 웹훅 실행용 console script이며 `src.chatbot.kakao_colab_bot:main`으로 연결된다.

Colab에서는 `colab/stock_predict_colab.py`의 `run_colab_pipeline(...)`을 통해 파이프라인을 호출한다. 입력 CSV가 데모/placeholder 성격이면 기본 KRX 유니버스를 먼저 받아 `data/real_ohlcv.csv`를 만든 뒤 실행할 수 있다.

## 13단계 파이프라인 흐름

`src/pipeline.py`의 `run_pipeline(...)`은 진행 상황을 `[1/13]` 형식으로 출력하며 다음 순서로 동작한다.

1. 앱 설정 로딩: `load_app_config()`가 기본 dataclass 설정, `--config-json`, CLI override를 병합하고 RNG seed를 고정한다.
2. 입력 데이터 로딩: `load_ohlcv_csv()`가 OHLCV CSV를 읽는다.
3. 데이터 정제와 유니버스 필터: `clean_ohlcv()`로 날짜/심볼/가격 컬럼을 정규화하고, `--universe-csv`가 있으면 `filter_by_universe()`를 적용한다.
4. 투자자 컨텍스트 추가: `--fetch-investor-context`가 켜져 있으면 수급 컨텍스트를 병합하고, DART/Naver 자격 정보가 있으면 사용자 표시용 raw event도 수집한다.
5. 가격 피처 생성: `build_features()`가 수익률, 이동평균, 변동성, RSI, MACD, ATR, OBV, 목표값 등을 만든다.
6. 외부 시장 피처 추가: `add_external_market_features_with_coverage()`가 KOSPI/KOSDAQ/S&P/Nasdaq/VIX/환율/금리 등 시장 지표를 붙이고 coverage를 기록한다.
7. walk-forward 검증: `walk_forward_validate_with_oof()`가 시간 순서를 지키는 fold와 OOF 예측을 만든다. fold가 부족하면 `_adaptive_training_cfg()`로 학습/검증 창을 줄여 재시도한다.
8. 기준 모델 평가: `evaluate_baselines()`가 단순 기준 성능을 계산한다.
9. OOF 예측 사용: OOF 예측을 `MultiHeadPrediction`으로 변환하고 상승 확률을 보정한 뒤 `build_scored_prediction_frame()`으로 신호 점수와 정책 계산용 컨텍스트를 붙인다.
10. 신호 가중치 튜닝: `tune_signal_weights()`가 OOF train split에서 수익률, 상승확률, 상대강도, 불확실성 패널티 가중치를 조정한다.
11. 최종 모델 학습과 최신 예측: 전체 또는 최근 lookback 데이터로 `MultiHeadStockModel`을 학습하고 최신 종목 행을 예측한다. 종목명, 히스토리 방향 정확도, 추천/리스크/이슈 요약도 이 단계에서 붙는다.
12. 산출물 저장: detail/simple/news/disclosure CSV, PM report JSON, pipeline report JSON을 `result/` 아래에 저장하고 콘솔 요약을 출력한다.

## 주요 모듈 책임

### `src/data`

- `loaders.py`: OHLCV CSV를 읽고 `Date`, `Symbol` 중심의 기본 형태를 만든다.
- `cleaners.py`: 가격/거래량 컬럼을 숫자로 정리하고 정렬/중복 제거 같은 기본 정제를 수행한다.
- `fetch_real_data.py`: 사용자 입력 종목코드를 yfinance 심볼로 정규화하고 실 OHLCV를 다운로드한다. `save_real_ohlcv_csv()`는 전체 갱신, `append_real_ohlcv_csv()`는 기존 CSV에 증분 병합을 담당한다.
- `cli_refresh.py`: CLI에서 real refresh 대상 심볼과 증분 시작일을 결정한다.
- `universe.py`: 유니버스 CSV의 `Symbol` 컬럼을 로딩하고 데이터프레임을 필터링한다.
- `krx_universe.py`: `data/krx_symbol_name_map.csv` 기반 종목명 매핑과 이름 검색 후보 생성을 제공한다.
- `investor_context.py`: 투자자 수급과 표시용 DART 공시/Naver 뉴스 raw event 수집을 담당한다. 외부 소스 실패 시 빈 컨텍스트와 coverage 정보를 반환하도록 설계되어 있다.

### `src/features`

- `price_features.py`: 가장 큰 피처 생성 모듈이다. 가격 기반 수익률/지표, 투자자 이벤트 피처, 52주 신고가/거래대금 플래그, 1일/5일/20일 target을 만든다. `select_feature_columns()`는 모델에 넣을 피처 컬럼을 선택한다. 운영 정책상 공시/뉴스는 표시용 컨텍스트로 남겨야 하며 예상 수익률 결정에는 반영하지 않는다.
- `external_features.py`: yfinance 외부 지표를 다운로드하고 날짜 기준으로 병합한다. 실패한 심볼과 fallback 여부를 coverage dict로 남긴다.
- `regime_features.py`: 시장 국면 주석을 추가한다.
- `investment_signals.py`: 거래대금 순위, 강한 외국인/기관 순매수, Nasdaq tailwind/headwind, RSI 관찰 구간 등 투자 조건 피처를 만든다.

### `src/models`

`lgbm_heads.py`의 `MultiHeadStockModel`은 여러 head를 함께 학습한다.

- 1일 수익률 회귀 head
- 1일 상승 확률 분류 head
- 분위수 회귀 head
- 선택 가능한 5일/20일 수익률 및 상승 확률 head

LightGBM이 설치되어 있으면 LightGBM을 쓰고, 없으면 sklearn `GradientBoostingRegressor`/`GradientBoostingClassifier`로 fallback한다. 모델 저장/로드 시 artifact version, backend, feature hash, quantile/horizon metadata를 sidecar JSON으로 남긴다.

### `src/validation`

- `walk_forward.py`: 시간 순서를 지키는 fold 생성과 OOF 예측을 담당한다. `purge_gap_days`와 `embargo_days`로 multi-horizon target 누수를 줄인다.
- `support.py`: OOF split, 확률 보정, OOF 진단, `MultiHeadPrediction` 변환을 제공한다.
- `backtest.py`: coverage gate, 유동성/거래대금 제한, 시장 구분별 보유 수 제한, turnover 비용을 반영한 long-only top-k 백테스트를 수행한다.
- `baselines.py`: 기준 모델 평가.
- `metrics.py`: 회귀/분류/확률 보정 지표.
- `signal_tuning.py`: 신호 점수 가중치 튜닝.

### `src/inference`와 `src/domain`

`src/inference/predict.py`는 예측값을 사람이 읽을 수 있는 프레임으로 바꾼다. `predicted_return`, `predicted_close`, `up_probability`, `uncertainty_score`, `norm_return`, `rel_strength`, `signal_score`, horizon별 예측값을 계산한다.

`src/domain/signal_policy.py`는 모델 출력에 정책을 더한다. 운영 기준의 기본 추천은 예측 수익률 임계값만 사용한다. `signal_score`, 상승 확률, 불확실성, Nasdaq headwind, RSI 과열, 데이터 coverage, 유동성, 히스토리 정확도 등은 risk flag, portfolio action, 진단 정보로 표시할 수 있지만 매수/매도/관망 라벨을 바꾸는 근거로 쓰지 않는다. 공시/뉴스는 이슈 요약으로 표시만 한다.

### `src/reports`

- `output.py`: 모든 출력 경로를 프로젝트 `result/` 아래로 정규화한다. CSV는 `utf-8-sig`로 저장하며, 파일이 열려 있어 `PermissionError`가 나면 `_fallback` 파일명으로 저장한다.
- `result_formatter.py`: 사용자용 `result_simple.csv` 스키마와 콘솔 요약 출력.
- `issue_summary.py`: rule-based 또는 OpenAI 기반 공시/뉴스 이슈 요약 컬럼 생성.
- `pm_report.py`: PM 관점의 coverage, risk count, horizon summary, top buy 후보 JSON 생성.

## 입력과 설정

필수 입력 CSV 컬럼은 `Date`, `Open`, `High`, `Low`, `Close`, `Volume`이다. `Symbol`은 다중 종목 처리에 사실상 필요하며, 없으면 loader가 기본 심볼을 보강하는 경로가 있다.

선택 입력으로는 `foreign_net_buy`, `institution_net_buy`, `market_type`, `venue`, `session`, `listing_date` 같은 수급/시장 구조 컬럼이 있다. 공시/뉴스 원문과 요약은 사용자 표시용 컨텍스트로 분리한다. 컬럼이 없으면 피처 생성기는 대부분 기본값으로 채워 로컬 스모크 실행이 가능하게 되어 있다.

설정은 `src/config/settings.py`의 `AppConfig` 아래에 있다.

- `UniverseConfig`: 유니버스 이름과 기대 크기.
- `FeatureConfig`: lookback, 이동평균, 변동성, RSI/CCI/Stochastic 기간.
- `ExternalFeatureConfig`: 외부 시장 심볼 목록과 enable flag.
- `TrainingConfig`: walk-forward 창, quantile, seed, 병렬도, GPU, purge/embargo, 최종 모델 lookback.
- `SignalConfig`: 신호 점수 가중치.
- `InvestmentCriteriaConfig`: 거래대금, 수급, RSI, Nasdaq, 52주 신고가 관련 정책 임계값.
- `BacktestConfig`: top-k, 포트폴리오 금액, 비용, turnover, 유동성, coverage gate, 시장별 보유 제한.

`configs/prod_conservative.json`과 `configs/research_balanced.json`은 이 dataclass 구조에 맞는 JSON override 프리셋이다. CLI override는 주로 `backtest`와 `training` 일부 값을 덮어쓴다.

## 외부 API와 환경 변수

외부 호출은 선택적이다.

- yfinance: 실 OHLCV와 외부 시장 지표 다운로드.
- 투자자 수급 컨텍스트: 현재 외부 수집 경로가 비활성화되어 있다.
- DART: 공시 컨텍스트와 raw disclosure event 수집.
- Naver News Search: 뉴스 raw event 수집.
- OpenAI: 공시/뉴스 이슈 요약 생성.
- Flask/pyngrok: Kakao webhook과 Colab 터널.

주요 환경 변수는 `OPENAI_API_KEY`, `OPENAI_MODEL`, `DART_API_KEY`, `DART_CORP_MAP_CSV`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`이다. API key는 CLI 인자로 넘길 수도 있지만, Kakao 백그라운드 subprocess는 secret을 argv가 아닌 환경 변수로 전달하도록 구현되어 있다.

## 산출물

파이프라인 산출물은 `src/reports/output.py`의 경로 정규화 규칙에 따라 `result/` 아래로 모인다.

- `result/result_detail.csv`: 최신 예측의 상세 행과 피처/정책 컨텍스트.
- `result/result_simple.csv`: Kakao 봇과 사람이 보기 위한 요약 CSV.
- `result/result_news.csv`: 당일 뉴스 raw event 또는 이슈 요약 snapshot.
- `result/result_disclosure.csv`: 당일 공시 raw event 또는 이슈 요약 snapshot.
- `result/pm_report.json`: PM 요약 JSON.
- `result/<report_json>`: `pipeline_report.json` 등 실행 요약 JSON.
- `result/chatbot_jobs.json`, `result/chatbot_sessions.json`, `result/prewarm_cache_meta.json`: Kakao/Colab 봇 운영 상태와 캐시 메타데이터.

## 모델, 검증, 신호 정책

검증은 OOF 예측을 중심으로 설계되어 있다. walk-forward fold에서 모델을 학습하고, 검증 구간 예측을 모아 OOF 데이터프레임을 만든다. 이 OOF 예측은 다음 용도로 재사용된다.

- 평균 walk-forward 성능 요약.
- 확률 보정과 calibration metric 계산.
- 신호 가중치 튜닝용 train split과 백테스트용 holdout split.
- 종목별 과거 방향 정확도 계산.
- 최신 예측의 uncertainty percentile scale과 score 해석 기준.

최종 최신 예측은 별도로 다시 학습한 모델에서 나온다. 이때 `TrainingConfig.final_model_lookback_days`가 0보다 크면 최근 N 거래일만 사용한다. 예측 후에는 OOF 기반 보정, 리스크 플래그, 포트폴리오 액션, 이슈 요약, 백테스트 요약 컬럼이 추가된다. 매수/매도/관망 신호는 다음날 예상 수익률 기준이며 이슈 요약은 표시용이다.

백테스트는 단순히 점수 상위 종목을 사는 구조가 아니라 coverage gate, 최소 거래대금, 포트폴리오 금액 대비 일일 참여율, 시장 구분별 최대 보유 수, turnover 비용을 반영한다. 이 때문에 모델 성능과 운영 가능성 지표를 함께 봐야 한다.

## Kakao/Colab 봇 흐름

`src/chatbot/kakao_colab_bot.py`는 Flask app과 `KakaoColabPredictionBot`을 제공한다. 기본 흐름은 다음과 같다.

1. 사용자의 발화를 종목코드, 상태 조회, 새로고침, 도움말, 종목명 검색으로 분류한다.
2. `result/result_simple.csv`를 mtime 기반 캐시로 읽고, 요청 종목의 기존 예측이 있으면 즉시 응답한다.
3. detail/news/disclosure CSV를 참고해 표시용 이슈 요약이 없으면 동기 또는 백그라운드로 보강한다.
4. 캐시가 없거나 refresh 요청이면 `PipelineRuntimeConfig.build_command()`로 `python src/pipeline.py --add-symbols ...` 명령을 구성해 백그라운드 예측 작업을 시작한다.
5. 작업 상태는 `result/chatbot_jobs.json`, 사용자 세션은 `result/chatbot_sessions.json`에 저장한다.
6. Colab 시작 시에는 `prewarm_prediction_cache()` 또는 bootstrap job으로 기본 종목 예측 캐시를 미리 만들 수 있다.

운영 구성은 GitHub에 저장된 코드를 Colab에서 불러와 실행하고, Colab의 Flask webhook을 KakaoTalk 챗봇과 연결하는 흐름을 전제로 한다. 봇 응답에는 다음날 예상 수익률 기반 신호와 함께 뉴스/공시 요약을 표시할 수 있지만, 해당 요약은 신호 산식과 분리된다.

봇은 오래된 running 상태를 실패로 내리는 방어 로직, 긴 Kakao simpleText 응답을 잘라내는 로직, formatter 예외 시 fallback formatter를 쓰는 로직을 가지고 있다. 테스트도 이 캐시/백그라운드 경로를 집중적으로 다룬다.

## 테스트 전략

테스트는 live network에 의존하지 않고 monkeypatch와 sample data를 우선 사용한다.

- `tests/test_pipeline_smoke.py`: 파이프라인 실행, 출력 경로, simple/detail 산출물, CLI 옵션, scoring frame, 외부 피처 실패 처리.
- `tests/test_walk_forward.py`: fold 생성, purge/embargo, 병렬 fold 실행 일관성.
- `tests/test_backtest_and_calibration.py`, `tests/test_probability_calibration_guard.py`: 백테스트 제약과 확률 보정 guard.
- `tests/test_lgbm_heads_persistence.py`: 모델 저장/로드와 feature hash/version 검증.
- `tests/test_investor_features.py`, `tests/test_investment_signal_features.py`, `tests/test_external_features.py`: 피처 생성과 외부 데이터 fallback.
- `tests/test_fetch_real_fallback.py`, `tests/test_universe_loader.py`, `tests/test_krx_symbol_names.py`: 실데이터 갱신, 기본 유니버스, 종목명 매핑.
- `tests/test_signal_policy_recommendation.py`, `tests/test_signal_policy_reason.py`, `tests/test_console_summary.py`: 추천 정책, 이유 문자열, 콘솔/simple 요약.
- `tests/test_issue_summary.py`, `tests/test_investor_context_news.py`, `tests/test_investor_context_integration.py`: 공시/뉴스 raw event와 이슈 요약.
- `tests/test_kakao_colab_bot.py`, `tests/test_colab_runner.py`: Kakao webhook, 캐시, prewarm, pyngrok, Colab runner.

문서만 바꾸는 작업에는 전체 `pytest`가 필수는 아니지만, 코드 동작을 바꾸는 경우에는 최소 영향 테스트와 `tests/test_pipeline_smoke.py`를 같이 실행하는 편이 좋다.

## 유지보수 주의점

- `src.pipeline.run_pipeline(...)`, `src.pipeline.build_cli_parser()`, console script 이름, `result_detail.csv`/`result_simple.csv` 등 출력 파일명은 외부에서 참조되는 public surface로 보고 보존해야 한다.
- `src/pipeline.py`에는 과거 테스트/호출자를 위한 compatibility wrapper가 남아 있다. 새 로직은 가능한 한 `src/data`, `src/validation`, `src/reports`, `src/domain` 등 책임 모듈에 두고 pipeline은 orchestration에 집중시키는 편이 좋다.
- 출력 경로는 `result/` 아래로 강제 정규화된다. 새 산출물을 만들 때도 같은 규칙을 재사용해야 운영/테스트 기대와 맞는다.
- CSV 출력은 Windows/Excel 호환을 위해 `utf-8-sig`를 사용한다.
- 외부 API 실패는 로컬 파이프라인 실행을 깨지 않도록 coverage/fallback 형태로 처리하는 패턴을 유지해야 한다.
- 입력 feature와 target은 시간 순서와 multi-horizon leakage에 민감하다. walk-forward의 purge/embargo 의미를 변경할 때는 `tests/test_walk_forward.py`를 먼저 확인해야 한다.
- 현재 일부 한국어 문자열과 주석에는 mojibake가 남아 있다. 동작 수정과 인코딩 정규화는 섞지 말고 별도 작업으로 분리하는 것이 안전하다.

## News Impact Scoring Package Update

`src/news_impact/` has been added as a vendored package under `src/` migrated from
`stock-news-impact`.

Key surfaces:

- Package modules: `src.news_impact.pipeline`, `src.news_impact.collectors`,
  `src.news_impact.scorer`, `src.news_impact.schema`, `src.news_impact.llm_client`, and
  `src.news_impact.stock_factors.*`.
- Console entry point: `stock-news-impact = src.news_impact.run:main`.
- Example runtime files:
  - `configs/news_impact.example.json`
  - `data/news_impact/watchlist.example.csv`
  - `data/news_impact/company_master.example.csv`
- Main pipeline bridge: `src/reports/news_impact_context.py`.
- CLI bridge: `python src/pipeline.py ... --news-impact-report <report.json>`.

Boundary rule:

- The standalone news-impact pipeline may collect, deduplicate, classify, score,
  backtest, and report news/disclosure context.
- The stock-prediction pipeline only joins selected report fields as
  `news_impact_*` display columns.
- Joined news-impact fields must not change model features, `predicted_return`,
  expected-return ranking, recommendation labels, or automated signal policy.

Testing coverage now includes:

- `tests/test_news_impact_full_package.py`: verifies the vendored package imports,
  package metadata, console entry point, and example runtime files.
- `tests/test_news_impact_context.py`: verifies optional report joining is
  display-only and no-op safe when the report is missing or invalid.
