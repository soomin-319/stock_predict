# 코드베이스 분석 (2026-06-22)

> 분석 대상 브랜치: `codex/report-metadata-hardening`
> 소스 규모: `src/` Python 약 16,400 LOC, 테스트 49개 모듈
> 작성: 전체 모듈 정독 기반 종합 분석

---

## 1. 프로젝트 개요

**Stock Predict**는 한국 주식(기본 유니버스 KOSPI200)의 **익일 종가 기준 기대수익률**을 예측하는 파이썬 파이프라인이다. 가격·시장·투자자 수급 피처를 만들고, Walk-Forward OOF로 검증하며, long-only top-k 백테스트를 거쳐 `result/` 아래에 CSV/JSON 산출물을 쓴다.

### 최상위 가드레일 (절대 규칙)

- 모든 산출물은 **투자 참고 자료**이며 투자자문/자동매매가 아니다.
- 매수/매도/관망 결정은 **오직 익일 기대수익률(`predicted_return`)** 로만 한다.
- 뉴스·공시·뉴스임팩트는 **표시 전용(display-only)** 컨텍스트다. 기대수익률·순위·권고·시그널에 **절대 영향을 주지 않는다**.

이 규칙은 코드 레벨에서 강제된다 (3.3, 6장 참조).

### 운영 흐름

GitHub → Google Colab 런타임에서 파이프라인/챗봇 실행 → ngrok 노출 → KakaoTalk 챗봇으로 사용자 응대.

---

## 2. 디렉터리 구조

```
src/
├── pipeline.py              # 메인 오케스트레이터 (1,442 LOC, run_pipeline)
├── pipeline_support.py      # 예측 프레임 빌드/스코어링 헬퍼
├── config/settings.py       # 중첩 dataclass 설정 + 검증 (AppConfig)
├── data/                    # 로드·정제·유니버스·실데이터 fetch·투자자 컨텍스트
├── features/                # 가격/기술/외부시장/레짐/투자시그널 피처 + 피처 선택
├── models/lgbm_heads.py     # 멀티헤드 LightGBM(+sklearn fallback) 모델
├── validation/              # walk-forward, 백테스트, 시그널 튜닝, 보정, 메트릭
├── domain/signal_policy.py  # 매수/매도/관망 정책, 이벤트 부스트, 리스크 플래그
├── inference/predict.py     # 예측→시그널 스코어 변환
├── reports/                 # 메타데이터·산출물·PM리포트·이슈요약·뉴스임팩트 컨텍스트
├── recommendation/          # 실시간 종가배팅 추천 (챗봇 '추천' 인텐트용)
├── chatbot/                 # 카카오/Colab 봇 (2,286 LOC) + 인텐트/응답
├── news_impact/             # 벤더드 stock-news-impact 패키지 (독립 실행 가능)
└── utils/                   # 원자적 파일 쓰기, 시크릿 레닥션, 결과 정리

configs/    # prod_conservative.json, research_balanced.json, news_impact 예시
data/       # sample_ohlcv.csv 등 입력
result/      # 생성 산출물 (latest/, runs/<run_id>/)
colab/      # Colab 실행 헬퍼
docs/        # 문서 (본 분석 포함)
tests/       # 49개 pytest 모듈
```

### 콘솔 진입점 (pyproject.toml)

| 명령 | 대상 |
|------|------|
| `stock-predict` | `src.pipeline:main` (메인 예측 파이프라인) |
| `stock-predict-kakao` | `src.chatbot.kakao_colab_bot:main` (카카오 봇) |
| `stock-news-impact` | `src.news_impact.run:main` (독립 뉴스임팩트 리서치) |

---

## 3. 파이프라인 아키텍처

`run_pipeline()` (`src/pipeline.py`)이 전 과정을 12스텝 진행률 출력과 함께 조율한다. `PipelineDiagnostics`가 단계별 타이밍/행수/상태/경고를 수집한다.

### 3.1 6개 핵심 단계

```
1. load_config_and_inputs    설정 로드 → OHLCV 로드 → clean_ohlcv → 품질 필터 → 유니버스 필터
2. prepare_context           투자자 수급/공시/뉴스 raw 이벤트 수집 (옵트인)
3. build_feature_matrix      가격피처 + 외부시장피처 + 레짐 + 투자시그널, target 결측 제거
4. validation_and_tuning     walk-forward OOF → 확률보정 → 시그널 가중 튜닝 → 홀드아웃 백테스트
5. train_final_and_predict_latest  최종 모델 학습 → 최신 행 예측 → 이슈요약/뉴스임팩트 컨텍스트 부착
6. save_pipeline_artifacts   CSV/JSON/모델/매니페스트 작성, 검증, latest 승격
```

### 3.2 데이터 흐름

```
real_ohlcv.csv
  → load_ohlcv_csv → clean_ohlcv (is_zero_volume / is_extreme_return 플래그로 품질 제외)
  → (옵션) add_investor_context_with_coverage  [foreign/institution net buy, 공시, 뉴스]
  → build_features              [가격/기술지표/수급 엔지니어드 피처 + target_log_return 등]
  → add_external_market_features [^KS11, ^IXIC, NQ=F, ^VIX, KRW=X, ^TNX ...]
  → annotate_market_regime
  → add_investment_signal_features
  → select_feature_columns      [display-only 컬럼 제외한 모델 입력 확정]
  → walk_forward_validate_result → OOF 예측
  → 확률 보정 + 시그널 가중 튜닝 (tune/eval 0.7 분할)
  → 최종 모델 fit (최근 final_model_lookback_days, 기본 3년) → 최신 예측
  → build_scored_prediction_frame → finalize → 정책/리스크/권고 부착
  → 산출물 작성
```

### 3.3 display-only 가드 (핵심 무결성 장치)

`src/features/feature_selection.py`:

- `DISPLAY_ONLY_CONTEXT_COLUMNS`: 뉴스/공시 관련 컬럼(`news_sentiment`, `disclosure_score`, `news_impact_*`, 복합 `investor_event_score` 등).
- `MODEL_FEATURE_COLUMN_BASE = FEATURE_COLUMN_BASE - DISPLAY_ONLY_CONTEXT_COLUMNS`.
- `select_feature_columns()`는 display-only 컬럼과 `news_impact_` 접두사 컬럼을 **모델 입력에서 명시적으로 제거**한다.
- 테스트 `test_display_only_feature_guard.py`가 이 경계를 회귀 보호한다.

→ 뉴스가 종가 예측에 새지 않음을 코드와 테스트 양쪽에서 보장.

---

## 4. 설정 시스템 (`config/settings.py`)

중첩 dataclass 기반 `AppConfig` (schema_version=1). JSON 파일(`--config-json`)과 CLI 오버라이드를 `_merge_dataclass_config`로 병합하고 `_validate_app_config`로 엄격 검증한다(미지의 키는 difflib 추천과 함께 에러).

| 섹션 | 핵심 기본값 |
|------|------------|
| `UniverseConfig` | KOSPI200, expected_size=200 |
| `FeatureConfig` | lookback [1,2,3,5,10,20,60], MA [5,10,20,60,120], vol [5,20,60], RSI 14 |
| `ExternalFeatureConfig` | ^KS11 ^KQ11 ^GSPC ^IXIC NQ=F ^SOX ^VIX KRW=X ^TNX |
| `TrainingConfig` | min_train 756, test 252, step 126, quantiles [0.1,0.5,0.9], purge_gap 1, final_model_lookback 756 |
| `SignalConfig` | return_weight 0.65, up_prob_weight 0.35, uncertainty_penalty 0.25 |
| `InvestmentCriteriaConfig` | top_turnover_rank 15, 고확신 순매수 1,000억, RSI 30~35 매수관찰/70 과매수, 나스닥 ±1% 임계 |
| `BacktestConfig` | top_k 20, 포트 10억, 참여율 0.10, fee 10bps, slippage 5bps, min_value_traded 30억 |

프리셋: `configs/prod_conservative.json`, `configs/research_balanced.json`.

---

## 5. 모델 (`models/lgbm_heads.py`)

`MultiHeadStockModel` — 독립 헤드 5개 이상을 한 번에 학습:

- **회귀** (`target_log_return`)
- **방향 분류** (`target_up`, `predict_proba`)
- **분위수** (기본 0.1/0.5/0.9) → 불확실성 폭

특징:
- LightGBM 우선, 없으면 sklearn GBDT fallback (메타데이터에 경고 기록).
- 결측 임퓨테이션: RSI/Stoch/CCI 등은 중립값, 나머지는 train 중앙값.
- `head_n_jobs > 1`이면 joblib thread 병렬(LightGBM은 GIL 해제).
- 분위수 정렬(`np.sort`)로 교차 방지.
- `save()`는 joblib 번들 + `.meta.json` 사이드카(시드/백엔드/feature hash). `load()`는 `artifact_version`과 feature hash 정합성 검증.
- `MODEL_ARTIFACT_VERSION = 2`.

---

## 6. 검증·백테스트 (`validation/`)

### Walk-Forward (`walk_forward.py`)

- 확장창(또는 `walk_forward_lookback_days` 슬라이딩) 폴드.
- **purge_gap_days**(기본 1) + **embargo_days**로 익일 타깃 누설 차단.
- 폴드 병렬: `ProcessPoolExecutor`, 병렬 시 모델 내부 스레드 1로 제한해 CPU 과점유 방지.
- OOF 중복(Date+Symbol)은 `date_symbol_mean` 정책으로 평균; target 충돌 시 에러, stable 컬럼 충돌은 진단에 기록.
- 폴드 수가 `min_required_folds`(기본 3) 미만이면 `_adaptive_training_cfg`로 윈도 축소 후 재시도.

### 시그널 튜닝/보정

- OOF를 tune/eval 0.7로 분할 (`split_oof_for_tuning_and_eval`).
- `fit_up_probability_calibrator`로 상승확률 보정 → eval 메트릭 기록.
- `tune_signal_weights`가 tune 분할에서 시그널 가중 최적화 → eval/최종 예측에 적용.
- 백테스트는 **eval 홀드아웃에서만** 수행(튜닝 데이터 누설 방지). `run_long_only_topk_backtest`, fee/slippage 모델, 커버리지 게이트, `evaluate_backtest_validity`로 유효성 판정 → 부적합 시 리포트 `status="warning"` + blocking_reasons.

---

## 7. 시그널 정책 (`domain/signal_policy.py`)

### 권고는 기대수익률만 사용 (가드레일 강제 지점)

```python
def recommendation_from_signal(...):
    if predicted_return > 2.0:  return "매수"
    if predicted_return <= -2.0: return "매도"
    return "관망"
```

다른 인자(상승확률 등)는 하위호환용으로 받지만 권고를 바꾸지 않는다.

### 부가 산출 (표시/포트폴리오 보조)

- **이벤트 부스트** (`vectorized_event_signal_boost`): 거래대금 상위, 외국인·기관 쌍방 순매수, 주도주 확인, 52주 신고가, RSI 풀백/과매수, 나스닥 순풍/역풍 → `signal_score`에 가감(권고가 아닌 시그널 스코어 한정).
- **리스크 플래그** (`_risk_flag_series`): COVERAGE_HALT, HIGH_UNCERTAINTY, LOW_UP_PROB, LOW_LIQUIDITY, MARKET_HEADWIND 등.
- **PM 요약**: portfolio_action(신규매수/관심관찰/비중축소/거래보류), trading_gate(정상/체결주의/보수모드/거래중단), position_size_hint, confidence_label.
- **종배 스코어 / prediction_reason**: 한국어 매매 근거 문자열.

row 단위 함수와 vectorized series 함수가 쌍으로 존재(테스트 하위호환 + 성능).

---

## 8. 뉴스 임팩트 모듈 (`news_impact/`)

`stock-news-impact` 형제 repo가 벤더드된 독립 패키지(약 5,000 LOC, 30+ 모듈). 두 실행 경로:

```
독립 (리서치 리포트)              통합 (표시 컨텍스트)
stock-news-impact CLI            run_pipeline --news-impact-report
  → run_daily_pipeline             → append_news_impact_context
  → report.json/csv + audit.json   → result_detail.csv 의 news_impact_* 컬럼
  (LLM 재현성 메타: model,           (display-only; select_feature_columns가
   temperature, prompt_hash)         news_impact_ 컬럼 전부 제거)
```

핵심 (`pipeline.py::run_daily_pipeline`):
- watchlist × 뉴스 클러스터로 LLM 임팩트 판정(`impact_judge`), 프롬프트 인젝션 탐지, summary-only 플래그.
- LLM 클라이언트는 OpenAI 기본/Gemma(llama.cpp) 옵션, `FileLLMResponseCache`로 캐시.
- 의미 클러스터링, 점수 집계, 랭킹.
- `RunAudit`: git_commit, config/watchlist/data 해시, LLM 메타, prompt_hash로 **재현성** 확보.
- LLM 시스템 프롬프트는 코드 자산으로 `src/news_impact/prompts/news_impact_llm_prompt.md`에 패키지 리소스화(최근 리팩터).

→ 양 경로 모두 review-only. 한국 뉴스 우선 수집 원칙.

---

## 9. 챗봇 (`chatbot/kakao_colab_bot.py`, 2,286 LOC)

가장 큰 단일 모듈. Flask 기반 카카오 웹훅 서버.

- `PipelineRuntimeConfig`로 백그라운드 예측 잡(`src/pipeline.py` subprocess) 커맨드 구성.
- 인텐트: 도움말 / 상태조회 / 최신화(쿨다운) / 추천 / 종목코드 인식(`\d{6}(.KS|.KQ)?`).
- `result/result_simple.csv`에서 캐시 예측 읽기, 없으면 백그라운드 잡 시작, 기본 종목 prewarm.
- 보안: HMAC 웹훅 서명, 허용 CIDR, 시크릿 레닥션(`utils/secrets.py`), 동시 잡 수 제한.
- **추천 인텐트**는 `recommendation/realtime_close_betting.py`의 실시간 종가배팅 서비스(라이브 OHLCV fetch → 기술지표/거래대금 랭크 → 후보 스코어)를 호출.
- 챗봇 응답의 뉴스/공시 요약도 표시 전용.

---

## 10. 산출물 라이프사이클 (`reports/run_artifacts.py`)

`RunArtifactManager`:
- 실행별 원본 → `result/runs/<run_id>/`, 공식 최신 → `result/latest/`.
- `run_id`는 안전 패턴 검증, 경로 탈출 방지(`path()`).
- 원자적 쓰기(`atomic_write_text`), CSV는 `utf-8-sig`.
- `manifest.json`: sha256/행수/컬럼/스키마버전 포함 아티팩트 목록.
- **승격 규칙**: status ∈ {pass, warning} **그리고** environment=production **그리고** data_mode=real 일 때만 `latest/` 교체 + 상위호환 CSV 복사. 샘플 smoke 실행은 운영 latest를 덮어쓰지 않는다.

`report_metadata.py`: run_id 생성, KRX 영업일 계산(`next_krx_business_day`, 2025~2026 공휴일 하드코딩), 캘린더 커버리지 만료 경고, config 해시, git commit.

산출물: `result_detail.csv`, `result_simple.csv`(카카오용), `result_news.csv`, `result_disclosure.csv`, `pm_report.json`, `pipeline_report.json`, 모델 아티팩트 + feature importance.

---

## 11. 테스트 (49개 모듈)

행위 영역별로 촘촘함:
- 파이프라인 smoke, 데이터/피처/모델 레이어 하드닝
- 시그널 정책(계약/추천/이벤트부스트/근거), 백테스트·보정 가드
- **display-only 피처 가드**, 피처 모듈 경계
- 뉴스임팩트 풀패키지/감사/LLM캐시/프롬프트안전
- 결과 검증/정리, 산출물 하드닝, 패키징 메타데이터
- 챗봇 헬퍼/Colab 러너, 시크릿 레닥션, 실시간 종가배팅

`pytest` cache는 `result/.pytest_cache`로 격리.

---

## 12. 설계 패턴 관찰

**강점**
- 가드레일(예측 시그널 격리)이 코드 + 테스트로 이중 강제됨.
- 누설 방지에 진심: purge/embargo, tune/eval 분할, 홀드아웃 전용 백테스트.
- 재현성: 시드 고정, config/feature/LLM 해시, 모델 사이드카 메타.
- 운영 견고성: 원자적 쓰기, 승격 게이트, 커버리지 게이트, 진단 수집, 광범위한 예외 → 경고 강등.
- `pipeline.py`의 다수 `_xxx` 함수는 하위호환 wrapper로 실제 로직은 도메인 모듈에 위임(테스트 안정성 유지).

**복잡도/주의점**
- `pipeline.py`(1,442 LOC), `chatbot/kakao_colab_bot.py`(2,286 LOC), `reports/issue_summary.py`(730 LOC)가 큰 편 — 추가 분해 여지.
- row-단위와 series-단위 정책 함수가 병존(의도적 하위호환이나 중복 표면적 존재).
- `news_impact`가 자체 LLM/스키마/백테스트 스택을 들고 와 메인과 별개 하위세계를 형성.

---

## 13. 발견된 정합성 이슈

1. **고아(dead) 코드 — `pipeline.py::_save_pipeline_figures`** (719~746행)
   - `resolve_output_dir`, `save_backtest_figures`, `save_signal_histogram`, `save_actual_vs_predicted_plot`, `save_diagnostic_figures`, `save_symbol_level_comparison_figures` 등을 호출하지만 이 심볼들은 `pipeline.py`에 **import되어 있지 않다**(그래프 생성 제거 작업의 잔재; 관련 plan `2026-06-05-remove-graph-generation`).
   - `run_pipeline`에서 **호출되지 않으므로** 현재 동작에는 영향 없으나, 호출되면 `NameError`가 난다. → 제거 권장.

2. **문서 인덱스 불일치 — `docs/README.md` & `README.md`**
   - 두 파일 모두 `docs/INDEX.md`, `docs/01_pipeline.md` ~ `docs/10_config.md`, `docs/ROADMAP.md`를 링크하지만 이 파일들은 **현재 존재하지 않는다**(`docs/`에는 README, 본 분석, TIMA 2종, archive/, superpowers/만 존재). `git status`상 `docs/ROADMAP.md`는 삭제 상태.
   - numbered 정규 문서 세트가 제거/이동되었으나 인덱스가 따라가지 못함. → README의 Documentation 섹션 갱신 필요.

3. **README CLI 옵션 일부 누락**
   - 실제 파서에는 `--news-impact-report`, `--walk-forward-n-jobs`, `--model-n-jobs`, `--model-head-n-jobs`, `--context-raw-event-n-jobs`, `--issue-summary-n-jobs` 등 병렬/뉴스임팩트 옵션이 있으나 README "Main CLI Options"에는 일부만 기술됨.

---

## 14. 빠른 시작 (참고)

```powershell
# 설치
python -m pip install -r requirements.txt
python -m pip install -e .

# 샘플 스모크 (외부 다운로드 없음)
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json

# 실데이터 fetch 후 실행
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv --real-symbols 005930.KS 000660.KS

# 테스트
pytest
pytest tests/test_pipeline_smoke.py

# 최신 산출물 매니페스트
Get-Content -Encoding utf8 result/latest/manifest.json
```
