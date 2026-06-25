# 코드베이스 분석 (CODEBASE_ANALYSIS)

> 작성: 2026-06-23 · 기준 브랜치 `ops-parallelism-report` (HEAD `31da0eb`)
> 방법: `src/` 핵심 모듈을 직접 정독해 사실 검증 후 작성. 테스트 실행 결과는 주장하지 않는다.
> 규모: `src/` 파이썬 **94개 파일**(약 17K LOC), `tests/` **54개 모듈**.
>
> 본 문서는 `src/` 핵심 모듈을 직접 정독한 **종합 레퍼런스 겸 개선 관찰**이며, `docs/`의 단일
> 코드베이스 문서다. 본문(§1~§15)은 작성 시점 정독 스냅샷이고, §16(개선 관찰)은 2026-06-25
> `docs/gemma-iq4xs-model` 기준으로 코드와 다시 대조해 수치·범위를 갱신했다.

---

## 1. 한 줄 요약

KOSPI200 약 200종목에 대해 **다음 거래일 종가 기준 기대수익률(`predicted_return`)**을 예측하고,
워크포워드 검증 → 확률 보정 → 신호 가중치 튜닝 → 롱온리 top-k 백테스트를 거쳐 CSV/JSON 산출물을
`result/`에 남기는 **투자 참고용(자동매매·자문 아님)** 파이썬 파이프라인. 운영은
**GitHub → Colab → KakaoTalk** 경로로 서빙된다.

---

## 2. 절대 가드레일 — 코드로 강제되는 지점

> **매수/매도/관망 권고는 다음 거래일 기대수익률(`predicted_return`)만으로 결정한다.
> 뉴스·공시는 화면 표시용 컨텍스트일 뿐, 기대수익률·순위·권고·신호를 절대 바꾸지 않는다.**

이 규칙은 선언이 아니라 다음 3중 방어로 코드에 박혀 있다(직접 확인):

1. **피처 선택 가드** — `src/features/feature_selection.py`
   `is_display_only_context_column()`이 ① 명시 집합 `DISPLAY_ONLY_CONTEXT_COLUMNS`(뉴스/공시
   점수 + 복합 `investor_event_score`), ② 접두 `news_`/`disclosure_`, ③ 부분문자열
   `_news_`/`_impact_`를 **패턴 기반**으로 제거한다. `select_feature_columns()`는 이 가드를
   통과한 컬럼만 모델 입력으로 넘긴다 → 신규 `*_impact_*` 컬럼이 추가돼도 자동 차단.
2. **권고 산출** — `src/domain/signal_policy.py::_recommendation_series()`
   오직 `predicted_return`과 설정 임계값만 본다. `signal_score`·`up_probability`·뉴스는
   시그니처에 받더라도 권고를 바꾸지 못한다(하위호환용 인자).
3. **회귀 테스트** — `test_display_only_feature_guard.py`, `test_feature_module_boundaries.py`,
   `test_signal_policy_contract.py` 등이 경계를 고정한다.

---

## 3. 진입점 (`pyproject.toml [project.scripts]`)

| 콘솔 명령 | 진입 함수 | 역할 |
|------|------|------|
| `stock-predict` | `src.pipeline:main` | 메인 예측 파이프라인 |
| `stock-predict-publish` | `src.ops.publish_predictions:main` | 기본 200종목 예측을 `published/`에 게시 + git push |
| `stock-predict-kakao` | `src.chatbot.kakao_colab_bot:main` | Colab/Kakao 챗봇 Flask 서버 |
| `stock-news-impact` | `src.news_impact.run:main` | 독립 뉴스 임팩트 리포트 CLI |

---

## 4. 디렉터리 구조 (`src/`)

| 패키지 | 책임 | 핵심 파일(LOC) |
|--------|------|------|
| `config/` | `AppConfig` 중첩 데이터클래스 + JSON 로딩/엄격 검증 | `settings.py`(314) |
| `data/` | OHLCV 로딩/정제, yfinance fetch, KRX 유니버스, 투자자 컨텍스트 | `investor_context.py`(435), `fetch_real_data.py`(314) |
| `features/` | 가격/기술/외부시장/투자자/레짐 피처 + **피처 선택 가드** | `price_features.py`(245), `feature_selection.py`(151) |
| `models/` | `MultiHeadStockModel` (LightGBM ↔ sklearn 폴백) | `lgbm_heads.py`(422) |
| `validation/` | 워크포워드, 보정, 신호 튜닝, 백테스트, 유효성 강등 | `backtest.py`(336), `walk_forward.py`(306) |
| `domain/` | 신호 정책 — 권고/리스크/이벤트부스트 (가드레일 강제 지점) | `signal_policy.py`(437) |
| `inference/` | 예측 프레임·신호 스코어 조립 | `predict.py`(51) |
| `reports/` | 산출물 포맷, run 매니페스트, 이슈요약, 뉴스임팩트 부착 | `issue_summary.py`(730), `news_impact_context.py`(449) |
| `recommendation/` | **별도** 실시간 종가베팅 규칙 스코어러 | `close_betting.py`(243) |
| `news_impact/` | 벤더링된 LLM 기반 뉴스 임팩트 패키지(표시 전용) | `pipeline.py`(623), `llm_client.py`(379) |
| `ops/` | `published/` 게시 스토어 + publish CLI | `publish_predictions.py`(257) |
| `chatbot/` | Kakao/Colab 봇 + 추출된 헬퍼 | `kakao_colab_bot.py`(2100) |
| `utils/` | 원자적 파일쓰기, 시크릿 레다크션, result 정리 | `result_cleanup.py`(206) |

`src/pipeline.py`(1473)와 `src/pipeline_support.py`(122)가 루트의 오케스트레이터다.

---

## 5. 메인 파이프라인 아키텍처 (`src/pipeline.py`)

`run_pipeline()`은 12스텝 진행로그를 출력하지만, 내부적으로 **6개 단계**(`PIPELINE_STAGE_KEYS`)로
묶이고 각 단계는 `PipelineDiagnostics`가 타이밍·행수·상태(ok/caution)·경고를 기록한다.
각 단계는 전용 헬퍼(`_load_pipeline_config_and_data`, `_prepare_pipeline_context`,
`_build_pipeline_feature_matrix`, `_run_pipeline_validation`, `_predict_pipeline_latest`,
`_write_pipeline_artifacts`)로 위임된다.

```
1) load_config_and_inputs       AppConfig 로드 → OHLCV 로딩/정제(품질 컬럼 제외) → 유니버스 필터
2) prepare_context              투자자 흐름/공시/뉴스 raw 이벤트 수집(옵션, 실패 시 폴백·커버리지 기록)
3) build_feature_matrix         가격+기술+외부시장+레짐+투자자 신호 피처 → 타깃 생성 → 피처 선택
4) validation_and_tuning        워크포워드 OOF → 확률 보정 → tune/eval 분할
                                → 신호 가중치 튜닝(tune split) → 백테스트(eval split)
5) train_final_and_predict_latest  최종 모델 학습 → 최신 행 예측 → 권고/리스크/이슈요약/뉴스임팩트 부착
6) save_pipeline_artifacts      detail/simple/news/disclosure CSV, 모델 pkl+meta, pm_report.json,
                                pipeline_report.json, 매니페스트(+ latest 승격)
```

**누수 방지 설계(직접 확인):**

- 타깃 `target_log_return = log(close[t+1]/close[t])` (`price_features.build_features`).
- 워크포워드 `_iter_fold_windows`가 `purge_gap_days`(기본 1)로 train 끝 인덱스를 뒤로 당기고
  `embargo_days`(기본 0)로 검증 시작을 밀어 다음날 타깃 겹침을 차단한다.
- **튜닝-평가 분리**: `_split_oof_for_tuning_and_eval`로 OOF를 tune(0.7)/eval로 시간 분할.
  신호 가중치는 tune에서만 튜닝(`tune_signal_weights`), 백테스트는 건드리지 않은 eval에서만.
- **확률 보정도 분리**: isotonic 보정기를 tune에서 학습해 eval과 최신 예측에 적용.

**적응 재시도**: 폴드 수가 `min_required_folds`(기본 3) 미만이면 `_adaptive_training_cfg`로
창을 데이터 길이에 맞춰 줄여 워크포워드를 **1회 재시도**한다(짧은 데이터 대응).

**호환 래퍼 잔존**: `pipeline.py` 상단(약 100~320행)에 다른 모듈로 위임만 하는 `_`-접두 래퍼가
다수 남아 있다(`_recommendation_from_signal`, `_apply_event_signal_boost`, `_split_oof_*` 등).
대부분 기존 테스트/임포트 호환용이다(→ §13 개선 관찰).

---

## 6. 설정 시스템 (`src/config/settings.py`)

중첩 데이터클래스 `AppConfig`(universe/feature/external/training/signal/
investment_criteria/backtest)가 단일 출처. `load_app_config(path, overrides)`가 JSON 머지 후
`_validate_app_config`로 **엄격 검증**한다.

- 알 수 없는 키는 `difflib.get_close_matches`로 오타 후보까지 제시하며 **거부**.
- 도메인 규칙 강제: `min_train_size > test_size`, `step_size <= test_size`, 퀀타일 3개 이상·증가·(0,1),
  RSI 임계값 단조성(`buy_watch_low ≤ buy_watch_high ≤ overbought`), 나스닥 headwind < tailwind.
- **권고 임계값이 설정에 있음**: `SignalConfig.recommendation_buy_threshold_pct`(기본 +2.0),
  `recommendation_sell_threshold_pct`(기본 -2.0). 검증은 buy>0, sell<0, **sell < buy**를 강제.
- 프리셋: `configs/prod_conservative.json`, `configs/research_balanced.json`.

---

## 7. 피처 엔지니어링 (`src/features/`)

`build_features()`(`price_features.py`)가 핵심. 종목별 그룹 연산으로:

- **가격/기술**: 로그수익률, 일간/갭/장중 수익률, range%, 거래대금·거래대금순위(일별),
  상위 3/10 플래그, 다중창 수익률·이동평균·`close_to_ma`·변동성, `vol_ratio_20`,
  RSI/MACD/Stochastic/CCI/ATR/OBV(`technical_indicators.compute_technical_indicator_block`).
- **투자자 컨텍스트 파생**: 외국인/기관 매수 신호·비율·z-score(20일)·누적(3·5일),
  52주 신고가/근접/돌파, 주도주 확인 플래그, 가격제한폭 상하한 히트(2015-06-15 전후 0.15/0.30).
- **표시 전용**: `news_positive/negative_signal`, `investor_event_score`(공시·뉴스 포함 복합점수) —
  피처 가드가 모델 입력에서 제거.
- 한국어/영문 컬럼 별칭을 정규화(`외국인순매수`→`foreign_net_buy` 등)하고 원시 입력 컬럼은
  `drop_source_cols`로 정제. 결측 지표는 `*_missing` 플래그 생성, RSI/Stoch/MACD는 중립값으로 채움.

`feature_selection.select_feature_columns()`가 접두 화이트리스트 + `MODEL_FEATURE_COLUMN_BASE`로
모델 피처를 고정(표시 전용 제외). 외부시장 피처는 커버리지와 함께 수집되며 실패 시 폴백.

> 주의: `select_feature_columns`는 `feature_selection.py`에 정의되고 `price_features.py`가
> 재노출한다. `pipeline.py`는 `price_features`에서 임포트 — 동일 함수, 경로만 두 갈래.

---

## 8. 모델 (`src/models/lgbm_heads.py`)

`MultiHeadStockModel` = **회귀 + 이진 방향분류 + 분위수(최소 3개)** 멀티헤드.

- LightGBM 있으면 사용, 없으면 sklearn GBDT 폴백 + `SKLEARN_BACKEND_WARNING`(비등가 경고)를
  메타데이터에 기록. `backend` 필드로 추적.
- 결측 임퓨트: 학습셋 중앙값, 단 RSI/Stoch/CCI는 중립 기본값(`NEUTRAL_FEATURE_DEFAULTS`).
- 헤드 병렬: LightGBM이 C++ 구간에서 GIL을 풀어 **스레드 병렬**(`head_n_jobs`, `prefer="threads"`).
- 분위수 교차 방지: `predict()`가 선택된 3개 분위수를 `np.sort`로 정렬.
- 영속화: joblib 번들 + `.meta.json` 사이드카(seed·backend·`feature_hash`). 로드 시
  `MODEL_ARTIFACT_VERSION`(2) 일치와 **피처 해시 일치**를 강제해 깨진 모델 사용을 차단.

---

## 9. 검증·튜닝·백테스트 (`src/validation/`)

- **`walk_forward.py`** — 확장창(또는 `walk_forward_lookback_days` 슬라이딩) 워크포워드.
  `walk_forward_n_jobs > 1`이면 `ProcessPoolExecutor`로 폴드 병렬 + 각 모델 내부 스레드를 1로
  고정(`replace(cfg, model_n_jobs=1, model_head_n_jobs=1)`)해 CPU 과점유 방지.
  OOF는 `aggregate_oof_predictions`가 (Date,Symbol) 평균으로 중복 정리하고 충돌 진단을 남긴다.
- **`support.py`** — OOF tune/eval 시간 분할, isotonic 상승확률 보정기(붕괴 시 0.3·cal+0.7·raw로
  완충), OOF 방향정확도·불확실성 진단.
- **`signal_tuning.py`** — tune 분할에서 그리드 탐색. 목적함수 = 상위decile 수익 + 0.1·rankIC
  − 0.25·downside. 일반화 갭이 0.15 초과하고 기본가중치가 허용오차 내면 **기본값으로 폴백**.
- **`backtest.py`** — 롱온리 top-k. 유동성/거래대금/시장유형 한도, 회전율 제한, 정적+동적 비용
  (불확실성·변동성 기반), 커버리지 halt 시 당일 0수익 처리, 보수/중립/공격 슬리피지 시나리오.
- **`result_validity.py`** — 백테스트 표본/거래가능 종목이 부족하면 리포트를 `status=warning` +
  `blocking_reasons`로 강등.

---

## 10. 신호 정책 (`src/domain/signal_policy.py`) — 결정 로직의 심장

**권고(가드레일 강제):**

```python
predicted_return >  recommendation_buy_threshold_pct(+2.0)  → "매수"
predicted_return <= recommendation_sell_threshold_pct(-2.0) → "매도"
그 외 / NaN                                                  → "관망"
```

`predicted_return`은 `inference/predict.py`에서 `expm1(log_return)*100`으로 이미 **퍼센트**.

**신호 스코어(순위/백테스트 선정용, 권고와 분리):**

```
signal_score = return_weight·norm_return + up_prob_weight·up_probability
             − uncertainty_penalty·uncertainty_score + event_boost_score
```

`event_boost_score`(`vectorized_event_signal_boost`)는 거래대금 상위·외국인/기관 동반순매수·
52주 신고가·RSI·나스닥 선물 등 **수급/기술 이벤트**에서 나온다(뉴스 아님). 순위엔 영향, 권고는
`predicted_return`에 묶여 가드레일 유지.

그 외 산출: `risk_flag`(COVERAGE_HALT/HIGH_UNCERTAINTY/LOW_LIQUIDITY/MARKET_HEADWIND 등),
`confidence_label`, `position_size_hint`, `portfolio_action`, `trading_gate`,
`jongbae_score`/`jongbae_signal`, 한국어 `prediction_reason`.

> **행단위 ↔ 벡터화 통합 확인**: `risk_flag(row)`, `_jongbae_score(row)`,
> `build_pm_summary_fields(row)` 등 행단위 함수는 모두 `_row_frame(row)`로 1행 프레임을 만들어
> `*_series` 벡터화 경로를 호출하는 **얇은 어댑터**로 정리되어 단일 출처를 이룬다.

---

## 11. 산출물 라이프사이클 (`src/reports/run_artifacts.py`)

- `RunArtifactManager`가 실행별 원본을 `result/runs/<run_id>/`에 쓰고(원자적 쓰기, 경로 탈출 차단),
  `finalize()`에서 매니페스트(파일별 sha256·행수·컬럼·스키마버전) 생성 + 필수 산출물 검증.
- **승격 규칙**: `status ∈ {pass, warning}` **AND** `environment == production` **AND**
  `data_mode == real`일 때만 `promoted=True` → `result/latest/`로 **원자적 디렉터리 교체**
  (백업/롤백 포함) + 최상위 호환 CSV 복사. **샘플/스모크 실행은 latest를 절대 덮지 않는다.**
- 매니페스트(`promoted`, `status`, `blocking_reasons`)가 "운영 산출물 여부"의 단일 판단 근거.
- CSV는 Excel/Windows 호환 위해 `utf-8-sig`.
- **메타데이터 계약**(`report_metadata.py`): run_id, environment, data_mode, input/prediction/
  context 일자, git_commit, config_hash, KRX 영업일 계산(`next_krx_business_day`), 캘린더 만료 경고.
- **보존·정리**: `utils/result_cleanup.py` docstring이 단일 출처(성공 run 10개·30일, 실패 30일,
  로그 14일, latest 보호). 자동 호출 안 됨 — 운영자/스케줄러가 명시 호출.

---

## 12. 뉴스 임팩트 / 서빙 / 독립 추천 엔진

**뉴스 임팩트(`src/news_impact/`)** — 형제 저장소를 벤더링한 LLM 파이프라인(로컬 gemma/llama.cpp
`localhost:8001` 또는 OpenAI). `run_daily_pipeline()`이 dedupe→클러스터→LLM 임팩트 판정→집계→
랭킹→`report.json/csv`+`audit.json`(모델/온도/프롬프트 해시 재현 메타). LLM 응답은 파일 캐시,
실패는 카운트만 올리고 진행. 메인 파이프라인 통합 시 `news_impact_*` 컬럼은 **표시 전용**으로
부착되고 피처 가드가 제거. 통합 경로 메타(`news_impact_runtime`: requested/actual mode,
fallback 여부/이유)가 리포트에 기록된다.

**서빙(publish → Colab → Kakao):**
- `ops/publish_predictions.py` — 200종목 파이프라인 → `ensure_operational_manifest()` 통과 시
  `published/latest/` + `published/history/<거래일>/` 복사 → `index.json` 갱신 → (옵션) git
  add/commit/push. `_runtime_meta_from_report`가 **실제 사용된 뉴스 모드·폴백을 `publish_meta`에
  표면화**(요청 gemma인데 무응답이면 rule 폴백이 메타에 드러남).
- `colab/stock_predict_colab.py` — `load_published_predictions()`로 GitHub 기준데이터를
  **파이프라인 미실행** 서빙. `run_colab_pipeline()`은 사용자가 명시 호출할 때만.
- `chatbot/kakao_colab_bot.py` — Flask 봇. 기본 `published/latest/` 기준 응답, 종목코드/이름 입력
  또는 "최신화" 시에만 해당 종목을 **세션 한정** 재예측(GitHub push 없음).

**독립 추천 엔진(`recommendation/close_betting.py`)** — ML 파이프라인과 **완전 독립**. 라이브
OHLCV 거래대금 상위 N을 신고가·이평·거래량급증·캔들로 점수화하는 규칙기반 "종가베팅" 스코어러.
챗봇의 추천 요청 핸들러가 사용. **스키마가 다름**(소문자 `symbol/date/close/...` vs ML의
`Symbol/Date/Close/...`) — 별도 시스템임에 유의.

---

## 13. 테스트 (`tests/`, 54개 모듈)

결정론적 단위/통합 테스트가 가드레일·계약을 촘촘히 고정한다:

- **가드레일/계약**: display-only 가드, 피처 모듈 경계, 시그널 정책 계약/권고/이벤트부스트/사유,
  확률보정 가드.
- **검증/모델**: 워크포워드, 백테스트+보정, 모델 영속화, 추론, 신호 튜닝, 결과 유효성.
- **운영/하드닝**: 캐시/파일 하드닝, result 정리(+하드닝), run 아티팩트, 리포트 메타, 출력 하드닝,
  시크릿 레다크션, 패키징 메타, publish/published store, 운영 하드닝.
- **데이터/피처**: 외부피처, 투자자 컨텍스트(통합/뉴스), 투자 신호, 레짐, 유니버스, fetch 폴백, KRX명.
- **뉴스 임팩트**: audit, context, fixture, full package, LLM 캐시/config, 프롬프트 안전성.
- **챗봇**: 헬퍼, job store, session store, 통합 봇, Colab 러너, 콘솔 요약.

원칙: 외부 네트워크는 모킹/비활성화. (검증 필요 시 `pytest tests --basetemp <임시폴더>` 실행 —
bare `pytest`는 `result/` ACL로 실패 가능.)

---

## 14. 데이터 흐름 (텍스트 다이어그램)

```
OHLCV CSV ──clean──▶ universe filter ──▶ (옵션) investor/news raw events
   │                                          │
   └──────────────── build_features ──────────┘
                          │  target_log_return = log(close[t+1]/close[t])
                          ▼
                 select_feature_columns  ──(news/disclosure 제거)──▶ model features
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                               ▼
  walk-forward OOF   isotonic 보정(tune)            최종 모델 학습(전체/lookback)
        │                 │                               │
        ▼                 ▼                               ▼
  tune/eval 분할 ──▶ signal weights(tune) ──▶ backtest(eval)   최신행 predict
                                                            │
                          ┌─────────────────────────────────┘
                          ▼
         build_scored_prediction_frame → event boost → 권고/리스크/사유
                          │
                          ▼   (issue summary / news_impact_* 표시전용 부착)
            result_detail/simple/news/disclosure.csv + pm_report.json
            + pipeline_report.json + manifest ──(production·real·pass)──▶ result/latest/
```

---

## 15. 강점

- 가드레일이 문서가 아니라 **코드(패턴 기반 가드)+테스트**로 3중 강제된다.
- 단계별 진단/커버리지 게이트/매니페스트로 운영 관측성이 높다.
- 엄격한 설정 검증(오타 후보 제시 포함)으로 잘못된 설정을 조기 차단.
- 모델 영속화에 버전·피처 해시 검증, 산출물 승격에 원자적 교체+롤백.
- 누수 방지(purge/embargo, tune/eval 분리, 보정 분리)가 설계에 일관되게 반영.
- 외부 의존(yfinance/DART/Naver/LLM)마다 폴백·커버리지 표기·런타임 모드 표면화.

---

## 16. 개선 관찰 (2026-06-25 코드 재확인 기준)

**초기 개선안 중 이미 코드에 반영 완료된 것:**
- ✅ 권고 임계값이 `SignalConfig`로 설정화 + 단조성 검증(`sell < buy`) 완료.
- ✅ 신호 정책 행단위/벡터화 **단일 출처화**(행단위는 1행 어댑터).
- ✅ display-only 가드의 **패턴 기반**(접두/부분문자열) 단언 반영.
- ✅ publish의 **실제 뉴스 모드/폴백 표면화**(`news_impact_runtime` → `publish_meta`).
- ✅ 병렬 워커 수 resolved 보고(`PipelineDiagnostics.set_parallelism`).

**남은 관찰(우선순위):**
- **P1 · 챗봇 god class** — `KakaoColabPredictionBot`이 약 1,990 LOC·메서드 110개로 HTTP·인텐트·
  포맷·서브프로세스 잡·부트스트랩 prewarm·라이브 이벤트·이슈요약·세션을 모두 떠안는다.
  `intent.py`/`job_store.py`/`session_store.py`/`message_formatter.py`/`responses.py`/`runtime_config.py`로
  **분해가 진행**됐으나 본체는 여전히 비대. 책임별 협력 객체로 점진 분해 권장.
- **P1 · `pipeline.py` 호환 래퍼** — 상단(약 226~298행)의 `_`-접두 위임 래퍼가 다수 잔존
  (`_recommendation_from_signal`·`_apply_event_signal_boost`·`_split_oof_for_tuning_and_eval` 등).
  테스트가 원본 모듈을 직접 임포트하도록 옮긴 뒤 제거하면 오케스트레이터 가독성↑.
- **P2 · KRX 휴일 캘린더 만료** — `report_metadata.KOREA_MARKET_HOLIDAYS`가 **2026-12-31까지**
  하드코딩(`KOREA_MARKET_HOLIDAY_COVERAGE_END = max(...)`). 만료/근접 경고 로직은 있으나, 2027 진입
  전 캘린더 갱신이 필요한 **연례 유지보수 부채**(2026-06-25 재확인 시점 기준 반년 내 만료 임박).
- **P2 · 매직넘버** — `confidence_label` 구간(0.34/0.67/0.80), 이벤트부스트 상수, risk_flag 임계값
  (0.75/0.5/0.45 등)이 모듈 상수. 프리셋별 보정/검증 대상이 아니다. 최소 신뢰도 구간은 설정화 검토.
- **P2 · `close_betting` 스키마 분기** — 추천 엔진이 소문자 OHLCV 스키마를 쓰는 점이 ML 파이프라인
  (Title-case)과 달라 데이터 어댑터 실수 여지가 있다. 경계/변환을 한곳에 명문화 권장.
- **P3 · 이중 임포트 경로** — `select_feature_columns`가 `feature_selection`/`price_features`
  양쪽에서 노출. 동일 함수지만 임포트 출처를 하나로 수렴하면 혼란↓.

---

## 17. 변경 시 체크리스트

- [ ] 뉴스/공시 신호가 `predicted_return`·권고·`signal_score`·순위에 들어가지 않는가?
- [ ] 새 컬럼이 모델 피처면 화이트리스트, 표시용이면 display-only(접두/부분문자열 포함)에 정확히 들어갔는가?
- [ ] 임계값/가중치 변경 시 `_validate_app_config` 검증과 관련 테스트를 갱신했는가?
- [ ] 행단위·벡터화 양쪽 정책 결과가 일치하는가(어댑터 단일 출처 유지)?
- [ ] 산출물 변경 시 매니페스트/리포트 계약 테스트(`test_report_metadata`, `test_run_artifacts`)를 확인했는가?
- [ ] `pytest tests --basetemp <임시폴더>`(최소 영향 테스트 + `test_pipeline_smoke.py`)를 통과하는가?
```
