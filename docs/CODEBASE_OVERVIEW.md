# 코드베이스 개요 및 개선안

> 현재 기준(branch 스냅샷) `src/` 전체를 읽고 정리한 **유지보수용 종합 레퍼런스**다.
> 날짜 스냅샷이 아니라 코드가 바뀌면 갱신하는 살아있는 문서를 의도한다.
> 규모: `src/` 파이썬 94개 파일·약 17,000 LOC, 테스트 53개 모듈.

## 0. 절대 가드레일 (먼저 읽을 것)

이 저장소의 모든 설계는 다음 한 줄에 종속된다.

> **매수/매도/관망 권고는 다음 거래일 기대수익률(`predicted_return`)만으로 결정한다.
> 뉴스·공시는 화면 표시용 컨텍스트일 뿐, 기대수익률·순위·권고·신호를 절대 바꾸지 않는다.**

이 규칙은 문서상 선언이 아니라 코드로 강제된다 (`src/features/feature_selection.py`):

- `select_feature_columns()`가 `DISPLAY_ONLY_CONTEXT_COLUMNS`(뉴스/공시 점수 일체)와
  `news_impact_*` 접두 컬럼을 **모델 피처에서 제거**한다.
- 권고는 `src/domain/signal_policy.py`에서 `predicted_return` 임계값만으로 산출된다.
- 회귀 테스트(`tests/test_display_only_feature_guard.py`, `test_feature_module_boundaries.py`)가
  이 경계를 지킨다.

---

## 1. 제품 목적과 진입점

투자자 **참고용** 프로그램이다. 자동매매·투자자문이 아니다. KOSPI200 약 200종목에 대해
다음 거래일 종가 기준 기대수익률을 예측하고, 워크포워드 검증·롱온리 top-k 백테스트를 거쳐
CSV/JSON 산출물을 `result/`에 남긴다. 운영은 GitHub → Colab → KakaoTalk 경로로 서빙된다.

콘솔 진입점 (`pyproject.toml [project.scripts]`):

| 명령 | 진입 | 역할 |
|------|------|------|
| `stock-predict` | `src.pipeline:main` | 메인 예측 파이프라인 |
| `stock-predict-publish` | `src.ops.publish_predictions:main` | 기본 200종목 예측을 `published/`에 게시 + git push |
| `stock-predict-kakao` | `src.chatbot.kakao_colab_bot:main` | Colab/Kakao 챗봇 서버(Flask) |
| `stock-news-impact` | `src.news_impact.run:main` | 독립 뉴스 임팩트 리포트 CLI |

---

## 2. 디렉터리 구조 (`src/`)

| 패키지 | 책임 |
|--------|------|
| `config/` | `AppConfig` 데이터클래스 + JSON 로딩/검증 |
| `data/` | OHLCV 로딩/정제, yfinance fetch, KRX 유니버스, 투자자 컨텍스트 수집 |
| `features/` | 가격/기술적/외부시장/투자자/레짐 피처, **피처 선택 가드** |
| `models/` | `MultiHeadStockModel` (LightGBM ↔ sklearn fallback) |
| `validation/` | 워크포워드, 베이스라인, 확률 보정, 신호 가중치 튜닝, 백테스트 |
| `domain/` | 신호 정책 — 권고/리스크플래그/이벤트 부스트 (가드레일 강제 지점) |
| `inference/` | 예측 프레임·신호 스코어 조립 |
| `reports/` | 산출물 포맷, run 아티팩트 매니페스트, 이슈요약, 뉴스임팩트 컨텍스트 부착 |
| `recommendation/` | **별도** 실시간 종가베팅 규칙 스코어러 |
| `news_impact/` | 벤더링된 LLM 기반 뉴스 임팩트 패키지(표시 전용) |
| `ops/` | `published/` 게시 스토어 + publish CLI |
| `chatbot/` | Kakao/Colab 봇 (단일 대형 클래스) |
| `utils/` | 원자적 파일쓰기, 시크릿 레다크션, result 정리 |

---

## 3. 메인 파이프라인 아키텍처 (`src/pipeline.py`)

`run_pipeline()`은 6개 단계(`PIPELINE_STAGE_KEYS`)로 구성되고, 각 단계는
`PipelineDiagnostics`가 타이밍·행수·상태(ok/caution)·경고를 기록한다.

```
1) load_config_and_inputs       AppConfig 로드 → OHLCV 로딩/정제 → 유니버스 필터
2) prepare_context              투자자 흐름/공시/뉴스 raw 이벤트 수집(옵션, 실패시 폴백)
3) build_feature_matrix         가격+기술적+외부시장+레짐+투자자 신호 피처 → 타깃 생성
4) validation_and_tuning        워크포워드 OOF → 확률 보정 → tune/eval 분할
                                → 신호 가중치 튜닝(tune split) → 백테스트(eval split)
5) train_final_and_predict_latest  최종 모델 학습 → 최신 행 예측 → 권고/리스크/이슈요약 부착
6) save_pipeline_artifacts      detail/simple/news/disclosure CSV, 모델 pkl,
                                pm_report.json, pipeline_report.json, manifest
```

데이터 흐름의 핵심 원칙:

- **누수 방지**: 타깃은 `target_log_return = log(close[t+1]/close[t])`. 워크포워드는
  `purge_gap_days`(기본 1)로 train 끝과 검증 시작 사이를 비운다.
- **튜닝 분리**: 신호 가중치는 OOF의 tune 분할에서만 학습하고, 백테스트는
  건드리지 않은 eval 분할에서만 돌린다 (튜닝-평가 누수 차단).
- **커버리지 게이트**: 외부/투자자 피처 수집 성공률이 임계 미만이면
  `coverage_gate_status`가 `halt`로 내려가 리스크 플래그/거래중단으로 표시된다.

설계 메모: `pipeline.py` 상단(약 100~330행)에는 다른 모듈로 위임만 하는
얇은 `_`-접두 래퍼 함수가 다수 있다. 대부분 "기존 테스트/임포트 호환"을 위해 남긴 것이다
(→ 9.2 개선안).

---

## 4. 설정 시스템 (`src/config/settings.py`)

중첩 데이터클래스 `AppConfig`(universe/feature/external/training/signal/
investment_criteria/backtest)로 단일 출처를 둔다.

- `load_app_config(path, overrides)` — JSON 머지 후 **엄격 검증**.
  알 수 없는 키는 `difflib`로 오타 후보까지 제시하며 거부한다.
- 검증이 도메인 규칙을 강제한다: 예) `min_train_size > test_size`,
  `step_size <= test_size`, RSI 임계값 단조성, 퀀타일 3개 이상·증가·(0,1).
- 프리셋: `configs/prod_conservative.json`, `configs/research_balanced.json`.

---

## 5. 모델 (`src/models/lgbm_heads.py`)

`MultiHeadStockModel` = **회귀 + 이진 방향분류 + 분위수(최소 3개)** 멀티헤드.

- LightGBM이 있으면 LightGBM, 없으면 sklearn GBDT로 폴백하며
  폴백 시 `SKLEARN_BACKEND_WARNING`을 메타데이터에 남긴다(비등가 경고).
- 피처 결측은 학습셋 중앙값으로 임퓨트하되 RSI/스토캐스틱/CCI는 중립값(50/50/0)으로.
- 헤드들은 GIL이 풀리는 LightGBM 특성을 이용해 스레드 병렬(`head_n_jobs`).
- 영속화: joblib 번들 + `.meta.json` 사이드카(시드·백엔드·피처 해시).
  로드 시 `artifact_version`과 **피처 해시 일치**를 강제해 깨진 모델 사용을 차단한다.

`predict()`는 `MultiHeadPrediction`(predicted_return, up_probability,
quantile_low/mid/high)을 돌려주며 분위수는 정렬해 교차를 방지한다.

---

## 6. 검증·백테스트 (`src/validation/`)

- `walk_forward.py` — 확장창(또는 `walk_forward_lookback_days` 슬라이딩) 워크포워드.
  폴드 수가 `min_required_folds` 미만이면 데이터 길이에 맞춰 창을 줄여 **1회 적응 재시도**.
- `support.py` — OOF를 tune/eval로 분할, 상승확률 보정기 학습/적용, OOF 진단.
- `signal_tuning.py` — tune 분할에서 신호 가중치(return/up_prob/uncertainty) 튜닝.
- `backtest.py` — 롱온리 top-k 백테스트(수수료/슬리피지/회전율/유동성/시장유형 한도).
- `result_validity.py` — 백테스트 표본/거래가능 종목 수가 부족하면 리포트를
  `status=warning` + `blocking_reasons`로 강등.

---

## 7. 신호 정책 (`src/domain/signal_policy.py`) — 결정 로직의 심장

**권고(가드레일 강제 지점):**

```python
predicted_return > +2.0(%)  → "매수"
predicted_return <= -2.0(%) → "매도"
그 외 / NaN                  → "관망"
```

`predicted_return`은 `inference/predict.py`에서 `expm1(log_return) * 100`으로
이미 **퍼센트**다. 권고는 오직 이 값만 본다.

**신호 스코어(순위/백테스트 선정용, 권고와 분리):**

```
signal_score = return_weight·norm_return + up_prob_weight·up_probability
             − uncertainty_penalty·uncertainty_score + event_boost_score
```

`event_boost_score`는 거래대금 상위·외국인/기관 동반순매수·52주 신고가·RSI·나스닥
선물 같은 **수급/기술적** 이벤트에서 나온다(뉴스 아님). 따라서 순위에는 영향을 주되
권고는 `predicted_return`에 묶여 가드레일이 유지된다.

그 외 산출: `risk_flag`(COVERAGE_HALT/HIGH_UNCERTAINTY/LOW_LIQUIDITY 등),
`confidence_label`, `position_size_hint`, `portfolio_action`, `trading_gate`,
`jongbae_score`/`jongbae_signal`, 한국어 `prediction_reason`.

---

## 8. 산출물 라이프사이클 (`src/reports/run_artifacts.py`)

- 실행별 원본은 `result/runs/<run_id>/`, 승격된 공식 최신본은 `result/latest/`.
- 매니페스트(`promoted`, `status`)가 운영 산출물 여부의 단일 판단 근거다.
- 샘플 스모크 실행은 운영 `latest/`를 덮어쓰지 않는다.
- CSV는 Excel/Windows 호환을 위해 `utf-8-sig`로 저장한다.

---

## 9. 뉴스 임팩트 패키지 (`src/news_impact/`)

형제 저장소 `stock-news-impact`를 `src` 하위로 벤더링한 LLM 파이프라인이다.
로컬 llama.cpp/gemma(`http://localhost:8001/v1`) 또는 OpenAI를 백엔드로 쓴다.

- `pipeline.run_daily_pipeline()` — 뉴스/공시 dedupe → 클러스터 → LLM 임팩트 판정
  → 점수 집계 → 랭킹 → `report.json/csv` + `audit.json`(재현 메타: 모델/온도/프롬프트 해시).
- LLM 실패·타임아웃은 카운트만 올리고 진행하며, 응답은 파일 캐시로 재사용한다.
- 두 실행 경로: **독립 리포트**(CLI) / **통합 컨텍스트**(`--news-impact-report` 또는
  `--news-impact-llm-config`로 `news_impact_*` 컬럼 부착). 둘 다 **표시 전용**이고
  `select_feature_columns()`가 모델 입력에서 떨어뜨린다.

---

## 10. 서빙: 게시 → Colab → Kakao

- `ops/publish_predictions.py` — 200종목 파이프라인 실행 → `ensure_operational_manifest()`
  통과 시 `published/latest/`와 `published/history/<거래일>/`에 복사 → `index.json` 갱신
  → (옵션) `git add/commit/push`. gemma 서버가 없으면 규칙기반으로 폴백한다.
- `colab/stock_predict_colab.py` — `load_published_predictions()`로 GitHub 기준데이터를
  **파이프라인 미실행** 상태로 서빙. `run_colab_pipeline()`은 사용자가 명시 호출할 때만.
- `chatbot/kakao_colab_bot.py` — Flask 봇. 기본은 `published/latest/` 기준 응답,
  사용자가 종목코드/이름 입력 또는 "최신화" 요청 시에만 해당 종목을 세션 한정 재예측해
  덮어 보여준다(GitHub push 없음).

별도 추천 엔진 `recommendation/close_betting.py` + `realtime_close_betting.py`는
ML 파이프라인과 **독립적으로** 라이브 OHLCV 거래대금 상위 20종목을 신고가/이평/거래량
급증/캔들 패턴으로 점수화하는 규칙기반 "종가베팅" 스코어러다.

---

## 11. 테스트 (`tests/`, 53개 모듈)

결정론적 단위/통합 테스트가 가드레일·계약을 촘촘히 지킨다: display-only 가드,
시그널 정책 계약/권고/이벤트부스트, 워크포워드, 확률보정 가드, 리포트 메타 계약,
캐시/파일 하드닝, 시크릿 레다크션, 패키징 메타, publish/published store,
Kakao 봇 헬퍼, 파이프라인 스모크 등. 외부 네트워크는 모킹/비활성화가 원칙이다.

> 본 문서는 코드 정독으로 작성했으며 테스트 실행 결과를 주장하지 않는다.
> 검증이 필요하면 `pytest`(+`tests/test_pipeline_smoke.py`)를 직접 돌릴 것.

---

## 12. 강점

- 가드레일이 문서가 아니라 **코드+테스트로 강제**된다.
- 단계별 진단/커버리지 게이트/매니페스트로 운영 관측성이 높다.
- 엄격한 설정 검증(오타 후보 제시 포함)으로 잘못된 설정을 조기 차단한다.
- 모델 영속화에 버전·피처 해시 검증이 있어 깨진 아티팩트 사용을 막는다.
- 외부 의존(yfinance/DART/Naver/LLM)마다 폴백과 커버리지 표기가 있다.

---

## 13. 개선안 (계획) — 우선순위별

> 가드레일을 깨지 않는 선에서, "작업 중인 코드를 개선한다" 수준의 표적 제안이다.
> 무관한 대규모 리팩터링은 의도적으로 제외했다.

### P0 — 가드레일 회귀 방지 유지·강화
- display-only 가드 테스트는 이미 있다. 신규 `news_*`/`*_impact_*` 컬럼이 늘 때마다
  가드 목록 누락이 없도록, "뉴스류 컬럼이 모델 피처에 절대 안 들어간다"를
  **패턴 기반**으로 단언하는 테스트를 1개 추가하면 드리프트에 더 강해진다.

### P1 — `chatbot/kakao_colab_bot.py` 단계적 분해
- 단일 클래스 `KakaoColabPredictionBot`이 약 1,800 LOC·메서드 ~90개로
  HTTP 핸들링·인텐트 라우팅·메시지 포맷·서브프로세스 잡 관리·부트스트랩 prewarm·
  라이브 이벤트 수집·이슈요약·세션 상태를 모두 떠안는 **god class**다.
- 책임별 협력 객체로 분리 권장: ① 인텐트/라우팅, ② 메시지 포맷터,
  ③ 예측 잡 매니저(서브프로세스/백그라운드), ④ 컨텍스트(이슈요약/뉴스임팩트) 부착,
  ⑤ 세션/레지스트리. 테스트가 잘 갖춰져 있어 안전망 위에서 점진 분해가 가능하다.

### P1 — `pipeline.py` 호환 래퍼 정리
- 상단의 `_`-접두 위임 래퍼들(약 100~330행)은 대부분 테스트/임포트 호환용이다.
  테스트가 원본 모듈을 직접 임포트하도록 옮기면, 오케스트레이터 가독성이 크게 오른다.
  한 번에 지우지 말고 "테스트 이전 → 래퍼 제거"를 묶음으로 진행.

### P2 — 매직넘버를 설정으로 승격
- 권고 임계값(±2.0%)이 `signal_policy.py`에 하드코딩되어 있고 `SignalConfig`에 없다.
  `confidence_label` 구간(0.34/0.67/0.80)과 이벤트 부스트 상수들도 모듈 상수다.
  최소한 **권고 임계값**은 설정(`SignalConfig`)으로 올려 프리셋별 보정과 검증 대상이
  되도록 권장. 옮길 때 `_validate_app_config`에 단조성/범위 검증을 함께 추가.

### P2 — `signal_policy.py`의 row/vectorized 이중 구현 통합
- 동일 로직이 행단위(`risk_flag`, `_jongbae_score`, `recommendation_from_signal`,
  `build_pm_summary_fields`)와 벡터화(`_risk_flag_series`, `_jongbae_score_series`,
  `_recommendation_series`, `_pm_summary_frame`)로 **양립**한다. 운영은 벡터화 경로를
  쓰고 행단위는 테스트 호환용이다. 드리프트(두 경로 결과 불일치) 리스크가 있으므로,
  ① 두 경로 동치성 테스트를 먼저 고정하고 ② 행단위를 벡터화 경로의 1행 어댑터로
  재정의해 단일 출처화하는 방향을 권장.

### P2 — publish의 실제 뉴스 모드 표면화
- `publish_predictions.run_publish()`는 요청된 모드(`gemma`/`rule`)를 메타에 기록하지만,
  스코어링 내부의 **무음 gemma→rule 폴백**은 리포트에 드러나지 않는다(코드 주석에도 명시).
  파이프라인 리포트에 "실제 사용된 LLM 백엔드/폴백 횟수"를 올려 `publish_meta`가
  실제값을 반영하도록 개선하면 운영 신뢰도가 오른다.

### P3 — `price_features.build_features`의 레거시 스캐폴딩 정리
- `legacy_removed_default_map`(약 40개 컬럼)을 0으로 추가했다가 마지막에 다시 drop하는
  방어 코드가 흐름을 가린다. 구 브랜치/입력 호환이 더 이상 필요 없어지면 제거 권장.

### P3 — `result/` 산출물 관리 정책 명문화
- runs/latest 라이프사이클은 잘 동작하나, 보존 기간·정리 주기·용량 상한을
  `utils/result_cleanup.py`와 함께 한 곳에 문서화하면 운영 부담이 준다.

---

## 14. 권장 실행 로드맵

1. **안전망 강화(P0)** — 뉴스류 컬럼 패턴 가드 테스트 추가. 1개 PR, 리스크 최저.
2. **운영 가시성(P2)** — publish 실제 뉴스 모드/폴백 표면화. 운영 신뢰 직결.
3. **정책 설정화(P2)** — 권고 임계값을 `SignalConfig`로 승격 + 검증. 행/벡터 동치성 테스트.
4. **챗봇 분해(P1)** — 책임별 협력 객체로 점진 분해(메서드 묶음 단위 PR).
5. **오케스트레이터 정리(P1)** — 테스트 이전 후 `pipeline.py` 호환 래퍼 제거.
6. **레거시 정리(P3)** — 피처 스캐폴딩/`result/` 정책 정리.

## 15. 변경 시 체크리스트

- [ ] 뉴스/공시 신호가 `predicted_return`·권고·`signal_score`·순위에 들어가지 않는가?
- [ ] 새 컬럼이 모델 피처면 `feature_selection`, 표시용이면 display-only 목록에 정확히 들어갔는가?
- [ ] 임계값/가중치 변경 시 `_validate_app_config` 검증과 관련 테스트를 갱신했는가?
- [ ] `pytest`(최소 영향 테스트 + `test_pipeline_smoke.py`)를 통과하는가?
- [ ] 산출물 변경 시 매니페스트/리포트 계약 테스트(`test_report_metadata` 등)를 확인했는가?
