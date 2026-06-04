# 코드베이스 분석 및 최적화 제안서

작성일: 2026-06-04  
대상: `C:\Users\카운\Desktop\stock_predict`  
범위: `src/`, `news_impact/`, `tests/`, 설정/문서/실행 흐름

> 본 프로젝트 산출물은 리서치/운영 보조 자료이며 투자 조언 또는 자동매매 시스템이 아니다. 특히 `predicted_return` 기반 매수/매도/관망 정책과 뉴스/공시의 display-only 원칙은 최적화 과정에서도 유지해야 한다.

## 0. 진행 현황

업데이트: 2026-06-04  
진행 브랜치/PR: `codex/add-kospi200-symbol-map`, PR #248  
진행 커밋: `afaa6e3`

### 완료(P0)

- [x] 뉴스/공시 display-only 컬럼을 모델 feature 선택에서 차단
  - `DISPLAY_ONLY_CONTEXT_COLUMNS` 추가
  - `MODEL_FEATURE_COLUMN_BASE`로 모델 허용 feature 분리
  - `select_feature_columns()`에서 뉴스/공시 context 컬럼 제외
- [x] display-only 불변성 회귀 테스트 추가
  - 뉴스/공시 값이 바뀌어도 모델 feature set/value가 변하지 않음을 검증
- [x] pytest cache 경로를 `result/.pytest_cache`로 이동
- [x] Windows subprocess UTF-8 처리 고정
  - `encoding="utf-8"`, `errors="replace"`
  - child env에 `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` 주입
- [x] 전체 검증 완료
  - `python -m compileall -q src news_impact`
  - `pytest -q --basetemp result\.pytest_tmp\p0_full_final2` → 189 passed

### 미진행

- [x] `run_pipeline()` 단계 함수 분해
- [x] `kakao_colab_bot.py` 저위험 helper 모듈 일부 분리
- [ ] 외부 market feature 캐시 추가 (이번 범위 제외)
- [x] `pipeline_report.json` timing/row-count/coverage diagnostics 추가

## 1. 요약

현재 코드베이스는 기능 범위가 넓고 테스트가 풍부하다. 파이프라인, 피처 생성, 워크포워드 검증, 백테스트, 리포트, Kakao/Colab 봇, vendored `news_impact` 패키지가 한 저장소 안에서 잘 연결되어 있다. 다만 운영 안정성과 확장성 관점에서는 다음 5개가 우선 개선 포인트다.

1. **뉴스/공시 display-only 가드레일 강화(P0)**  
   `src/features/price_features.py`의 feature whitelist에 `disclosure_score`, `news_sentiment`, `news_impact_score`, `news_article_count` 등이 포함되어 있어, 설정에 따라 뉴스/공시성 데이터가 모델 입력이 될 수 있다. 저장소 가드레일과 충돌 가능성이 크다.
2. **대형 오케스트레이터 분해(P1)**  
   `run_pipeline()` 423라인, `kakao_colab_bot.py` 약 1,953라인. 변경 리스크와 회귀 테스트 범위가 커진다.
3. **실행/테스트 환경 안정화(P0/P1)**  
   일반 `pytest`는 Windows temp/cache 권한 문제로 실패했고, `--basetemp result/.pytest_tmp/analysis_run` 사용 시 전체 통과했다. 테스트 캐시 경로 정책과 실제 설정이 어긋난다.
4. **성능 병목 완화(P1/P2)**  
   워크포워드 fold마다 전체 DataFrame 직렬화, LightGBM 고정 파라미터/early stopping 없음, 외부 지표 다운로드 캐시 부재, rolling/groupby 반복이 주요 병목 후보다.
5. **운영 관측성/장애 대응 강화(P1)**  
   broad `except Exception`, `print()` 기반 로그, subprocess 인코딩 경고, JSON state 파일 경쟁 조건 가능성이 남아 있다.

## 2. 관찰 지표

### 규모

- Python 파일: 80개
- 총 Python 라인: 약 13,720라인
- 대형 파일 상위:
  - `src/chatbot/kakao_colab_bot.py`: 1,953라인, 함수/메서드 92개
  - `src/pipeline.py`: 840라인
  - `src/reports/issue_summary.py`: 661라인
  - `news_impact/pipeline.py`: 609라인
  - `src/features/price_features.py`: 512라인

### 긴 함수 상위

| 함수 | 라인 | 리스크 |
|---|---:|---|
| `src.pipeline.run_pipeline` | 423 | 데이터/학습/검증/리포트 책임 혼재 |
| `src.features.price_features.build_features` | 282 | 피처/타깃/legacy alias 혼재 |
| `src.validation.backtest.run_long_only_topk_backtest` | 164 | 포트폴리오 선택/비용/리포트 혼재 |
| `KakaoColabPredictionBot._handle_symbol_request` | 136 | intent/cache/job/summary 응답 혼재 |
| `news_impact.pipeline.run_daily_pipeline` | 129 | 독립 패키지 오케스트레이션 집중 |

### 검증 결과

```powershell
python -m compileall -q src news_impact
# 통과

pytest -q
# 권한 문제로 실패: C:\Users\카운\AppData\Local\Temp\pytest-of-카운 접근 거부

pytest -q --basetemp result\.pytest_tmp\analysis_run
# 186 passed, 3 warnings
```

주의: 테스트 통과는 코드 기능의 좋은 신호지만, 일반 실행 환경의 temp/cache 권한과 subprocess UTF-8 경고는 별도 조치가 필요하다.

## 3. 아키텍처 평가

### 강점

- `src/data`, `src/features`, `src/models`, `src/validation`, `src/reports`, `src/domain`, `src/chatbot`로 기본 계층 분리가 되어 있다.
- 워크포워드 OOF, probability calibration, holdout backtest, final model train 흐름이 명확하다.
- `news_impact/`가 top-level vendored package로 분리되어 있어 독립 실행 가능하다.
- `recommendation_from_signal()`은 `predicted_return`만으로 매수/매도/관망을 결정하도록 계약이 잡혀 있다.
- 테스트 커버리지가 넓다. 특히 chatbot, signal policy, news impact display-only, persistence, smoke test가 존재한다.

### 약점

- `src/pipeline.py`가 호환 wrapper와 실제 orchestration을 동시에 가진다.
- chatbot이 단일 클래스에 intent parsing, job registry, subprocess 실행, cache, live event fetch, message formatting까지 포함한다.
- feature registry가 명시적 계약보다 prefix/whitelist에 의존한다.
- display-only context와 model features의 물리적 경계가 약하다.
- 외부 API/yfinance/OpenAI 실패가 일부 경로에서 조용히 삼켜져 운영 디버깅 비용이 커질 수 있다.

## 4. P0: 즉시 권장 조치

### 4.1 뉴스/공시 display-only 강제 분리

현재 위험:

- `FEATURE_COLUMN_BASE`에 다음 계열이 포함되어 있음:
  - `disclosure_score`
  - `news_sentiment`
  - `news_relevance_score`
  - `news_impact_score`
  - `news_article_count`
  - `news_positive_signal`
  - `news_negative_signal`
- `build_features()`에서 이 값을 수치화하고 파생 피처를 만든다.

권장:

1. 모델 입력 whitelist에서 뉴스/공시 계열 제거.
2. `display_context_columns`와 `model_feature_columns`를 별도 상수/모듈로 분리.
3. 테스트 추가:
   - `select_feature_columns()`가 뉴스/공시 계열을 절대 반환하지 않음.
   - `--fetch-investor-context` ON/OFF 또는 raw news/disclosure 값 변경이 `predicted_return`에 영향 없음.
4. 허용 피처는 가격/거래대금/시장지수/투자자 수급 등으로 명시.

### 4.2 테스트 temp/cache 경로 정렬

현재:

- `pyproject.toml`: `cache_dir = ".pytest_cache"`
- 저장소 가이드: pytest cache/temp는 ignored `result/` 하위 권장
- 실제 일반 `pytest`는 권한 문제 발생

권장:

```toml
[tool.pytest.ini_options]
cache_dir = "result/.pytest_cache"
```

추가로 CI/로컬 문서에 다음 권장:

```powershell
pytest -q --basetemp result\.pytest_tmp
```

### 4.3 subprocess UTF-8 고정

테스트 경고:

- `UnicodeDecodeError: 'cp949' codec can't decode byte ...`

권장:

- `subprocess.Popen(..., text=True, encoding="utf-8", errors="replace")`
- child env에 `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` 주입
- Windows/Colab 공통 로그 파일은 UTF-8로 명시

## 5. P1: 구조 개선

### 5.1 `run_pipeline()` 단계 객체화

현재 `run_pipeline()`은 13개 진행 단계를 한 함수에 직접 구현한다. 아래 단위로 분해 권장:

- `load_config_and_inputs()`
- `prepare_context()`
- `build_feature_matrix()`
- `run_validation_and_tuning()`
- `train_final_and_predict_latest()`
- `attach_display_context()`
- `save_pipeline_artifacts()`

권장 데이터 구조:

```python
@dataclass(slots=True)
class PipelineRunContext:
    cfg: AppConfig
    input_csv: str
    result_dir: Path
    feature_columns: list[str]
    coverage: dict

@dataclass(slots=True)
class PipelineArtifacts:
    detail_path: Path
    simple_path: Path
    report_path: Path | None
    figure_dir: Path
```

효과:

- 테스트를 단계별로 좁게 작성 가능
- 파이프라인 재실행/캐시/partial retry 구현 쉬움
- display-only context 경계 명확화

### 5.2 Kakao bot 모듈 분리

권장 분리:

| 새 모듈 | 책임 |
|---|---|
| `src/chatbot/intent.py` | 코드/종목명/help/status/recommendation intent parsing |
| `src/chatbot/cache.py` | `result_simple/detail/news/disclosure` 로딩, mtime cache |
| `src/chatbot/jobs.py` | subprocess command, registry, 상태 전이 |
| `src/chatbot/summaries.py` | issue/news impact on-demand summary |
| `src/chatbot/responses.py` | Kakao SimpleText/quick reply formatting |
| `src/chatbot/server.py` | Flask/ngrok launch |

효과:

- 1,953라인 단일 파일 변경 리스크 감소
- 테스트 fixture 단순화
- 동시 요청/상태 관리 개선 여지 확보

### 5.3 Feature registry 명시화

현재는 `FEATURE_COLUMN_PREFIXES` + `FEATURE_COLUMN_BASE`로 선택한다. 권장:

- `src/features/registry.py` 추가
- `FeatureSpec(name, source, allowed_for_model, display_only, leakage_risk, description)` 구조화
- `select_feature_columns(df, policy="model")`
- `display_only=True`는 학습/검증 함수에서 assert로 차단

핵심 assert:

```python
for c in feature_columns:
    assert c not in DISPLAY_ONLY_CONTEXT_COLUMNS
```

## 6. P1/P2: 성능 최적화

### 6.1 워크포워드 검증 비용 절감

현재:

- `ProcessPoolExecutor`에 fold별 `train_df`, `valid_df` 전체를 전달한다.
- fold 수/데이터 크기가 커질수록 pickle/메모리 비용 증가.

권장:

1. fold input은 날짜 index 범위만 전달.
2. worker 내부에서 공유 원본 또는 parquet/memmap slice 로딩.
3. LightGBM `early_stopping`, `valid_set` 사용.
4. `walk_forward_n_jobs * model_n_jobs * model_head_n_jobs` 총 스레드 예산을 config에서 검증.

### 6.2 Feature engineering 벡터화/캐시

병목 후보:

- `build_features()` 내 다중 `groupby().transform(lambda ...)`
- CCI의 `rolling.apply(lambda ...)`
- 외부 지표 다운로드 후 매번 merge

권장:

- OHLCV 정렬/타입 보장 후 groupby 객체 재사용 최소화
- 외부 market features는 `result/cache/external_market_features.parquet`로 날짜+심볼 기준 캐시
- sample/production feature build 시간을 `pipeline_report.json`에 단계별 기록
- feature matrix를 parquet로 저장해 재학습/챗봇 반복 실행 단축

### 6.3 모델 학습/저장

현재 `MultiHeadStockModel.save/load()`는 있으나 메인 pipeline에서 적극 활용되지 않는다.

권장:

- final model artifact 저장: `result/models/latest.joblib`
- metadata에 data cutoff, feature hash, config hash, git commit 추가
- Kakao on-demand는 전체 재학습보다 기존 모델 + incremental data feature refresh 우선 검토
- 모델 성능 drift 기준 추가: OOF calibration, hit-rate, turnover, missing coverage threshold

## 7. 운영 안정성 개선

### 7.1 로그 표준화

권장:

- `print()`는 CLI progress 전용으로 제한
- 외부 API 실패, LLM 실패, cache parse 실패는 `logging`에 structured context 포함
- report JSON에 `warnings`, `degraded_features`, `external_failures` 섹션 추가

### 7.2 broad exception 축소

`except Exception`은 네트워크 fallback에는 유용하나, 데이터 계약 위반까지 숨길 수 있다.

권장:

- 외부 API: `NetworkError`, `TimeoutError`, provider parse error로 축소
- 내부 데이터: required column/type 오류는 fail-fast
- fallback 시 coverage에 실패 원인 코드 저장

### 7.3 상태 파일 경쟁 조건

chatbot은 JSON registry/state 파일과 background thread를 사용한다.

권장:

- state write는 atomic replace 유지/도입
- file lock 또는 process-local lock 범위 검토
- 상태 전이 enum화: `queued -> running -> completed|failed|stale`
- 오래된 running job 정리 기준을 config화

## 8. 데이터/모델링 품질 개선

### 8.1 의사결정 시점 명시

현재 피처는 `Close`, `Volume` 등 당일 종가 기준 next-day 예측에 적합하다. 장중/종가베팅 기능과 섞일 때 leakage 오해가 생길 수 있다.

권장:

- `prediction_cutoff = after_close | intraday | close_betting` 명시
- cutoff별 허용 feature set 분리
- `target_log_return` 생성 전후 컬럼 계약 문서화

### 8.2 백테스트 현실성 강화

현재 비용/슬리피지/turnover/market-type cap이 존재하는 점은 좋다. 추가 권장:

- 매수 가능 가격 가정 명시: 종가 체결, 익일 시가, VWAP 등
- KRX 상하한가/거래정지/관리종목 데이터가 있으면 체결 가능성 필터 추가
- survivorship bias 방지를 위한 universe snapshot 관리
- benchmark를 KOSPI/KOSDAQ 혼합 또는 universe equal-weight로 병행

### 8.3 리포트 품질

권장:

- `pipeline_report.json`에 단계별 소요시간/행 수/결측률/feature count 변화 기록
- `result_detail.csv`와 `result_simple.csv` 스키마 버전 명시
- Excel용 한글 컬럼과 내부 영문 컬럼을 mapping 파일로 관리

## 9. 권장 실행 로드맵

### Phase 0: 1~2일

- 뉴스/공시 display-only feature 차단
- 관련 테스트 추가
- pytest cache/temp 경로 정리
- subprocess UTF-8 경고 제거

완료 기준:

- `pytest -q --basetemp result/.pytest_tmp` 통과
- 뉴스/공시 값 변경 테스트에서 `predicted_return`, recommendation 불변 확인

### Phase 1: 1~2주

- `run_pipeline()` 단계별 모듈 분리
- chatbot 5~6개 모듈로 분해
- feature registry 도입
- pipeline report에 단계별 timing/coverage 추가

완료 기준:

- 기존 CLI/API 호환
- smoke test와 Kakao 테스트 전부 통과
- 주요 함수 150라인 이하 목표

### Phase 2: 2~4주

- external feature cache/parquet 도입
- 워크포워드 직렬화 비용 개선
- final model artifact 저장 및 재사용
- drift/coverage alerting 추가

완료 기준:

- 동일 데이터 기준 pipeline runtime 감소
- 결과 artifact 재현성 강화
- 운영 실패 원인 파악 가능

## 10. 최우선 체크리스트

- [x] `select_feature_columns()`에서 뉴스/공시 계열 제거 (완료: `DISPLAY_ONLY_CONTEXT_COLUMNS`/`MODEL_FEATURE_COLUMN_BASE`)
- [x] display-only context 불변성 테스트 추가 (완료: `tests/test_display_only_feature_guard.py`)
- [x] `pyproject.toml` pytest cache를 `result/.pytest_cache`로 변경
- [x] Windows subprocess 로그 UTF-8 처리 (완료: `encoding="utf-8"`, `errors="replace"`, UTF-8 env)
- [x] `run_pipeline()`을 6~7개 단계 함수로 분해
- [x] `kakao_colab_bot.py` 저위험 helper 모듈 일부 분리
- [ ] 외부 market feature 캐시 추가 (이번 범위 제외)
- [x] `pipeline_report.json`에 timing/row-count/coverage diagnostics 추가

## 결론

코드베이스는 이미 기능과 테스트 기반이 탄탄하다. 가장 중요한 최적화는 단순 속도 개선이 아니라 **의사결정 가드레일의 물리적 강제**, **대형 오케스트레이터 분해**, **운영 환경 재현성 강화**다. 특히 뉴스/공시가 display-only라는 원칙은 모델 입력 단계에서 assert로 차단해야 장기 유지보수와 신뢰성을 확보할 수 있다.
