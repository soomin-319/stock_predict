# 01. 파이프라인 — 전체 흐름 및 진입점

## 개요

`src/pipeline.py`가 전체 파이프라인의 단일 진입점이다. `run_pipeline()` 함수가 12개 단계를 순서대로 실행하고 결과물을 `result/` 디렉터리에 저장한다.

## 파이프라인 12단계

| 단계 | 내용 | 관련 모듈 |
|------|------|-----------|
| 1 | 설정 로드 | `src/config/settings.py` |
| 2 | OHLCV CSV 로드 | `src/data/loaders.py` |
| 3 | 데이터 정제 및 유니버스 필터 | `src/data/cleaners.py`, `src/data/universe.py` |
| 4 | 투자자 컨텍스트 추가 (선택) | `src/data/investor_context.py` |
| 5 | 가격 피처 빌드 | `src/features/price_features.py` |
| 6 | 외부 시장 피처 추가 (선택) | `src/features/external_features.py` |
| 7 | Walk-Forward 검증 실행 | `src/validation/walk_forward.py` |
| 8 | 기준선 평가 | `src/validation/baselines.py` |
| 9 | OOF 예측 처리 및 캘리브레이션 | `src/validation/support.py` |
| 10 | 시그널 가중치 튜닝 | `src/validation/signal_tuning.py` |
| 11 | 홀드아웃 백테스트 실행 | `src/validation/backtest.py` |
| 12 | 최종 모델 학습 + 최신 예측 + 아티팩트 저장 | `src/models/lgbm_heads.py`, `src/reports/` |

## 핵심 함수

### `run_pipeline()`

```python
# src/pipeline.py:972
def run_pipeline(
    input_csv: str,
    output_csv: str,
    universe_csv: str | None = None,
    use_external: bool = True,
    use_investor_context: bool = False,
    ...
) -> dict
```

모든 파이프라인 단계를 실행하고 `pipeline_report.json` 경로를 포함한 결과 딕셔너리를 반환한다.

### `build_cli_parser()`

```python
# src/pipeline.py:1166
def build_cli_parser() -> argparse.ArgumentParser
```

CLI 인수 파서를 반환한다. 콘솔 스크립트 `stock-predict`가 이 파서를 사용한다.

## CLI 콘솔 스크립트

| 명령어 | 진입점 | 설명 |
|--------|--------|------|
| `stock-predict` | `src/pipeline.py:main` | 메인 예측 파이프라인 |
| `stock-predict-kakao` | `src/chatbot/kakao_colab_bot.py` | 카카오 챗봇 서버 |
| `stock-news-impact` | `src/news_impact/run.py` | 뉴스 임팩트 독립 실행 |

## 주요 CLI 옵션

```bash
# 샘플 데이터로 테스트 (네트워크 없음)
stock-predict --input data/sample_ohlcv.csv --disable-external

# 실시간 OHLCV 갱신 후 실행
stock-predict --fetch-real --input data/real_ohlcv.csv

# 투자자 컨텍스트 포함
stock-predict --fetch-investor-context

# 증분 갱신 (최신 날짜 이후만 다운로드)
stock-predict --auto-refresh-real --input data/real_ohlcv.csv

# 특정 종목 추가
stock-predict --add-symbols 005930 000660.KS
```

## PipelineDiagnostics

```python
# src/pipeline.py:327
@dataclass(slots=True)
class PipelineDiagnostics:
    timings_seconds: dict[str, float]
    row_counts: dict[str, int]
```

각 단계의 실행 시간과 데이터 행 수를 추적하여 `pipeline_report.json`의 `diagnostics` 필드에 저장한다.

## 파이프라인 지원 함수 (`src/pipeline_support.py`)

| 함수 | 역할 |
|------|------|
| `build_scored_prediction_frame()` | OOF/최신 예측에 시그널 점수 계산 |
| `build_symbol_history_accuracy()` | 종목별 과거 방향성 정확도 집계 |
| `finalize_latest_prediction_frame()` | 최종 예측 프레임 컬럼 정리 |
| `PredictionFrameContext` | 커버리지 비율 등 컨텍스트 전달 |

## 실행 환경: GitHub → Colab → KakaoTalk

```
GitHub (코드 저장)
    ↓ clone/pull
Google Colab (파이프라인 실행)
    ↓ ngrok webhook
KakaoTalk (사용자 인터페이스)
```

Colab 통합 코드는 `colab/stock_predict_colab.py`에 있다.

---

## 개선 및 수정 제안

> 코드 분석으로 도출한 제안. 우선순위: **P0(정확성/버그) > P1(견고성/재현성) > P2(성능/품질/문서)**.

### P1 — `cfg.signal` 전역 in-place 변형 (재현성/직렬화 오염)

- **문제**: 튜닝 단계에서 `cfg.signal.return_weight = tuned[...]` 식으로 **공유 `AppConfig`를 직접 변형**한다(`src/pipeline.py:591-594`). 이후 `app_config_to_dict(cfg)`로 리포트에 직렬화되는 값이 "사용자가 준 설정"이 아니라 "튜닝 후 값"이 되어, 리포트만 보고 재현이 어렵다. 동일 프로세스에서 파이프라인을 2회 호출하면 1회차 튜닝값이 2회차 기본값으로 새어든다.
- **제안**: 튜닝 결과는 `dataclasses.replace(cfg.signal, **tuned)`로 **새 객체**를 만들어 지역 변수로 전달하고, 리포트에는 `config_input`(원본)과 `signal_weights_tuned`(결과)를 **분리**해 기록.

### P1 — 단계별 실패 격리 부재

- **문제**: 12단계 중 한 단계(예: 외부 데이터 다운로드, 이슈 요약 LLM 호출)에서 예외가 나면 전체가 중단된다. `pipeline_report.json`에 `status: "error"`와 실패 단계/스택을 남기는 상위 가드가 없다.
- **제안**: 각 단계를 `try/except`로 감싸 `diagnostics`에 `stage_status`(ok/skipped/error)와 사유를 기록하고, 선택 단계(external/investor/news) 실패는 `caution`으로 강등해 핵심 예측은 산출되도록 한다.

### P1 — 적응형 재시도 조건이 "폴드 0개"에만 동작

- **문제**: `_adaptive_training_cfg`는 walk-forward가 **폴드를 전혀 못 만들 때만** 재시도한다(`src/pipeline.py:141` 부근). 폴드가 1~2개로 과소 생성되어도 그대로 진행되어 통계적으로 빈약한 검증 결과가 나온다.
- **제안**: `len(folds) < min_required_folds`(예: 3) 조건에서도 적응형 설정으로 재시도하고, 최종 폴드 수를 `diagnostics`에 노출.

### P2 — `PipelineDiagnostics` 타이밍 커버리지 검증

- **문제**: `timings_seconds`/`row_counts`가 모든 단계를 포괄하는지 보장하는 테스트가 없어, 단계 추가 시 누락되기 쉽다.
- **제안**: 12단계 키 집합을 상수로 정의하고, 실행 후 누락 키를 경고로 남기는 가벼운 사후 검증 추가.

### P2 — CLI 사용성/문서

- 종료 코드 규약(정상 0 / 검증 경고 / 데이터 실패)을 표로 문서화하면 Colab·CI 자동화에서 분기하기 쉽다.
