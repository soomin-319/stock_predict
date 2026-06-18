# 01. 파이프라인 — 전체 흐름 및 진입점

## 개요

`src/pipeline.py`가 전체 예측 파이프라인의 단일 진입점이다. `run_pipeline()`이 12개 진행 단계를
순서대로 실행하고 결과물을 `result/` 디렉터리에 저장한다. 내부적으로는 단계들이 6개의 스테이지
헬퍼로 묶여 있으며, 각 스테이지는 `PipelineDiagnostics`에 실행 시간·행 수·상태(ok/caution/error)를
기록한다.

## 파이프라인 12단계 (진행 표시 기준)

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

## 내부 스테이지 헬퍼

`run_pipeline()`은 다음 헬퍼를 순서대로 호출하며, 각 호출은 `diagnostics.time_stage(...)`로 감싸진다.

| 스테이지 키 | 헬퍼 | 역할 |
|-------------|------|------|
| `load_config_and_inputs` | `_load_pipeline_config_and_data` | 설정 로드, OHLCV 로드/정제, 유니버스 필터 |
| `prepare_context` | `_prepare_pipeline_context` | 투자자 컨텍스트 + 원본 뉴스/공시 이벤트 수집(선택) |
| `build_feature_matrix` | `_build_pipeline_feature_matrix` | 가격/외부/투자자 이벤트 피처 빌드, 피처 컬럼 선택 |
| `validation_and_tuning` | `_run_pipeline_validation` | walk-forward, 기준선, OOF 캘리브레이션, 시그널 튜닝, 백테스트 |
| `train_final_and_predict_latest` | `_predict_pipeline_latest` | 최종 모델 학습, 최신 예측, 이슈요약/뉴스임팩트 컨텍스트 부착 |
| `save_pipeline_artifacts` | `_write_pipeline_artifacts` | CSV/JSON 저장, 리포트 조립, `latest/` 승격 |

## 핵심 함수

### `run_pipeline()`

```python
# src/pipeline.py:1064
def run_pipeline(
    input_csv: str,
    output_csv: str,
    universe_csv: str | None = None,
    report_json: str | None = None,
    use_external: bool = True,
    use_investor_context: bool = False,
    ...
) -> dict
```

모든 파이프라인 단계를 실행하고 `pipeline_report.json` 경로와 요약을 포함한 결과 딕셔너리를 반환한다.

### `build_cli_parser()`

```python
# src/pipeline.py:1284
def build_cli_parser() -> argparse.ArgumentParser
```

CLI 인수 파서를 반환한다. 콘솔 스크립트 `stock-predict`(`main`, `src/pipeline.py:1372`)가 이 파서를 사용한다.

## 진단 (`PipelineDiagnostics`)

```python
# src/pipeline.py:336
@dataclass(slots=True)
class PipelineDiagnostics:
    timings_seconds: dict[str, float]
    row_counts: dict[str, int]
    stage_status: dict[str, str]      # ok / caution / error
    warnings: list[str]
```

- `time_stage(key)`: 컨텍스트 매니저로 스테이지 실행 시간을 기록한다.
- `mark_stage(key, status, reason)`: 스테이지 상태와 사유를 남긴다. 선택 단계(외부/투자자/뉴스
  임팩트)의 데이터 수집 실패는 `caution`으로 강등되어 핵심 예측은 계속 산출된다.
- `set_rows(key, df)`: 단계별 데이터 행 수를 추적한다.
- `validate_stage_coverage()`: 기대 스테이지 키 누락 여부를 사후 검증한다.

이 정보는 `pipeline_report.json`의 `diagnostics` 필드에 저장된다.

## 재현성

시그널 가중치 튜닝 결과는 공유 `AppConfig`를 변형하지 않는다. `_run_pipeline_validation`이
튜닝된 새 `SignalConfig`(`tuned_signal_config`)를 만들어 최신 예측 단계에 전달하므로, 리포트의
입력 설정과 튜닝 결과가 분리되어 기록된다. 동일 프로세스에서 파이프라인을 2회 호출해도 1회차
튜닝값이 2회차로 새어들지 않는다.

## 파이프라인 지원 함수 (`src/pipeline_support.py`)

| 함수 | 역할 |
|------|------|
| `build_scored_prediction_frame()` | OOF/최신 예측에 시그널 점수 계산 |
| `build_symbol_history_accuracy()` | 종목별 과거 방향성 정확도 집계 |
| `finalize_latest_prediction_frame()` | 최종 예측 프레임 컬럼 정리 |
| `PredictionFrameContext` | 커버리지 비율 등 컨텍스트 전달 |

## 주요 CLI 사용 예

```powershell
# 샘플 데이터로 테스트 (네트워크 없음)
stock-predict --input data/sample_ohlcv.csv --disable-external

# 실시간 OHLCV 갱신 후 실행 (심볼 미지정 시 번들된 KOSPI200 200종목)
stock-predict --fetch-real --input data/real_ohlcv.csv

# 증분 갱신 (최신 날짜 이후만 다운로드)
stock-predict --auto-refresh-real --input data/real_ohlcv.csv

# 특정 종목 추가
stock-predict --add-symbols 005930 000660.KS

# 투자자 컨텍스트 포함
stock-predict --fetch-investor-context
```

## 실행 환경: GitHub → Colab → KakaoTalk

```
GitHub (코드 저장)
    ↓ clone/pull
Google Colab (파이프라인 실행)
    ↓ ngrok webhook
KakaoTalk (사용자 인터페이스)
```

Colab 통합 코드는 `colab/`에 있고, 챗봇 서버는 [09_chatbot.md](09_chatbot.md) 참조.

## CLI 종료 코드 규약

| 종료 코드 | 의미 | 자동화 처리 |
|-----------|------|-------------|
| `0` | 파이프라인 실행 완료. `pipeline_report.json`의 `status`가 `ok` 또는 `warning`일 수 있다. | 리포트의 `status`, `blocking_reasons`, `diagnostics`로 후속 분기 |
| non-zero | 입력 로드, 검증, 모델 학습, 아티팩트 저장 등 핵심 단계의 fatal error | 실패로 처리하고 로그/스택 확인 |

선택 단계(외부 시장 피처, 투자자 컨텍스트, 이슈 요약, 뉴스 임팩트 컨텍스트)는 실패해도 가능한 경우
`diagnostics.stage_status`에 `caution`과 사유를 남기고 핵심 예측 산출을 계속한다.
