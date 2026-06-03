# 코드베이스 분석 및 수정/개선 제안

작성일: 2026-06-03 KST

## 요약

이 저장소는 다음 거래일 및 다중 기간 수익률 신호를 연구하기 위한 Python 주식 예측 파이프라인이다. 핵심 흐름은 OHLCV/시장/수급/외부 컨텍스트 로드, 피처 생성, LightGBM 또는 sklearn fallback 모델 학습, walk-forward OOF 검증, long-only top-k 백테스트, `result/` 산출물 생성, Kakao/Colab 표시 연동이다.

중요 가드레일:

- 매수/매도/관망 및 top-k 선택은 다음 거래일 예상 수익률(`predicted_return`) 기준이어야 한다.
- 뉴스/공시 및 `news_impact` 값은 표시·검토용 컨텍스트일 뿐이며 `predicted_return`, 추천, 랭킹, 자동 신호 결정을 변경하면 안 된다.
- CSV/JSON/figure 산출물은 `result/` 아래에 둔다.
- CSV는 Windows/Excel 호환을 위해 `utf-8-sig`를 유지한다.

## 우선순위 과제

### P0

1. 한글 mojibake 복구 및 재발 방지 테스트 추가
   - 코드, 테스트, 문서에 유니코드 replacement character, 깨진 `예측 이유` 컬럼명, 깨진 종가매매 안내 문구가 남지 않도록 검사한다.
2. 신호 정책 계약 강화
   - `recommendation_from_signal()`은 호환 인자를 유지하되 추천 결과는 `predicted_return`만 사용한다.
   - 뉴스/공시 append 후 `predicted_return`, `predicted_close`, `recommendation`, `portfolio_action`, `signal_score`가 바뀌지 않아야 한다.
3. 백테스트 랭킹 기준 명확화
   - long-only top-k 백테스트는 `predicted_return` 내림차순으로 선택한다.
   - `signal_score`는 동률 보조 또는 연구용 점수로만 사용하고, 뉴스/공시 표시 컬럼은 선택에 반영하지 않는다.

### P1

1. `KakaoColabPredictionBot` 분해
   - routes, session_state, cache, jobs, formatters, live_context로 책임을 나눈다.
2. `run_pipeline()` 단계 분해
   - 데이터 준비, 컨텍스트 수집, 피처 생성, 검증, 예측, 백테스트/figure, 산출물 저장을 작은 함수로 분리한다.
3. 외부 연동 에러 모델 표준화
   - credential_missing, timeout, rate_limited, empty_response, schema_changed, network_error 등으로 원인을 기록한다.
4. 의존성 관리 정리
   - `pyproject.toml`과 `requirements.txt`의 버전 제약을 일관화한다.

### P2

1. 피처 레지스트리 도입
   - feature name, source, allowed_for_model, allowed_for_signal, allowed_for_display, leakage_risk를 명시한다.
2. 모델 메타데이터/아티팩트 추적 강화
   - backend, feature hash, training window, horizon list, seed를 report에 저장한다.
3. 성능 개선
   - 외부 데이터/뉴스/DART 캐시 TTL, figure 생성 제한/병렬화, `build_features()` vectorization을 검토한다.
4. 운영 문서 정리
   - architecture, operations, live integration failure handling 문서를 보강한다.

## 이번 수정 반영 범위

- 추천 정책을 next-day `predicted_return` 전용으로 고정.
- 백테스트 및 PM 리포트의 top 후보 정렬을 `predicted_return` 우선으로 변경.
- 콘솔 상위 표시 정렬에서 `signal_score` 의존 제거.
- display-only 뉴스/공시 컨텍스트 불변 계약 테스트 추가.
- mojibake 금지 테스트 추가 및 이 문서/기존 리뷰 문서의 깨진 예시 정리.

## 검증 명령

```powershell
pytest tests/test_signal_policy_contract.py tests/test_backtest_and_calibration.py tests/test_console_summary.py tests/test_p0_import_and_encoding.py
pytest tests/test_pipeline_smoke.py
pytest
```
