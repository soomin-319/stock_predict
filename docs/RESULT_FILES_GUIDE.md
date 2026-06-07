# `result/` 파일 설명서

`result/`는 파이프라인 예측 결과, 실행 리포트, 그래프, 챗봇 상태, 테스트 임시 파일을 저장하는 로컬 산출물 영역이다.

> 뉴스와 공시는 표시·검토용 컨텍스트다. `predicted_return`, 권고, 자동 신호 결정에는 영향을 주지 않는다.

## 핵심 예측 CSV

### `result/result_simple.csv`

사용자와 챗봇이 주로 읽는 간단 예측 결과다.

- 종목코드, 종목명
- 매수·매도·보유 권고
- 내일 예상 종가
- 내일 예상 수익률
- 상승확률
- 예측 신뢰도
- 공시·뉴스 요약

권고는 다음 날 예상 수익률인 `predicted_return`을 기준으로 결정된다.

### `result/result_detail.csv`

분석과 문제 진단을 위한 전체 상세 결과다.

- OHLCV 원본 데이터
- 이동평균, RSI, MACD, ATR 등 기술지표
- 거래대금, 회전율, 투자자 수급 지표
- 모델 예측 수익률, 예상 종가, 상승확률
- 불확실성, 신뢰도, 위험 플래그
- 권고, 포트폴리오 행동, 거래 게이트
- 백테스트 요약
- 예측 근거, 이슈 요약, 주의사항

### `result/result_news.csv`

오늘 날짜의 종목별 뉴스 원문과 요약을 저장한다.

- 종목코드, 종목명
- 뉴스 제목
- 게시 시각, 제공자, URL
- 뉴스 요약, 종합 판단, 주의사항

뉴스가 없으면 종목별 요약 행이 대신 저장될 수 있다.

### `result/result_disclosure.csv`

오늘 날짜의 DART 공시 원문과 요약을 저장한다. 구조는 `result_news.csv`와 유사하다.

## 실행 리포트 JSON

### `result/pm_report.json`

포트폴리오 운영 관점의 압축 요약이다.

- 데이터 coverage gate 상태
- 포트폴리오 행동별 종목 수
- 위험 플래그별 종목 수
- 평균 예상 수익률과 상승확률
- 상위 매수 후보 최대 10개

### `result/pipeline_report_*.json`, `result/report.json`

파이프라인 전체 실행 진단 리포트다.

- 사용 유니버스와 종목 수
- 피처 수와 실행 설정
- Walk-forward 검증 결과
- 기준 모델 성능
- 신호 튜닝 결과
- 백테스트 성과
- 외부 데이터·수급 데이터 coverage
- 확률 calibration
- 예측 생성 성공·누락 수
- 생성된 CSV와 그래프 경로

현재 사용되는 대표 파일:

| 파일 | 의미 |
|---|---|
| `pipeline_report_smoke.json` | 샘플 데이터 smoke 실행 결과 |
| `pipeline_report_added_symbols.json` | 종목 추가 실행 결과 |
| `pipeline_report_without_news.json` | 뉴스 컨텍스트 없이 실행한 결과 |
| `pipeline_report_with_context.json` | 외부·수급 컨텍스트 포함 실행 결과 |
| `pipeline_report_analysis.json` | 분석 목적으로 저장한 실행 결과 |
| `report.json` | 일반 이름으로 저장한 실행 결과 |

파일명은 `--report-json` 인자로 지정할 수 있다.

## 챗봇 상태와 캐시

### `result/chatbot_jobs.json`

챗봇 백그라운드 작업 상태를 기록한다.

- 실행 명령과 로그 경로
- 제출·완료 시각
- 실행 상태
- PID와 종료 코드

`__bootstrap__` 항목은 초기 전체 준비 작업이다.

### `result/chatbot_sessions.json`

사용자별 챗봇 대화 상태를 저장한다.

- 사용자 식별 해시
- 마지막 조회 종목
- 마지막 요청 의도
- 마지막 갱신 시각

종목명이 생략된 후속 요청을 처리할 때 사용한다.

### `result/prewarm_cache_meta.json`

챗봇 시작 시 전체 예측을 다시 실행할지 판단하는 캐시 메타데이터다.

- 입력 CSV 수정 시각과 크기
- 기본 유니버스 상태
- 주요 실행 옵션
- 보고서와 그래프 경로
- 설정 전체의 해시값

입력과 설정이 동일하면 불필요한 재실행을 건너뛸 수 있다.

### `result/chatbot_logs/*.log`

챗봇이 실행한 subprocess와 bootstrap 작업의 표준 출력·오류 로그다.

- `{종목}_{timestamp}.log`
- `bootstrap_{timestamp}.log`
- `recommendation_{timestamp}.log`

## 그래프 폴더

실행 옵션의 `--figure-dir` 값에 따라 다음과 같은 폴더가 생성될 수 있다.

- `result/figures/`
- `result/figures_smoke/`
- `result/figures_added_symbols/`
- `result/figures_without_news/`
- `result/figures_with_context/`

공통 그래프:

| 파일 | 설명 |
|---|---|
| `equity_curve.png` | 백테스트 누적 자산 곡선 |
| `drawdown_curve.png` | 고점 대비 손실률 곡선 |
| `signal_score_hist.png` | 종목별 신호 점수 분포 |
| `actual_vs_predicted_return.png` | 전체 종목 일별 평균 실제·예측 수익률 비교 |
| `actual_vs_predicted_price.png` | 전체 종목 평균 실제·예측 다음 종가 비교 |
| `up_probability_calibration.png` | 예측 상승확률과 실제 상승비율 비교 |
| `uncertainty_vs_error.png` | 모델 불확실성과 실제 예측 오차 관계 |
| `symbol_summary_table_top20.png` | 상위 20개 종목 예측 요약표 |

종목별 그래프:

| 파일 패턴 | 설명 |
|---|---|
| `symbol_level/{종목}_actual_vs_predicted_price.png` | 종목별 실제·예측 다음 종가 |
| `symbol_level/{종목}_actual_vs_predicted_return.png` | 종목별 실제·예측 수익률 |
| `symbol_level/recent_month/{종목}_recent_month_price.png` | 최근 한 달 종가 비교 |
| `symbol_level/recent_month/{종목}_recent_month_return.png` | 최근 한 달 수익률 비교 |

## pytest 캐시와 임시 파일

### `result/.pytest_cache/`

pytest 실행 속도와 재실행 정보를 저장한다.

| 파일 | 설명 |
|---|---|
| `.gitignore` | 캐시 Git 추적 방지 |
| `CACHEDIR.TAG` | 캐시 디렉터리 표준 표시 |
| `README.md` | pytest 캐시 설명 |
| `v/cache/lastfailed` | 이전 실패 테스트 목록 |
| `v/cache/nodeids` | 발견된 테스트 목록 |

### 테스트 임시 폴더

- `result/.pytest_tmp/`
- `result/analysis_pytest_tmp/`
- `result/pr_pytest_tmp/`

테스트 격리 환경에서 생성한 CSV, JSON, 모델, 로그, 그래프가 반복 저장된다. 실제 운영 결과가 아니므로 테스트 종료 후 삭제할 수 있다.

## 기타 파일

### `result/codex-mcp-test/config.toml`

Codex/MCP 연결 테스트 설정이다. 주식 예측 파이프라인의 핵심 산출물은 아니다.

### 뉴스 영향 모듈 산출물

뉴스 영향 모듈의 출력 디렉터리를 `result/` 아래로 지정하면 다음 파일이 생성될 수 있다.

- `raw_snapshot.json`: 수집된 원본 이벤트
- `normalized_snapshot.json`: 정규화된 이벤트
- `impact_events.json`: 영향 분석 이벤트
- `audit.json`: 처리 감사 정보
- `report.json`, `report.csv`: 뉴스 영향 분석 결과
- `llm_cache/impact_judgments/*.json`: LLM 영향 판단 캐시
- `llm_cache/semantic_clusters/*.json`: 의미 군집화 캐시

이 데이터도 예측 수익률과 권고를 변경하지 않는 표시·검토용 정보다.

## 권장 확인 순서

1. `pipeline_report_*.json`: 실행 성공, 검증, coverage 확인
2. `result_simple.csv`: 최종 사용자용 결과 확인
3. `result_detail.csv`: 예측 근거와 이상값 분석
4. `figures*/`: 백테스트와 예측 품질 시각 확인
5. `result_news.csv`, `result_disclosure.csv`: 표시용 이슈 확인

## 정리 원칙

`result/`는 생성 산출물 영역이다. 보존할 보고서를 먼저 백업한 후 정리한다.

자세한 안전 정리 절차는 [`RESULT_CLEANUP.md`](RESULT_CLEANUP.md)를 참고한다.
