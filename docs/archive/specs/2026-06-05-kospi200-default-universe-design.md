# KOSPI200 기본 유니버스 전환 설계

## 목표

프로젝트 기본 종목 유니버스를 KOSPI50 + KOSDAQ50에서 KOSPI200으로 전환한다. 사용자가 실데이터 수집 종목을 명시하지 않으면 KOSPI200 전체 200종목을 수집한다. KOSDAQ 종목은 기본 처리 대상에서 제외한다.

## 기본 데이터 소스

- `data/kospi200_symbol_name_map.csv`를 유일한 기본 유니버스 CSV로 사용한다.
- `Symbol` 컬럼에서 순서를 보존하며 중복 없는 200개 종목을 읽는다.
- 기존 `data/default_universe_kospi50_kosdaq50.csv`는 삭제한다.
- KOSPI200 구성 종목 갱신은 `data/kospi200_symbol_name_map.csv` 교체로 수행한다.

## 코드 변경

- `src/data/universe.py`
  - 기본 유니버스 경로를 `data/kospi200_symbol_name_map.csv`로 변경한다.
- `src/data/cli_refresh.py`
  - 기본 fallback 수집의 5종목 제한을 제거한다.
  - 종목 미지정 시 KOSPI200 전체를 반환한다.
- `src/recommendation/realtime_close_betting.py`
  - 기본 추천 스캔 데이터 소스를 KOSPI200 CSV로 변경한다.
  - 기존 기본 제한값이 KOSPI200 전체를 포함하는지 보장한다.
- `src/chatbot/kakao_colab_bot.py`
  - 런타임 캐시 서명이 KOSPI200 CSV 변경을 추적하도록 경로를 변경한다.
- `src/config/settings.py`
  - 기본 유니버스 이름과 예상 크기를 KOSPI200, 200으로 변경한다.
- 문서와 테스트에서 기존 KOSPI50 + KOSDAQ50 기본값 참조를 KOSPI200으로 변경한다.

## 데이터 흐름

1. 사용자가 `--real-symbols` 또는 `--universe-csv`를 지정하면 해당 값을 우선 사용한다.
2. 둘 다 없으면 `data/kospi200_symbol_name_map.csv`에서 200종목 전체를 읽는다.
3. `--fetch-real`은 전체 데이터를 새로 저장한다.
4. `--auto-refresh-real`은 같은 200종목의 최신 데이터를 증분 추가한다.
5. 일반 파이프라인 입력 필터링 규칙은 유지한다. `--universe-csv`가 없으면 입력 CSV에 존재하는 종목을 처리한다.

## 오류 처리

- 기본 CSV가 없거나 `Symbol` 컬럼이 없거나 비어 있으면 기존 유니버스 로더 오류를 그대로 발생시킨다.
- 명시적 `--universe-csv` 로딩 실패 시 기존 동작대로 기본 KOSPI200 목록으로 fallback한다.

## 테스트

- 기본 유니버스 로더가 KOSPI200 CSV 경로와 200종목을 사용하는지 검증한다.
- 기본 fallback 수집이 5개가 아닌 200개 전체를 반환하는지 검증한다.
- 추천 서비스가 기본 KOSPI200 CSV를 읽는지 검증한다.
- 챗봇 캐시 서명이 KOSPI200 CSV 변경을 감지하는지 검증한다.
- 설정 기본값이 KOSPI200과 200인지 검증한다.
- 관련 테스트와 전체 `pytest`를 실행한다.

## 비범위

- KOSPI200 구성 종목을 외부 서비스에서 동적으로 수집하는 기능
- 일반 파이프라인 입력을 항상 KOSPI200으로 강제 필터링하는 변경
- 뉴스·공시 데이터를 예상수익률 또는 매매 신호에 반영하는 변경
