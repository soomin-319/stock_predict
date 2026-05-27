# 실시간 종가배팅 추천 챗봇 통합 설계

**날짜:** 2026-05-27

## 목표

`m_stock_predict`의 종가배팅 추천 기능을 `stock_predict` 프로젝트 안으로 모듈화해 추가한다. 사용자가 카카오톡 챗봇에서 `추천`을 입력하면 실시간으로 KOSPI 데이터를 수집하고 추천 후보 상위 종목을 카카오 simpleText 응답으로 보여준다.

## 범위

### 포함

- `stock_predict/src/recommendation/` 패키지 추가
- 실시간 추천 서비스 함수 추가
- `추천` 챗봇 명령 추가
- 카카오 응답 포맷 추가
- 단위 테스트 추가

### 제외

- 기존 개별 종목 예측 파이프라인 변경
- 기존 `결과`, `최신화`, 종목코드/종목명 명령 의미 변경
- 실시간 추천 결과 캐싱
- 백그라운드 추천 작업 큐

## 선택한 접근

모듈 이식 방식이다. `m_stock_predict`의 핵심 추천 흐름을 `stock_predict` 내부 패키지로 옮긴다.

이 방식은 `stock_predict` 단일 프로젝트만 실행하면 카카오 챗봇과 추천 기능이 함께 동작한다. Colab/GitHub 배포도 별도 로컬 패키지 의존 없이 단순하다.

## 사용자 흐름

1. 사용자가 카카오톡에서 `추천` 입력
2. `KakaoColabPredictionBot.handle_utterance()`가 추천 명령으로 인식
3. `RealTimeCloseBettingRecommendationService.get_recommendations()` 호출
4. 서비스가 실시간 OHLCV 수집 및 추천 후보 산출
5. 챗봇이 상위 N개 추천 결과를 simpleText로 응답

예시 응답:

```text
[실시간 추천]
기준일: 2026-05-27

1위 삼성전자(005930) - 매수 후보
점수: 85 / 1차 매수비중: 60%
근거: 거래대금 상위, 20일 신고가 근접

2위 SK하이닉스(000660) - 매수 후보
점수: 80 / 1차 매수비중: 60%
근거: 거래대금 상위, 단기 추세 양호
```

## 파일 설계

### `src/recommendation/__init__.py`

추천 패키지 공개 API를 관리한다.

### `src/recommendation/close_betting.py`

`m_stock_predict`의 후보 선택, 점수 결과 정리, 응답용 DTO를 담당한다.

### `src/recommendation/realtime_close_betting.py`

실시간 추천 서비스 진입점이다.

주요 인터페이스:

```python
@dataclass(frozen=True)
class CloseBettingRecommendation:
    rank: int
    symbol: str
    name: str
    grade: str
    final_score: int
    first_buy_ratio: float
    reasons: tuple[str, ...]

class RealTimeCloseBettingRecommendationService:
    def get_recommendations(self, top_n: int = 3) -> list[CloseBettingRecommendation]:
        ...
```

실제 데이터 수집/점수 계산 함수는 테스트에서 주입 가능하게 만든다.

### `src/chatbot/kakao_colab_bot.py`

변경점:

- `_RECOMMENDATION_KEYWORDS = {"추천"}` 추가
- `KakaoColabPredictionBot.__init__()`에 `recommendation_service` 선택 인자 추가
- `handle_utterance()`에서 help/status/refresh보다 뒤, 종목명 lookup보다 앞에 추천 분기 추가
- `_handle_recommendation_request()` 추가
- `_format_recommendation_message()` 추가
- `_guide_response()` 빠른 답변에 `추천` 추가

## 오류 처리

추천 서비스에서 예외가 발생하면 챗봇은 다음 안내를 반환한다.

```text
실시간 추천 생성에 실패했습니다.
데이터 수집 또는 네트워크 상태를 확인한 뒤 다시 '추천'을 입력해주세요.
```

내부 예외는 `_console_log()`로 기록한다.

## 테스트 설계

- `tests/test_realtime_close_betting.py`
  - 추천 서비스가 데이터프레임/후보를 DTO로 변환하는지 검증
  - 빈 결과면 빈 리스트 반환 검증

- `tests/test_kakao_colab_bot.py`
  - `추천` 입력 시 추천 서비스 호출 검증
  - 추천 결과가 simpleText에 rank/name/symbol/score/grade로 표시되는지 검증
  - 추천 서비스 예외 시 실패 안내문 반환 검증
  - 도움말 quickReplies에 `추천` 포함 검증

## 성공 기준

- `pytest tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py -q` 통과
- `stock-predict-kakao` 실행 후 카카오 웹훅에서 `추천` 입력 시 실시간 추천 응답 생성
- 기존 개별 종목 예측 테스트 동작 유지
