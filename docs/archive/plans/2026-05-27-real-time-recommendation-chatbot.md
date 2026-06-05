# Real-Time Recommendation Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a modular real-time close-betting recommendation feature to `stock_predict` and trigger it from KakaoTalk when the user types `추천`.

**Architecture:** Create a focused `src/recommendation` package that ports the `m_stock_predict` scoring/selection flow using the existing `stock_predict` real-data fetcher. Inject a recommendation service into `KakaoColabPredictionBot`, route the `추천` utterance before stock-name lookup, and format results as Kakao simpleText.

**Tech Stack:** Python 3.10+, pandas, existing `src.data.fetch_real_data.fetch_real_ohlcv`, pytest, Flask Kakao webhook.

---

## File Structure

- Create `src/recommendation/__init__.py`: package exports.
- Create `src/recommendation/close_betting.py`: dataclass DTO, technical indicators, scoring, candidate selection, message formatting.
- Create `src/recommendation/realtime_close_betting.py`: real-time service using injected fetcher/symbol provider.
- Create `tests/test_realtime_close_betting.py`: unit tests for service and formatting.
- Modify `src/chatbot/kakao_colab_bot.py`: add `추천` keyword, injected service, handler, quick reply.
- Modify `tests/test_kakao_colab_bot.py`: tests for `추천` success/failure and guide quick reply.
- Modify `pyproject.toml`: include `src.recommendation` in setuptools packages.

---

### Task 1: Recommendation module tests

**Files:**
- Create: `tests/test_realtime_close_betting.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from src.recommendation.close_betting import format_recommendation_message
from src.recommendation.realtime_close_betting import RealTimeCloseBettingRecommendationService


def _history(symbol: str, name: str, close: int, volume: int) -> pd.DataFrame:
    rows = []
    for i in range(260):
        current = close + i
        rows.append(
            {
                "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "Symbol": symbol,
                "Open": current - 1,
                "High": current + 2,
                "Low": current - 2,
                "Close": current,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows)


def test_realtime_service_scores_and_ranks_recommendations():
    raw = pd.concat(
        [
            _history("005930.KS", "삼성전자", 70000, 1000000),
            _history("000660.KS", "SK하이닉스", 120000, 900000),
        ],
        ignore_index=True,
    )

    service = RealTimeCloseBettingRecommendationService(
        symbols_provider=lambda: pd.DataFrame(
            [
                {"Symbol": "005930.KS", "Name": "삼성전자", "Market": "KOSPI"},
                {"Symbol": "000660.KS", "Name": "SK하이닉스", "Market": "KOSPI"},
            ]
        ),
        ohlcv_fetcher=lambda symbols, start, end: raw,
        today_provider=lambda: date(2026, 5, 27),
    )

    recommendations = service.get_recommendations(top_n=2)

    assert [item.rank for item in recommendations] == [1, 2]
    assert recommendations[0].symbol == "005930"
    assert recommendations[0].name == "삼성전자"
    assert recommendations[0].final_score >= 100
    assert recommendations[0].reasons


def test_format_recommendation_message_includes_rank_symbol_score_and_reason():
    service = RealTimeCloseBettingRecommendationService(
        symbols_provider=lambda: pd.DataFrame([{"Symbol": "005930.KS", "Name": "삼성전자", "Market": "KOSPI"}]),
        ohlcv_fetcher=lambda symbols, start, end: _history("005930.KS", "삼성전자", 70000, 1000000),
        today_provider=lambda: date(2026, 5, 27),
    )

    text = format_recommendation_message(service.get_recommendations(top_n=1), as_of=date(2026, 5, 27))

    assert "[실시간 추천]" in text
    assert "기준일: 2026-05-27" in text
    assert "1위 삼성전자(005930)" in text
    assert "점수:" in text
    assert "근거:" in text
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_realtime_close_betting.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.recommendation'`.

---

### Task 2: Implement recommendation package

**Files:**
- Create: `src/recommendation/__init__.py`
- Create: `src/recommendation/close_betting.py`
- Create: `src/recommendation/realtime_close_betting.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add minimal implementation**

Implement:

- `CloseBettingRecommendation`
- `add_technical_indicators()`
- `score_candidates()`
- `select_close_betting_candidates()`
- `format_recommendation_message()`
- `RealTimeCloseBettingRecommendationService.get_recommendations()`

The service must normalize `Date/Symbol/Open/High/Low/Close/Volume` to the m_stock-style schema, rank by latest `close * volume`, score latest rows, and return DTOs.

- [ ] **Step 2: Run recommendation tests**

Run: `pytest tests/test_realtime_close_betting.py -q`

Expected: PASS.

---

### Task 3: Chatbot tests for `추천`

**Files:**
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Write failing chatbot tests**

Add helper fake service:

```python
class FakeRecommendationService:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = 0

    def get_recommendations(self, top_n=3):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result[:top_n]
```

Add tests:

```python
def test_recommendation_keyword_returns_realtime_recommendations(tmp_path: Path):
    from src.recommendation.close_betting import CloseBettingRecommendation

    fake_service = FakeRecommendationService(
        [
            CloseBettingRecommendation(
                rank=1,
                symbol="005930",
                name="삼성전자",
                grade="강력 후보",
                final_score=120,
                first_buy_ratio=0.6,
                reasons=("52주 종가 기준 신고가", "거래대금 1위"),
            )
        ]
    )
    bot = make_bot(tmp_path)
    bot.recommendation_service = fake_service

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "추천", "user": {"id": "u-rec"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert fake_service.calls == 1
    assert "[실시간 추천]" in text
    assert "1위 삼성전자(005930)" in text
    assert "점수: 120" in text


def test_recommendation_keyword_returns_failure_message_when_service_raises(tmp_path: Path):
    bot = make_bot(tmp_path)
    bot.recommendation_service = FakeRecommendationService(error=RuntimeError("network down"))

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "추천", "user": {"id": "u-rec-fail"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "실시간 추천 생성에 실패했습니다" in text
    assert "다시 '추천'" in text


def test_guide_response_includes_recommendation_quick_reply(tmp_path: Path):
    bot = make_bot(tmp_path)

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "도움말", "user": {"id": "u-guide-rec"}}})

    assert any(reply["messageText"] == "추천" for reply in response["template"]["quickReplies"])
```

- [ ] **Step 2: Run chatbot focused tests to verify RED**

Run: `pytest tests/test_kakao_colab_bot.py -q`

Expected: FAIL because `추천` is treated as a stock-name lookup and quick reply is absent.

---

### Task 4: Wire `추천` into Kakao chatbot

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Add imports and keyword**

```python
from src.recommendation.close_betting import format_recommendation_message
from src.recommendation.realtime_close_betting import RealTimeCloseBettingRecommendationService

_RECOMMENDATION_KEYWORDS = {"추천"}
```

- [ ] **Step 2: Add injectable service**

In `KakaoColabPredictionBot.__init__`, add parameter:

```python
recommendation_service: Any | None = None,
```

Set:

```python
self.recommendation_service = recommendation_service or RealTimeCloseBettingRecommendationService()
```

- [ ] **Step 3: Add command branch**

In `handle_utterance()`, before `_extract_stock_code()`:

```python
if self._is_recommendation_request(text):
    return self._handle_recommendation_request()
```

Add methods:

```python
def _is_recommendation_request(self, text: str) -> bool:
    return text.strip().lower() in _RECOMMENDATION_KEYWORDS


def _handle_recommendation_request(self) -> dict[str, Any]:
    try:
        recommendations = self.recommendation_service.get_recommendations(top_n=3)
        return self._build_response(
            format_recommendation_message(recommendations),
            quick_replies=[("다시 추천", "추천"), ("도움말", "도움말")],
        )
    except Exception as exc:
        self._console_log(f"실시간 추천 처리 오류({type(exc).__name__}): {exc}")
        return self._build_response(
            "실시간 추천 생성에 실패했습니다.\n데이터 수집 또는 네트워크 상태를 확인한 뒤 다시 '추천'을 입력해주세요.",
            quick_replies=[("다시 추천", "추천"), ("도움말", "도움말")],
        )
```

- [ ] **Step 4: Update guide quick replies**

Add guide line `5) 실시간 추천: 추천` and quick reply `("실시간 추천", "추천")`.

- [ ] **Step 5: Run chatbot tests**

Run: `pytest tests/test_kakao_colab_bot.py -q`

Expected: PASS.

---

### Task 5: Final verification

**Files:**
- All touched files

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py -q`

Expected: PASS.

- [ ] **Step 2: Run git diff summary**

Run: `git diff --stat`

Expected: Shows only recommendation package, chatbot, tests, pyproject, docs plan/spec changes for this feature plus pre-existing unrelated dirty files.

---

## Self-Review

- Spec coverage: recommendation module, `추천` command, formatting, failure handling, and tests are covered.
- Placeholder scan: no TBD/TODO/implement-later steps remain.
- Type consistency: DTO and service names match across tests, implementation, and chatbot wiring.
