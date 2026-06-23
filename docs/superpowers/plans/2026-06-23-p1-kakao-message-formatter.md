# P1 Kakao Message Formatter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository `AGENTS.md` forbids subagents, so execute inline only.

**Goal:** Extract cached prediction message formatting from `KakaoColabPredictionBot` into a focused formatter module without changing user-visible output or recommendation behavior.

**Architecture:** Add `PredictionMessageFormatter` in `src/chatbot/message_formatter.py`. Keep `KakaoColabPredictionBot` compatibility wrapper methods, but delegate formatting to one formatter instance. Tests lock output parity and display-only guard behavior.

**Tech Stack:** Python 3.10+, pandas, pytest, existing Kakao chatbot tests.

---

## File Structure

- Create `src/chatbot/message_formatter.py`
  - Owns pure formatting functions for cached prediction rows.
  - Has no Flask, subprocess, session, network, or filesystem dependency.
- Modify `src/chatbot/kakao_colab_bot.py`
  - Imports and instantiates `PredictionMessageFormatter`.
  - Replaces method bodies for formatting helpers with delegation wrappers.
- Modify `tests/test_kakao_colab_bot.py`
  - Adds tests for direct formatter parity and wrapper delegation.

---

### Task 1: Add formatter parity tests

**Files:**
- Modify: `tests/test_kakao_colab_bot.py`
- No production changes in this task.

- [ ] **Step 1: Add failing tests**

Append these tests near existing cached prediction message formatting tests:

```python
def test_prediction_message_formatter_matches_bot_output_for_cached_row(tmp_path):
    from src.chatbot.message_formatter import PredictionMessageFormatter

    bot = make_bot(tmp_path)
    row = pd.Series(
        {
            "종목코드": "005930",
            "종목명": "삼성전자",
            "권고": "매수",
            "내일 예상 수익률(%)": 2.34,
            "상승확률(%)": 61.2,
            "내일 예상 종가": 71000,
            "예측 신뢰도": "높음",
            "예측 이유": "거래대금 상위 / 외국인 기관 순매수 / 나스닥 선물 +1%",
            "공시 요약": "신규 공급계약 체결",
            "뉴스 요약": "AI 반도체 수요 증가",
            "뉴스/공시 영향 점수": "긍정 2",
            "뉴스/공시 영향 요약": "참고용 호재",
            "뉴스/공시 영향 참고": "참고용·예측값 미반영",
        }
    )

    assert PredictionMessageFormatter().format_prediction_message(row) == bot._build_prediction_message_from_row(row)


def test_prediction_message_formatter_display_context_does_not_change_recommendation(tmp_path):
    from src.chatbot.message_formatter import PredictionMessageFormatter

    formatter = PredictionMessageFormatter()
    base = pd.Series(
        {
            "종목코드": "000001",
            "종목명": "테스트",
            "권고": "관망",
            "내일 예상 수익률(%)": 0.1,
            "상승확률(%)": 50.0,
            "내일 예상 종가": 1000,
            "예측 신뢰도": "보통",
        }
    )
    with_context = base.copy()
    with_context["뉴스 요약"] = "강한 호재"
    with_context["공시 요약"] = "대형 계약"
    with_context["뉴스/공시 영향 점수"] = "긍정 9"
    with_context["뉴스/공시 영향 요약"] = "표시 전용"

    rendered = formatter.format_prediction_message(with_context)

    assert "권고: 관망" in rendered
    assert "강한 호재" in rendered
    assert "표시 전용" in rendered
```

- [ ] **Step 2: Run test to verify import failure**

Run:

```bash
pytest tests/test_kakao_colab_bot.py::test_prediction_message_formatter_matches_bot_output_for_cached_row tests/test_kakao_colab_bot.py::test_prediction_message_formatter_display_context_does_not_change_recommendation -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.chatbot.message_formatter'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_kakao_colab_bot.py
git commit -m "Lock Kakao message formatter parity"
```

---

### Task 2: Create formatter module and delegate bot wrappers

**Files:**
- Create: `src/chatbot/message_formatter.py`
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Create formatter implementation**

Create `src/chatbot/message_formatter.py` with this implementation:

```python
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


class PredictionMessageFormatter:
    """Pure formatter for Kakao cached prediction rows."""

    def format_prediction_message(self, row: pd.Series) -> str:
        code = str(row.get("종목코드", "-"))
        name = str(row.get("종목명", "-"))
        recommendation = str(row.get("권고", "-"))
        predicted_return = self.format_percent(row.get("내일 예상 수익률(%)"))
        up_probability = self.format_percent(row.get("상승확률(%)"))
        predicted_close = self.format_price(row.get("내일 예상 종가"))
        confidence = self.format_confidence(row.get("예측 신뢰도"))
        reason_line = self.build_reason_line(row)
        issue_block = self.build_issue_summary_block(row)
        news_impact_block = self.build_news_impact_block(row)
        return (
            f"[{code} {name}]\n"
            f"권고: {recommendation}\n"
            f"상승확률: {up_probability}\n"
            f"내일 예측 수익률: {predicted_return}\n"
            f"내일 예측 종가: {predicted_close}\n"
            f"신뢰도: {confidence}\n"
            f"{reason_line}"
            f"{issue_block}"
            f"{news_impact_block}"
        )

    def build_reason_line(self, row: pd.Series) -> str:
        raw_reason = self.get_clean_issue_text(row.get("예측 이유"))
        if not raw_reason:
            raw_reason = self.get_clean_issue_text(row.get("예측 사유"))
        if not raw_reason:
            return ""

        labels: list[str] = []
        if "거래대금" in raw_reason:
            labels.append("거래대금 상위")
        if "외국인" in raw_reason and "기관" in raw_reason and "순매수" in raw_reason:
            labels.append("외국인/기관 순매수")
        if "나스닥" in raw_reason and "+1%" in raw_reason:
            labels.append("나스닥 선물 +1% 이상")

        if not labels:
            return ""
        return "사유: " + ", ".join(dict.fromkeys(labels)) + "\n"

    def build_issue_summary_block(self, row: pd.Series) -> str:
        disclosure_text = self.get_clean_issue_text(row.get("공시 요약"))
        news_text = self.get_clean_issue_text(row.get("뉴스 요약"))
        if not disclosure_text and not news_text:
            return ""
        disclosure_lines = self.to_bullet_lines(disclosure_text or "당일 공시 없음.")
        news_lines = self.to_bullet_lines(news_text or "당일 뉴스 없음.")
        return (
            "\n[공시 요약]\n"
            + "\n".join(f"- {line}" for line in disclosure_lines)
            + "\n\n[뉴스 요약]\n"
            + "\n".join(f"- {line}" for line in news_lines)
        )

    def build_news_impact_block(self, row: pd.Series) -> str:
        score_text = self.get_clean_issue_text(row.get("뉴스/공시 영향 점수"))
        summary_text = self.get_clean_issue_text(row.get("뉴스/공시 영향 요약"))
        note_text = self.get_clean_issue_text(row.get("뉴스/공시 영향 참고"))
        if not score_text and not summary_text:
            return ""
        lines = []
        if score_text:
            lines.append(f"- 점수: {score_text}")
        if summary_text:
            lines.append(f"- 요약: {summary_text}")
        lines.append(f"- 참고: {note_text or '참고용·예측값 미반영'}")
        return "\n\n[뉴스/공시 영향 점수]\n" + "\n".join(lines)

    def get_clean_issue_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        if not isinstance(raw, str) and pd.isna(raw):
            return ""
        text = str(raw).strip()
        if not text or text == "-":
            return ""
        return text

    def to_bullet_lines(self, text: str) -> list[str]:
        normalized = str(text).replace("\r", "\n")
        normalized = re.sub(r"^\[(공시 요약|뉴스 요약)\]\s*", "", normalized.strip(), flags=re.IGNORECASE)
        raw_lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if not raw_lines:
            return []

        bullets: list[str] = []
        split_pattern = r"\s*(?:/|\||;|·)\s+"
        for raw_line in raw_lines:
            line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", raw_line).strip()
            if not line:
                continue
            parts = [part.strip() for part in re.split(split_pattern, line) if part.strip()]
            bullets.extend(parts or [line])
        return bullets

    def format_percent(self, value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if math.isnan(numeric):
            return "-"
        return f"{numeric:.2f}%"

    def format_price(self, value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if math.isnan(numeric):
            return "-"
        return f"{numeric:,.0f}원"

    def format_confidence(self, value: Any) -> str:
        text = self.get_clean_issue_text(value)
        return text or "-"
```

- [ ] **Step 2: Wire formatter into bot**

In `src/chatbot/kakao_colab_bot.py`:

1. Add import near existing chatbot imports:

```python
from src.chatbot.message_formatter import PredictionMessageFormatter
```

2. In `KakaoColabPredictionBot.__init__`, add:

```python
self._message_formatter = PredictionMessageFormatter()
```

3. Replace formatting helper bodies with delegation:

```python
    def _build_prediction_message_from_row(self, row: pd.Series) -> str:
        return self._message_formatter.format_prediction_message(row)

    def _build_reason_line(self, row: pd.Series) -> str:
        return self._message_formatter.build_reason_line(row)

    def _build_issue_summary_block(self, row: pd.Series) -> str:
        return self._message_formatter.build_issue_summary_block(row)

    def _build_news_impact_block(self, row: pd.Series) -> str:
        return self._message_formatter.build_news_impact_block(row)

    def _get_clean_issue_text(self, raw: Any) -> str:
        return self._message_formatter.get_clean_issue_text(raw)

    def _to_bullet_lines(self, text: str) -> list[str]:
        return self._message_formatter.to_bullet_lines(text)

    def _format_percent(self, value: Any) -> str:
        return self._message_formatter.format_percent(value)

    def _format_price(self, value: Any) -> str:
        return self._message_formatter.format_price(value)

    def _format_confidence(self, value: Any) -> str:
        return self._message_formatter.format_confidence(value)
```

Keep `_format_prediction_message`, `_safe_format_prediction_message`, `_minimal_format_prediction_message`, and `_format_cached_prediction_message` unchanged except that they now reach the delegated builder.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/test_kakao_colab_bot.py::test_prediction_message_formatter_matches_bot_output_for_cached_row tests/test_kakao_colab_bot.py::test_prediction_message_formatter_display_context_does_not_change_recommendation -q
```

Expected: `2 passed`.

- [ ] **Step 4: Commit formatter extraction**

```bash
git add src/chatbot/message_formatter.py src/chatbot/kakao_colab_bot.py tests/test_kakao_colab_bot.py
git commit -m "Extract Kakao prediction message formatter"
```

---

### Task 3: Add delegation regression test

**Files:**
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Add delegation test**

Append:

```python
def test_kakao_bot_prediction_message_helpers_delegate_to_formatter(tmp_path, monkeypatch):
    bot = make_bot(tmp_path)
    row = pd.Series({"종목코드": "000001", "종목명": "테스트", "권고": "관망"})
    calls: list[str] = []

    class StubFormatter:
        def format_prediction_message(self, value):
            calls.append("format")
            assert value is row
            return "formatted"

        def build_reason_line(self, value):
            calls.append("reason")
            assert value is row
            return "reason"

        def build_issue_summary_block(self, value):
            calls.append("issue")
            assert value is row
            return "issue"

        def build_news_impact_block(self, value):
            calls.append("impact")
            assert value is row
            return "impact"

        def get_clean_issue_text(self, value):
            calls.append("clean")
            return "clean"

        def to_bullet_lines(self, value):
            calls.append("bullets")
            return ["bullet"]

        def format_percent(self, value):
            calls.append("percent")
            return "1.00%"

        def format_price(self, value):
            calls.append("price")
            return "1,000원"

        def format_confidence(self, value):
            calls.append("confidence")
            return "보통"

    bot._message_formatter = StubFormatter()

    assert bot._build_prediction_message_from_row(row) == "formatted"
    assert bot._build_reason_line(row) == "reason"
    assert bot._build_issue_summary_block(row) == "issue"
    assert bot._build_news_impact_block(row) == "impact"
    assert bot._get_clean_issue_text("x") == "clean"
    assert bot._to_bullet_lines("x") == ["bullet"]
    assert bot._format_percent(1) == "1.00%"
    assert bot._format_price(1000) == "1,000원"
    assert bot._format_confidence("보통") == "보통"
    assert calls == ["format", "reason", "issue", "impact", "clean", "bullets", "percent", "price", "confidence"]
```

- [ ] **Step 2: Run delegation test**

Run:

```bash
pytest tests/test_kakao_colab_bot.py::test_kakao_bot_prediction_message_helpers_delegate_to_formatter -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run impacted chatbot tests**

Run:

```bash
pytest tests/test_kakao_colab_bot.py tests/test_chatbot_helpers.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit delegation test**

```bash
git add tests/test_kakao_colab_bot.py
git commit -m "Cover Kakao formatter delegation"
```

---

### Task 4: Verification and PR

**Files:**
- No planned code changes unless verification reveals failures.

- [ ] **Step 1: Run impacted plus smoke tests**

Run:

```bash
pytest tests/test_kakao_colab_bot.py tests/test_chatbot_helpers.py tests/test_pipeline_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run sample pipeline**

Run:

```bash
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code 0.

- [ ] **Step 3: Run full suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short
git log --oneline --decorate -5
git diff --stat HEAD~3..HEAD
```

Expected: only formatter extraction, tests, spec, and plan are present.

- [ ] **Step 5: Push branch and open draft PR**

Push:

```bash
git push -u origin p1-kakao-message-formatter
```

Open a draft PR against `p2-signal-policy-unification` with:

- summary of formatter extraction;
- tests run;
- note that recommendation/signal behavior is unchanged;
- note that news/disclosure remains display-only.
