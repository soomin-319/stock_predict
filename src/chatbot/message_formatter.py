from __future__ import annotations

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
            clean_line = raw_line.strip("-• ").strip()
            if not clean_line:
                continue
            parts = [part.strip() for part in re.split(split_pattern, clean_line) if part.strip()]
            for part in parts:
                if part and part not in bullets:
                    bullets.append(part)
        return bullets or [normalized.strip()]

    def format_percent(self, value: Any) -> str:
        if isinstance(value, str) and value.strip().endswith("%"):
            return value.strip()
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):.3f}%"

    def format_price(self, value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            cleaned = re.sub(r"[^\d.\-]", "", cleaned)
        else:
            cleaned = value
        numeric = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):,.0f}원"

    def format_confidence(self, value: Any) -> str:
        raw_value = value.strip() if isinstance(value, str) else value
        if isinstance(raw_value, str) and raw_value.endswith("%"):
            numeric = pd.to_numeric(pd.Series([raw_value[:-1]]), errors="coerce").iloc[0]
            display = raw_value
            if not pd.isna(numeric):
                numeric = float(numeric) / 100.0
        else:
            numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            display = None
        if pd.isna(numeric):
            return "-"
        if numeric >= 0.67:
            label = "높음"
        elif numeric >= 0.34:
            label = "보통"
        else:
            label = "낮음"
        if display is None:
            display = f"{float(numeric) * 100.0:.1f}%"
        return f"{display} ({label})"
