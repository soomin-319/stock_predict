from __future__ import annotations


FRESHNESS_KEYWORDS: tuple[str, ...] = (
    "오늘",
    "최신",
    "현재",
    "방금",
    "최근",
    "기준금리",
    "BOK",
    "Fed",
    "FOMC",
    "환율",
    "USD/KRW",
    "DXY",
    "KOSPI",
    "KOSDAQ",
    "외국인",
    "수출",
    "무역수지",
    "반도체 수출",
    "MSCI",
    "공매도",
    "상법",
    "밸류업",
    "유가",
    "WTI",
    "Brent",
    "브렌트",
    "호르무즈",
    "SOX",
    "HBM",
)


_ITEM_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BOK 기준금리", ("BOK", "한국은행", "기준금리")),
    ("Fed/FOMC", ("Fed", "FOMC")),
    ("USD/KRW", ("USD/KRW", "원/달러", "환율", "원화 약세", "원화 강세")),
    ("DXY", ("DXY", "달러 인덱스")),
    ("KOSPI/KOSDAQ", ("KOSPI", "KOSDAQ")),
    ("외국인 수급", ("외국인",)),
    ("수출/무역수지", ("수출", "무역수지", "반도체 수출")),
    ("MSCI", ("MSCI",)),
    ("공매도", ("공매도",)),
    ("상법/밸류업", ("상법", "밸류업")),
    ("WTI/Brent", ("유가", "WTI", "Brent", "브렌트")),
    ("중동 리스크", ("호르무즈", "중동", "이스라엘", "이란")),
    ("SOX", ("SOX", "필라델피아 반도체")),
    ("HBM 수요 전망", ("HBM",)),
)


def freshness_required(text: str) -> bool:
    normalized = _lower(text)
    return any(_lower(keyword) in normalized for keyword in FRESHNESS_KEYWORDS)


def detect_freshness_items(text: str) -> tuple[str, ...]:
    normalized = _lower(text)
    items: list[str] = []
    for label, needles in _ITEM_RULES:
        if any(_lower(needle) in normalized for needle in needles):
            items.append(label)
    return tuple(items)


def _lower(value: str) -> str:
    return value.casefold()
