from __future__ import annotations

from src.news_impact.stock_factors.output_schema import FactorCode


FACTOR_KEYWORDS: dict[FactorCode, tuple[str, ...]] = {
    "US_RISK": ("S&P500", "Nasdaq", "나스닥", "SOX", "필라델피아 반도체", "Nvidia", "엔비디아", "VIX"),
    "US_RATE": ("Fed", "FOMC", "미국 금리", "미국 10년물", "실질금리", "DXY", "달러 인덱스"),
    "FX_KRW": ("원/달러", "USD/KRW", "환율", "원화 약세", "원화 강세"),
    "SEMI": ("반도체", "HBM", "DRAM", "D램", "NAND", "낸드", "AI 서버", "CapEx", "삼성전자", "SK하이닉스"),
    "EXPORT": ("수출", "무역수지", "1~20일", "월간 수출", "수출입"),
    "CHINA": ("중국", "미중", "G2", "관세", "중국 부동산", "중국 소비"),
    "FLOW": ("외국인", "기관", "개인", "선물", "ETF", "신용잔고", "반대매매", "레버리지"),
    "BOK_CREDIT": ("BOK", "한국은행", "기준금리", "CPI", "물가", "가계부채", "PF"),
    "GOVERNANCE": ("밸류업", "배당", "자사주", "소각", "상법", "지배구조", "주주환원"),
    "ACCESS": ("공매도", "MSCI", "영문공시", "FX 개방", "옴니버스 계좌", "시장 접근성"),
    "GEO_OIL": ("중동", "호르무즈", "유가", "WTI", "Brent", "브렌트", "LNG", "이스라엘", "이란"),
}


FACTOR_LABELS: dict[FactorCode, str] = {
    "US_RISK": "미국 증시·위험선호",
    "US_RATE": "미국 금리·달러 유동성",
    "FX_KRW": "원/달러 환율",
    "SEMI": "반도체 사이클",
    "EXPORT": "수출·무역수지",
    "CHINA": "중국 경기·G2 리스크",
    "FLOW": "수급",
    "BOK_CREDIT": "국내 금리·신용",
    "GOVERNANCE": "주주환원·지배구조",
    "ACCESS": "제도·시장 접근성",
    "GEO_OIL": "지정학·유가",
}


FACTOR_ORDER: tuple[FactorCode, ...] = (
    "US_RISK",
    "US_RATE",
    "FX_KRW",
    "SEMI",
    "EXPORT",
    "CHINA",
    "FLOW",
    "BOK_CREDIT",
    "GOVERNANCE",
    "ACCESS",
    "GEO_OIL",
)
