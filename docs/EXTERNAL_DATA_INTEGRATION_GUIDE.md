# 외부 데이터 소스 연동 가이드 (거래대금/공시/뉴스/수급)

요청하신 아래 항목 중 가격/거래대금/수급/52주 고점 관련 데이터는 예측과 진단에 활용할 수 있고, 공시/뉴스는 사용자에게 보여주는 표시용 컨텍스트로만 다루기 위한 실무 설계입니다. 클라이언트 운영 정책상 매수/매도 신호는 다음날 예상 수익률만 참고하며, 공시/뉴스는 예상 수익률에 영향을 주지 않습니다.

- 거래대금 상위 종목
- 호재/공시/뉴스 표시
- 외국인/기관 매수 종목
- 52주 신고가 근처/돌파 종목

## 1) 항목별 추천 소스

### A. 거래대금 상위 10개
- **현재 코드만으로도 계산 가능**: `Close * Volume`.
- 이 값으로 일자별 `top10` 플래그 생성(`is_top_turnover_10`)을 이미 반영.

### B. 공시 데이터
- **OpenDART API (권장)**: 금융감독원 전자공시(사업/분기/주요사항 공시).
- 연동 방식:
  1. 종목코드 ↔ corp_code 매핑 테이블 구축.
  2. 일자별 공시 건수/유형(유상증자, 실적정정, 수주 등) 집계.
  3. 원문/요약/건수/기준일을 일자-종목 키로 저장해 리포트와 챗봇 응답에 표시.
  4. 운영 정책상 `disclosure_score` 같은 점수는 예상 수익률 또는 매수/매도 신호에 반영하지 않음.

### C. 호재/뉴스
- 소스 옵션:
  - 네이버/다음 증권 뉴스 API/크롤링(서비스 약관 준수 필요)
  - 상용 뉴스 API(예: Finnhub, RavenPack, AlphaSense 등)
- 연동 방식:
  1. 헤드라인 수집 → 종목 매핑(NER/룰 기반).
  2. 중복 기사 제거 후 종목-일자 기준 원문 목록과 요약 생성.
  3. 필요하면 감성/관련도 점수를 보조 메타데이터로 저장할 수 있으나, 예상 수익률 또는 매수/매도 신호에는 반영하지 않음.

### D. 외국인/기관 수급
- **pykrx (권장)**: 국내 주식 투자자별 순매수(외국인/기관/개인) 조회 가능.
- 연동 방식:
  1. 일자별 종목 순매수 데이터 수집.
  2. `foreign_net_buy`, `institution_net_buy` 컬럼으로 저장.
  3. 신호화(`smart_money_buy_signal`) 및 롤링 누적 feature 추가.

### E. 52주 신고가 근처/돌파
- 외부 API 없이 OHLCV만으로 계산 가능.
- 현재 코드에 `close_to_52w_high`, `near_52w_high_flag`, `breakout_52w_flag` 반영.

---

## 2) 권장 데이터 파이프라인 구조

1. **수집 레이어**
   - `fetch_disclosures.py` (OpenDART)
   - `fetch_news_issues.py` (뉴스 원문/요약)
   - `fetch_investor_flow.py` (pykrx)

2. **정규화 레이어**
   - 컬럼 표준화: `Date`, `Symbol`, `disclosure_summary`, `news_summary`, `foreign_net_buy`, `institution_net_buy`
   - 타임존/장마감 기준 정렬(한국장 D+1 누수 방지)

3. **표시용 컨텍스트 병합 레이어**
   - `Date, Symbol` left join
   - 결측 안전 처리(0 또는 중립값)
   - 뉴스/공시는 `result_news.csv`, `result_disclosure.csv`, `result_simple.csv`, Kakao 응답에 표시
   - 예측 피처와 추천 정책에는 연결하지 않음

4. **학습/추론 레이어**
   - 매수/매도/관망은 다음날 예상 수익률(`predicted_return`) 기준으로만 결정
   - 뉴스/공시는 사용자가 결과를 해석할 때 참고할 수 있도록 별도 표시

---

## 3) 운영 시 주의사항 (중요)

- **데이터 라이선스/약관 준수**: 뉴스 크롤링은 출처별 정책 확인 필수.
- **예측 영향 차단**: 공시/뉴스는 표시용 컨텍스트이며 모델 피처, 예상 수익률, 매수/매도 신호에 반영하지 않음.
- **결측 처리 일관성**: 소스 장애 시 0 또는 중립값으로 fallback하고 coverage 로깅.
- **품질 모니터링**: 일자별 수집 성공률, 종목 커버리지, 평균 감성점수 변동 알림.

---

## 4) 최소 구현 로드맵 (2주)

### Week 1
- OpenDART + pykrx 수집 스크립트 작성
- 일자-종목 스키마 통일 CSV 생성

### Week 2
- 뉴스/공시 요약 표시 연결
- 리포트/챗봇 표시용 merge 자동화
- 리포트에 coverage 추가(성공/실패/결측률)

이 순서가 예측 산식과 표시용 이슈 컨텍스트를 분리하면서 운영 설명력을 높이는 데 적합합니다.

---

## 5) News Impact Scoring Package

`news_impact/` is now included as a standalone news/disclosure scoring module.
It is operationally separate from the prediction model and can be run through
the `stock-news-impact` console script.

Recommended source policy:

- Korean company names first.
- Korean industry/search keywords by default.
- Korean news first.
- Non-Korean or overseas media only when explicitly needed for a company,
  industry, supply chain, or global macro event.

Runtime templates:

- `configs/news_impact.example.json`: OpenAI-default LLM/runtime config template.
- `configs/news_impact.gemma.example.json`: optional local Gemma/llama.cpp config template.
- `data/news_impact/watchlist.example.csv`: ticker watchlist template.
- `data/news_impact/company_master.example.csv`: company metadata template.

Integration with `stock_predict`:

```powershell
stock-news-impact --help
python src/pipeline.py --input data/real_ohlcv.csv --news-impact-report result/news_impact_report.json
```

The prediction pipeline reads the report with `src/reports/news_impact_context.py`
and appends display columns such as `news_impact_final_score`,
`news_impact_top_reason`, `news_impact_risk_flags`, and
`news_impact_top_evidence_url`.

Do not feed these values into expected-return modeling, ranking, or
buy/sell/hold decisions. They are review context only.
