# 외부 데이터 소스 연동 가이드 (거래대금/공시/뉴스/수급)

요청하신 아래 4개 항목을 예측 feature에 안정적으로 넣기 위한 실무 설계입니다.

- 거래대금 상위 10개 공시
- 호재/공시/뉴스
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
  3. 룰 기반 점수화(`disclosure_score`) 후 일자-종목 키로 병합.

### C. 호재/뉴스
- 소스 옵션:
  - 네이버/다음 증권 뉴스 API/크롤링(서비스 약관 준수 필요)
  - 상용 뉴스 API(예: Finnhub, RavenPack, AlphaSense 등)
- 연동 방식:
  1. 헤드라인 수집 → 종목 매핑(NER/룰 기반).
  2. 감성분석(BERT/KoELECTRA/룰 기반)으로 `news_sentiment` 산출.
  3. 중복 기사 제거 후 종목-일자 점수로 병합.

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
   - `fetch_news_sentiment.py` (뉴스/감성)
   - `fetch_investor_flow.py` (pykrx)

2. **정규화 레이어**
   - 컬럼 표준화: `Date`, `Symbol`, `disclosure_score`, `news_sentiment`, `foreign_net_buy`, `institution_net_buy`
   - 타임존/장마감 기준 정렬(한국장 D+1 누수 방지)

3. **피처 병합 레이어**
   - `Date, Symbol` left join
   - 결측 안전 처리(0 또는 중립값)
   - 누수 방지: 당일 장중/장후 데이터 cutoff 정책 분리

4. **학습/추론 레이어**
   - 본 프로젝트 `build_features`가 추가 컬럼 alias를 읽어 자동 반영

---

## 3) 운영 시 주의사항 (중요)

- **데이터 라이선스/약관 준수**: 뉴스 크롤링은 출처별 정책 확인 필수.
- **타이밍 누수 방지**: 공시/뉴스 게시시각이 장마감 이후면 다음 거래일 feature로 이월.
- **결측 처리 일관성**: 소스 장애 시 0 또는 중립값으로 fallback하고 coverage 로깅.
- **품질 모니터링**: 일자별 수집 성공률, 종목 커버리지, 평균 감성점수 변동 알림.

---

## 4) 최소 구현 로드맵 (2주)

### Week 1
- OpenDART + pykrx 수집 스크립트 작성
- 일자-종목 스키마 통일 CSV 생성

### Week 2
- 뉴스 감성 점수 연결
- 파이프라인 입력 merge 자동화
- 리포트에 coverage 추가(성공/실패/결측률)

이 순서가 비용 대비 성능/설명력 개선 효과가 가장 큽니다.
