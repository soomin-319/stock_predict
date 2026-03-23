# 현재 코드 기준, 적용 완료된 개선과 다음 우선순위

## 이미 반영된 핵심 개선

### 1) 구조 분리
- 시그널 정책/리스크/PM 액션 로직을 `src/domain/signal_policy.py`로 분리했습니다.
- 사용자용 결과 포맷팅과 콘솔 출력 로직을 `src/reports/result_formatter.py`로 분리했습니다.

### 2) 시그널/신뢰도 의미 분리
- `signal_label`: 모델 점수 구간
- `confidence_label`: 신뢰도 라벨

### 3) 투자자 컨텍스트 세분화
- flow / disclosure / news를 각각 켜고 끌 수 있는 CLI 옵션을 추가했습니다.
- 뉴스 점수화는 `auto / rule / ai` 모드를 지원합니다.

### 4) 백테스트 현실화 1차
- 최소 거래대금 필터
- benchmark/excess return
- conservative / neutral / aggressive 비용 시나리오
- turnover cap 유지

### 5) PM 요약 체계
- `portfolio_action`
- `trading_gate`
- `risk_flag`
- `position_size_hint`
- `prediction_reason`

### 6) 설정 프로필
- `configs/research_balanced.json`
- `configs/prod_conservative.json`

---

## 남아 있는 다음 우선순위

### P1: 성능·운영 고도화
1. `price_features.py`의 컬럼 생성 방식을 `pd.concat` 중심으로 바꿔 fragmentation warning 제거
2. sector/industry 메타데이터를 붙여 포트폴리오 편중 cap 추가
3. 다중 호라이즌(5일/20일) 모델 헤드 확장

### P2: 실전 운용 확장
4. 드리프트 감지/경보
5. 실험 레지스트리와 모델 버전 추적
6. 챗봇 응답에 종목 근거 카드/리스크 카드 추가
