# 06. 시그널 정책 및 추천

예측 결과를 매수/매도/관망 추천으로 변환하는 정책 레이어. 이 프로젝트의 모든 출력은 리서치/운영 지원용이며 투자 조언이나 자동매매 시스템이 아니다.

> 핵심 원칙: 매수/매도/관망 결정은 다음날 예상 수익률인 `predicted_return`만 사용한다. 뉴스, 공시, `news_impact_*`, 이벤트 부스트는 표시·진단·정렬 보조 정보이며 추천 결정을 바꾸면 안 된다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `src/domain/signal_policy.py` | 핵심 추천 정책, 이벤트 부스트, 리스크/PM 요약 필드 |
| `src/inference/predict.py` | 예측 프레임과 시그널 레이블 생성 |
| `src/pipeline_support.py` | 최신 예측 프레임 조립 및 정책 프레임 마무리 |
| `src/recommendation/close_betting.py` | 종가 베팅 추천 메시지 포맷 |
| `src/recommendation/realtime_close_betting.py` | 실시간 종가 추천 서비스 |

---

## 추천 정책 (`signal_policy.py`)

### 핵심 규칙

```python
def recommendation_from_signal(
    signal_score, predicted_return, up_probability=None, uncertainty_score=None
) -> str:
    if predicted_return > 2.0:
        return "매수"
    if predicted_return <= -2.0:
        return "매도"
    return "관망"
```

| 조건 | 추천 |
|------|------|
| `predicted_return > 2.0%` | 매수 |
| `predicted_return <= -2.0%` | 매도 |
| 그 외 또는 결측 | 관망 |

`signal_score`, `up_probability`, `uncertainty_score`, 뉴스, 공시, 이벤트 부스트는 추천 결정에 영향을 주지 않는다.

---

## 시그널 점수 계산

```python
signal_score = (
    return_weight × norm_return
  + up_prob_weight × up_probability
  - uncertainty_penalty × uncertainty_score
  + event_boost_score
)
```

시그널 점수는 종목 **순위**, Top-K 선택, 진단 표시용이다. 추천 라벨(매수/매도/관망)의 근거가 아니다.

> 참고: 과거 `rel_strength` 항은 `norm_return`과 동일한 예측 수익률 백분위라 중복이어서 제거했다. 그 비중(0.20)은 `return_weight`(기본 0.65)에 흡수해 순위는 동일하게 유지된다.

### 신뢰도 점수 (`confidence_score`)

실제 코드의 `confidence_label`은 4단계다.

| 조건 | 라벨 |
|------|------|
| `confidence_score >= 0.80` | `신뢰도 매우 높음` |
| `confidence_score >= 0.67` | `신뢰도 높음` |
| `confidence_score >= 0.34` | `신뢰도 보통` |
| 그 외 | `신뢰도 낮음` |
| 결측 | `신뢰도 보통` |

---

## 이벤트 부스트 (`vectorized_event_signal_boost`)

특정 시장/수급/기술 조건에 따라 `event_boost_score`를 계산한다. 부스트와 패널티는 **누적 가산**이다.

| 이벤트 | 부스트/패널티 | 조건 |
|--------|--------------|------|
| 거래대금 상위 3위 | `+0.05` | `turnover_rank_daily <= 3` |
| 거래대금 상위권 | `+0.04` | `turnover_rank_daily <= top_turnover_rank`(기본 15) |
| 외국인+기관 동시 순매수 | `+0.04` | 두 흐름 모두 양수 |
| 강한 동시 순매수 | `+0.06` | 외국인·기관 각각 `high_conviction_net_buy_krw` 이상 |
| 상위 거래대금+강한 동시 순매수 결합 | `+0.08` | 상위권 거래대금이며 강한 동시 순매수 |
| 섹터 리더 확인 | `+0.05` | `leader_confirmation_flag > 0` |
| 52주 신고가 근접/돌파 | `+0.03` | `near_52w_high_flag > 0` 또는 `breakout_52w_flag > 0` |
| RSI 풀백 | `+0.02` | RSI 30~35 기본 구간 |
| NASDAQ 양수 테일윈드 | `+0.03` | `nq_f_ret_1d > 0` |
| NASDAQ 강한 테일윈드 | `+0.06` | `nq_f_ret_1d >= 0.01` 기본 |
| NASDAQ 강한 헤드윈드 | `-0.12` | `nq_f_ret_1d <= -0.01` 기본 |
| RSI 과매수 | `-0.08` | RSI 70 이상 기본 |

### 누적 규칙 예시

- 거래대금 1~3위는 `상위 3위(+0.05)`와 `상위권(+0.04)`가 함께 적용되어 거래대금 조건만으로 `+0.09`가 된다.
- 강한 동시 순매수는 `동시 순매수(+0.04)`, `강한 동시 순매수(+0.06)`, 조건 충족 시 `상위 거래대금+강한 동시 순매수(+0.08)`가 함께 쌓일 수 있다.
- NASDAQ 수익률이 `+1%` 이상이면 양수 테일윈드 `+0.03`과 강한 테일윈드 `+0.06`이 함께 적용된다.

### 중복 적용 방지

`vectorized_event_signal_boost`는 `event_boost_score`가 이미 있는 프레임을 다시 받을 수 있다. 이 경우 기존 부스트를 `signal_score`에서 빼고 새 부스트를 더해, 최신 예측 경로에서 이벤트 부스트가 두 번 들어가지 않도록 한다.

---

## 52주 신고가 근접 기준

`near_52w_high_flag`는 `src/features/investment_signals.py`에서 설정 기반으로 생성한다.

```python
distance_to_52w_high = 1.0 - close_to_52w_high
near_52w_high_flag = distance_to_52w_high <= near_52w_distance_threshold
```

기본 `near_52w_distance_threshold=0.03`이므로 `close_to_52w_high >= 0.97`과 같다. 임계값은 `InvestmentCriteriaConfig`로 조정한다.

---

## 시그널 레이블 (`inference/predict.py`)

`signal_score`를 고정 경계로 구간화한다.

| 점수 범위 | 라벨 |
|-----------|------|
| `(-inf, 0.25]` | `strong_negative` |
| `(0.25, 0.45]` | `weak_negative` |
| `(0.45, 0.55]` | `neutral` |
| `(0.55, 0.75]` | `weak_positive` |
| `(0.75, inf)` | `strong_positive` |

---

## 종가 베팅 추천 (`recommendation/`)

### `close_betting.py`

```python
def format_recommendation_message(pred_df: pd.DataFrame, symbol: str) -> str
```

카카오 챗봇 응답용 추천 메시지를 생성한다.

```text
[삼성전자 (005930)]
권고: 매수
예상 수익률: +2.34%
상승확률: 62.1%
신뢰도 높음
```

### `realtime_close_betting.py`

```python
class RealTimeCloseBettingRecommendationService:
    def get_recommendation(self, symbol: str) -> str
```

특정 종목의 최신 예측 결과를 조회해 추천 메시지를 반환한다. 캐시된 `result_simple.csv`를 우선 사용한다.

---

## 포트폴리오 액션 및 리스크 플래그

최종 예측 프레임에 추가되는 주요 필드:

| 필드 | 값 예시 | 설명 |
|------|---------|------|
| `recommendation` | `매수` / `매도` / `관망` | `predicted_return` 기반 추천 |
| `signal_score` | `0.0 ~ 1.0+` | 종합 시그널 점수. 이벤트 누적으로 1을 넘을 수 있음 |
| `signal_label` | `weak_positive` / `neutral` 등 | 시그널 점수 구간 라벨 |
| `confidence_score` | `0.0 ~ 1.0` | 예측 신뢰도 |
| `confidence_label` | `신뢰도 높음` 등 | 신뢰도 라벨 |
| `portfolio_action` | `신규매수` / `관망` 등 | PM 표시용 액션 |
| `risk_flag` | `LOW_LIQUIDITY` 등 | 리스크 플래그 |
| `coverage_gate_status` | `normal` / `caution` / `halt` | 커버리지 상태 |

### `LOW_LIQUIDITY`

`risk_flag`는 `value_traded < min_liquidity_threshold`이면 `LOW_LIQUIDITY`를 붙인다. `min_liquidity_threshold`가 없거나 0 이하이면 백테스트 기본값 `BacktestConfig().min_value_traded`(기본 30억)를 사용한다.

### 벡터화된 정책 프레임

`build_prediction_policy_frame`은 행 단위 `apply` 대신 벡터화 헬퍼를 사용한다.

- `_pm_summary_frame`: 추천, 리스크 플래그, 포지션 크기, PM 액션, 거래 게이트, 신뢰도 라벨 생성
- `_jongbae_score_series`: 종배 점수 일괄 계산
- `_prediction_reason_series`: 표시용 예측 이유 일괄 생성

기존 단일 행 함수(`build_pm_summary_fields`, `_jongbae_score`, `prediction_reason`)는 호환성과 테스트 비교용으로 유지한다.

---

## 중요 제약사항

- 뉴스, 공시, `news_impact_*` 컬럼은 **표시/리뷰 전용**이다.
- 뉴스/공시 정보는 `predicted_return`, 추천, 순위, 신호를 변경하면 안 된다.
- 매수/매도/관망 추천은 오직 다음날 예상 수익률 `predicted_return` 기준이다.

---

## 개선 및 수정 진행 현황

> 우선순위: **P0(버그) > P1(정합성) > P2(문서/품질)**.

### 완료 — P2 행 단위 `apply` 성능

- `build_prediction_policy_frame`의 `build_pm_summary_fields`, `_jongbae_score`, `prediction_reason` 경로를 벡터화했다.
- 추천 정책은 변경하지 않았다. 매수/매도/관망은 계속 `predicted_return`만 사용한다.
- `min_liquidity_threshold`가 결측(`NaN`)인 경우도 미설정으로 보고 백테스트 기본 유동성 기준을 사용한다.
