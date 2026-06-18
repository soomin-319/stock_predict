# 06. 시그널 정책 및 추천

예측 결과를 매수/매도/관망 추천으로 변환하는 정책 레이어. 모든 출력은 리서치·운영 보조용이며
투자 자문이나 자동매매 시스템이 아니다.

> 핵심 원칙: 매수/매도/관망 결정은 익일 예상 수익률 `predicted_return`만 사용한다. 뉴스, 공시,
> `news_impact_*`, 이벤트 부스트는 표시·진단·정렬 보조 정보이며 추천 결정을 바꾸지 않는다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `src/domain/signal_policy.py` | 추천 정책, 이벤트 부스트, 리스크/PM 요약 필드 |
| `src/inference/predict.py` | 예측 프레임과 시그널 레이블 생성 |
| `src/pipeline_support.py` | 최신 예측 프레임 조립 및 정책 프레임 마무리 |
| `src/recommendation/close_betting.py` | 종가 베팅 추천 메시지 포맷 |
| `src/recommendation/realtime_close_betting.py` | 실시간 종가 추천 서비스 |

---

## 추천 정책 (`signal_policy.py`)

```python
def recommendation_from_signal(signal_score, predicted_return, up_probability=None, uncertainty_score=None) -> str:
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
벡터화 경로(`build_prediction_policy_frame`)에서도 동일하게 `predicted_return > 2.0`/`<= -2.0` 기준만 적용한다.

---

## 시그널 점수 계산

```
signal_score = return_weight × norm_return
             + up_prob_weight × up_probability
             - uncertainty_penalty × uncertainty_score
             + event_boost_score
```

시그널 점수는 종목 **순위**, Top-K 선택, 진단 표시용이다. 추천 라벨의 근거가 아니다.
(과거 `rel_strength` 항은 `norm_return`과 중복이라 제거되고 그 비중은 `return_weight=0.65`에 흡수됨.)

### 신뢰도 라벨 (`confidence_label`)

| 조건 | 라벨 |
|------|------|
| `confidence_score >= 0.80` | 신뢰도 매우 높음 |
| `confidence_score >= 0.67` | 신뢰도 높음 |
| `confidence_score >= 0.34` | 신뢰도 보통 |
| 그 외 | 신뢰도 낮음 |
| 결측 | 신뢰도 보통 |

---

## 이벤트 부스트 (`vectorized_event_signal_boost`)

시장/수급/기술 조건에 따라 `event_boost_score`를 **누적 가산**한다(상수는 `signal_policy.py` 상단 정의).

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
| NASDAQ 강한 테일윈드 | `+0.06` | `nq_f_ret_1d >= nasdaq_tailwind_threshold`(기본 0.01) |
| NASDAQ 강한 헤드윈드 | `-0.12` | `nq_f_ret_1d <= nasdaq_headwind_threshold`(기본 -0.01) |
| RSI 과매수 | `-0.08` | RSI `rsi_overbought`(기본 70) 이상 |

### 중복 적용 방지

`vectorized_event_signal_boost`는 `event_boost_score`가 이미 있는 프레임을 다시 받을 수 있다. 이 경우
기존 부스트를 `signal_score`에서 빼고 새 부스트를 더해, 최신 예측 경로에서 이벤트 부스트가 두 번 들어가지 않게 한다.

---

## 시그널 레이블 (`inference/predict.py`)

`signal_score`를 고정 경계로 구간화한다(`signal_label_series`).

| 점수 범위 | 라벨 |
|-----------|------|
| `(-inf, 0.25]` | `strong_negative` |
| `(0.25, 0.45]` | `weak_negative` |
| `(0.45, 0.55]` | `neutral` |
| `(0.55, 0.75]` | `weak_positive` |
| `(0.75, inf)` | `strong_positive` |

---

## 포트폴리오 액션 및 리스크 플래그

`build_prediction_policy_frame`이 최종 예측 프레임에 추가하는 주요 필드:

| 필드 | 설명 |
|------|------|
| `recommendation` | `predicted_return` 기반 매수/매도/관망 |
| `signal_score` / `signal_label` | 종합 시그널 점수와 구간 라벨(이벤트 누적으로 1 초과 가능) |
| `confidence_score` / `confidence_label` | 예측 신뢰도와 라벨 |
| `portfolio_action` | PM 표시용 액션(신규매수/관망 등) |
| `risk_flag` | 리스크 플래그(`LOW_LIQUIDITY`, `DATA_COVERAGE_LOW` 등) |
| `coverage_gate_status` | `normal` / `caution` / `halt` |

### `LOW_LIQUIDITY`

`value_traded < min_liquidity_threshold`이면 `LOW_LIQUIDITY`를 붙인다. `min_liquidity_threshold`가
결측(`NaN`)이거나 0 이하이면 미설정으로 보고 `DEFAULT_MIN_LIQUIDITY_THRESHOLD`(백테스트 기본 30억)를 사용한다.

### 벡터화된 정책 프레임

행 단위 `apply` 대신 벡터화 헬퍼(`_pm_summary_frame`, `_jongbae_score_series`, `_prediction_reason_series`)를
사용한다. 기존 단일 행 함수(`build_pm_summary_fields`, `_jongbae_score`, `prediction_reason`)는 호환성/테스트
비교용으로 유지된다.

---

## 종가 베팅 추천 (`recommendation/`)

- `close_betting.format_recommendation_message(pred_df, symbol)`: 카카오 응답용 추천 메시지를 생성한다.
- `realtime_close_betting.RealTimeCloseBettingRecommendationService.get_recommendation(symbol)`:
  특정 종목의 최신 예측을 조회해 추천 메시지를 반환한다(캐시된 `result_simple.csv` 우선).

외부 뉴스/공시 맥락은 메시지 표시용이며 추천 신호를 바꾸지 않는다.

---

## 중요 제약사항

- 뉴스, 공시, `news_impact_*` 컬럼은 표시·리뷰 전용이다.
- 뉴스/공시는 `predicted_return`, 추천, 순위, 신호를 변경하지 않는다.
- 매수/매도/관망 추천은 오직 익일 예상 수익률 `predicted_return` 기준이다.
