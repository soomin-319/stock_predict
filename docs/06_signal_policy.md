# 06. 시그널 정책 및 추천

예측 결과를 매수/매도/관망 추천으로 변환하는 정책 레이어.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `src/domain/signal_policy.py` | 핵심 추천 정책 및 이벤트 부스트 |
| `src/inference/predict.py` | 시그널 레이블 생성 |
| `src/recommendation/close_betting.py` | 종가 베팅 추천 메시지 포맷 |
| `src/recommendation/realtime_close_betting.py` | 실시간 종가 추천 서비스 |

---

## 추천 정책 (`signal_policy.py`)

### 핵심 규칙

```python
# src/domain/signal_policy.py:34
def recommendation_from_signal(
    signal_score, predicted_return, up_probability=None, uncertainty_score=None
) -> str:
    if predicted_return > 2.0:
        return "매수"
    if predicted_return <= -2.0:
        return "매도"
    return "관망"
```

**추천 결정 기준은 `predicted_return`(다음날 예상 수익률 %)만 사용한다.**

| 조건 | 추천 |
|------|------|
| `predicted_return > 2.0%` | 매수 |
| `predicted_return <= -2.0%` | 매도 |
| 그 외 | 관망 |

`signal_score`, `up_probability`, 뉴스, 공시 등은 **지원 진단 정보**이며 추천 결정에 영향을 주지 않는다.

---

## 시그널 점수 계산

```python
signal_score = (
    return_weight    × norm_return        # 예상 수익률 정규화
  + up_prob_weight   × up_probability     # 상승 확률
  + rel_strength_weight × rel_strength   # 상대 강도
  - uncertainty_penalty × uncertainty_score  # 불확실성 패널티
  + event_boost_score                    # 이벤트 부스트 (아래 참조)
)
```

시그널 점수는 종목 **순위** 결정에 사용된다 (백테스트 Top-K 선택).

### 신뢰도 점수 (`confidence_score`)

```python
# src/domain/signal_policy.py:56
def confidence_label(confidence_score: float) -> str:
    if c >= 0.80: return "신뢰도 높음"
    if c >= 0.60: return "신뢰도 보통"
    return "신뢰도 낮음"
```

---

## 이벤트 부스트 (`vectorized_event_signal_boost`)

```python
# src/domain/signal_policy.py
def vectorized_event_signal_boost(pred_df: pd.DataFrame) -> pd.DataFrame
```

특정 이벤트 조건에 따라 `event_boost_score`를 계산한다:

| 이벤트 | 부스트/패널티 | 조건 |
|--------|--------------|------|
| 거래대금 상위 3위 | +0.05 | `is_top_turnover_3 == 1` |
| 외국인+기관 동시 순매수 | +0.04 | 두 흐름 모두 양수 |
| 고확신 동시 순매수 (1,000억↑) | +0.08 | 외국인+기관 각각 1,000억 이상 |
| 강성 동시 순매수 | +0.06 | 두 흐름 합계 대규모 |
| 섹터 리더 확인 | +0.05 | 동행 상승 종목 N개 이상 |
| 52주 신고가 근접 | +0.03 | 종가 ≥ 52주 고가 × 0.97 |
| RSI 풀백 | +0.02 | RSI 30~35 구간 |
| NASDAQ 강한 테일윈드 | +0.06 | 나스닥 선물 +1% 이상 |
| NASDAQ 강한 헤드윈드 | -0.12 | 나스닥 선물 -1% 이하 |
| RSI 과매수 | -0.08 | RSI ≥ 70 |

---

## 시그널 레이블 (`inference/predict.py`)

```python
# src/inference/predict.py
def signal_label_series(signal_score: pd.Series) -> pd.Series
```

`signal_score`를 구간별 레이블로 변환 (순위·정렬용):

| 점수 범위 | 레이블 |
|-----------|--------|
| 상위 | `strong_buy` |
| 중상 | `buy` |
| 중립 | `neutral` |
| 중하 | `sell` |
| 하위 | `strong_sell` |

---

## 종가 베팅 추천 (`recommendation/`)

### `close_betting.py`

```python
# src/recommendation/close_betting.py
def format_recommendation_message(pred_df: pd.DataFrame, symbol: str) -> str
```

카카오 챗봇 응답용 추천 메시지를 생성한다:

```
[삼성전자 (005930)]
권고: 매수
예상 수익률: +2.34%
상승확률: 62.1%
신뢰도: 높음
```

### `realtime_close_betting.py`

```python
# src/recommendation/realtime_close_betting.py
class RealTimeCloseBettingRecommendationService:
    def get_recommendation(self, symbol: str) -> str
```

실시간으로 특정 종목의 최신 예측 결과를 조회하여 추천 메시지를 반환. 캐시된 `result_simple.csv`를 우선 사용한다.

---

## 포트폴리오 액션 및 리스크 플래그

최종 예측 프레임에 추가되는 필드:

| 필드 | 값 예시 | 설명 |
|------|---------|------|
| `recommendation` | `매수` / `매도` / `관망` | 추천 결정 |
| `signal_score` | 0.0 ~ 1.0 | 시그널 종합 점수 |
| `signal_label` | `buy` / `neutral` | 시그널 레이블 |
| `confidence_score` | 0.0 ~ 1.0 | 예측 신뢰도 |
| `portfolio_action` | `buy` / `hold` / `skip` | 포트폴리오 액션 |
| `risk_flag` | `low_liquidity` 등 | 리스크 플래그 |
| `coverage_gate_status` | `normal` / `caution` / `halt` | 커버리지 상태 |

---

## 중요 제약사항

> 뉴스, 공시, `news_impact_*` 컬럼은 **표시 전용**이며 추천 결정에 절대 사용하지 않는다.
>
> `predicted_return`만이 매수/매도/관망의 결정 기준이다.
>
> — `docs/ARCHITECTURE.md` 추천 정책 섹션

---

## 개선 및 수정 제안

> 우선순위: **P0(버그) > P1(정합성) > P2(문서/품질)**.

### P0 — 최신 예측 경로에서 이벤트 부스트가 `signal_score`에 이중 적용

- **문제**: 최신 예측 산출 시 `build_scored_prediction_frame`가 `vectorized_event_signal_boost`를 호출해 `signal_score += event_boost_score`를 1회 적용한다(`pipeline.py:707` → `pipeline_support.py:86`). 직후 `finalize_latest_prediction_frame` → `build_prediction_policy_frame`가 **같은 부스트를 다시 더한다**(`pipeline.py:719` → `signal_policy.py:285,173`). 결과적으로 최신 예측의 `signal_score`/`signal_label`에 이벤트 부스트가 **두 번** 들어간다. (OOF/백테스트 경로는 `apply_tuned_signal`이 `signal_score`를 처음부터 재계산하므로 1회만 적용되어 영향 없음.)
- **영향**: 추천(매수/매도/관망)은 `predicted_return`만 쓰므로 불변이지만, `result_detail/simple.csv`에 노출되는 `signal_score`·`signal_label`·순위가 왜곡된다.
- **제안**: `build_prediction_policy_frame`가 이미 부스트가 적용된 프레임을 받을 때는 재적용하지 않도록 가드(`event_boost_score` 존재 시 skip)를 두거나, 부스트 적용 책임을 한 함수로 단일화.

### P0 — 문서의 `confidence_label` 임계값이 코드와 불일치

- **문제**: 문서는 `≥0.80 높음 / ≥0.60 보통 / else 낮음`(3단계)로 적었지만, 코드는 `≥0.80 매우 높음 / ≥0.67 높음 / ≥0.34 보통 / else 낮음`(4단계)이다(`signal_policy.py:56-66`).
- **제안**: 문서를 4단계 실제 임계값으로 정정.

### P1 — 문서의 52주 근접 기준(×0.97)과 코드(0.95) 불일치

- **문제**: 이벤트 부스트 표는 "종가 ≥ 52주 고가 × 0.97"이라 적었으나, 플래그 생성은 `>= 0.95`로 하드코딩이며(`price_features.py:262`) 설정값 `near_52w_distance_threshold=0.03`은 미사용이다. (상세는 `03_features.md` 참고.)
- **제안**: 코드를 설정 기반(`1 - near_52w_distance_threshold`)으로 바꾸고 문서·설정·코드를 일치.

### P1 — 이벤트 부스트의 의도적/비의도적 가산 중복 명시

- **문제**: 거래대금 1~3위 종목은 `TOP3_TURNOVER_EVENT_BOOST(+0.05)`와 `TOP_TURNOVER_EVENT_BOOST(+0.04)`가 **동시 가산**되어 +0.09가 된다(`signal_policy.py:158-159`). 고확신 동시순매수도 `dual_buy + strong_dual_buy + combined`가 누적된다. 의도일 수 있으나 문서에는 단일 항목처럼 보인다.
- **제안**: 부스트가 누적 가산임을 표에 주석으로 명시하고, 상한(clip)·정규화 여부를 정의.

### P1 — `risk_flag`의 `LOW_LIQUIDITY` 기준이 사실상 비활성

- **문제**: `value_traded < min_liquidity_threshold`로 판정하는데(`signal_policy.py:80`), `min_liquidity_threshold` 기본은 컨텍스트에서 0으로 전달되는 경우가 많아 거의 트리거되지 않는다(`pipeline_support.py:42`).
- **제안**: 백테스트의 `min_value_traded`(기본 30억)와 동일 임계를 위험 플래그에도 연결.

### P2 — `signal_label` 경계 문서화

- **문제**: 문서 표는 `strong_buy/buy/neutral/sell/strong_sell`로 적었으나 실제 라벨은 `strong_negative/weak_negative/neutral/weak_positive/strong_positive`이고 경계는 `0.25/0.45/0.55/0.75`이다(`inference/predict.py:19-24`).
- **제안**: 실제 라벨·경계로 정정.

### P2 — 행 단위 `apply`의 성능

- **문제**: `build_prediction_policy_frame`가 `prediction_reason`·`_jongbae_score`·`build_pm_summary_fields`를 `df.apply(..., axis=1)`로 종목마다 호출(`signal_policy.py:286-290`). 종목이 많으면 느리다.
- **제안**: 이벤트 부스트처럼 벡터화하거나, 사유 텍스트는 상위 노출 종목으로 한정 생성.
