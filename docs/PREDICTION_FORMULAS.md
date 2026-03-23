# 주가 예측 공식 정리

이 문서는 현재 프로젝트에서 **주가 예측에 실제로 사용되는 공식**을 단계별로 정리한 문서다.  
중요한 점은 이 프로젝트의 최종 예측이 하나의 단일 수식으로 표현되는 구조가 아니라,

1. 피처 생성 공식
2. 학습 타깃 공식
3. 트리 기반 모델 학습
4. 확률 보정 / 시그널 점수 / 권고 생성

이 순서로 이어지는 **파이프라인 구조**라는 점이다.

---

## 1. 예측 파이프라인 개요

현재 파이프라인은 대략 아래 순서로 동작한다.

1. `build_features()`에서 가격/거래량/수급/뉴스/이벤트 기반 피처를 생성한다.
2. 필요하면 `add_external_market_features_with_coverage()`로 외부 시장 피처를 붙인다.
3. `annotate_market_regime()`로 시장 국면 메타정보를 만든다.
4. `MultiHeadStockModel`이 다음날 로그수익률, 상승확률, 분위수 구간을 학습한다.
5. `build_prediction_frame()`이 예측값을 종가/수익률/시그널 점수로 변환한다.
6. `pipeline.py`에서 확률 보정, 확률 override, 권고/신뢰도 계산을 적용한다.

---

## 2. 입력 피처 공식

### 2.1 가격/수익률 기본 피처

- `log_return = log(Close / Close.shift(1))`
- `daily_return = Close.pct_change()`
- `gap_return = (Open / prev_close) - 1`
- `intraday_return = (Close / Open) - 1`
- `range_pct = (High - Low) / Close`
- `value_traded = Close * Volume`
- `turnover_rank_daily = 일자별 value_traded 내림차순 rank`
- `is_top_turnover_10 = turnover_rank_daily <= 10`

### 2.2 기간별 수익률 / 이동평균 / 변동성

기본 설정은 아래 윈도우를 사용한다.

- 수익률 윈도우: `[1, 2, 3, 5, 10, 20, 60]`
- 이동평균 윈도우: `[5, 10, 20, 60, 120]`
- 변동성 윈도우: `[5, 20, 60]`

공식은 다음과 같다.

- `ret_wd = Close.pct_change(w)`
- `ma_w = rolling_mean(Close, w)`
- `close_to_ma_w = Close / ma_w - 1`
- `vol_w = rolling_std(log_return, w)`
- `vol_ratio_20 = Volume / rolling_mean(Volume, 20)`

### 2.3 기술적 지표

#### RSI

- `delta = close.diff()`
- `gain = max(delta, 0)`
- `loss = max(-delta, 0)`
- `avg_gain = EWM(gain, alpha=1/period)`
- `avg_loss = EWM(loss, alpha=1/period)`
- `RS = avg_gain / avg_loss`
- `RSI = 100 - 100 / (1 + RS)`

#### MACD

- `EMA12 = EWM(close, span=12)`
- `EMA26 = EWM(close, span=26)`
- `MACD = EMA12 - EMA26`
- `Signal = EWM(MACD, span=9)`
- `Hist = MACD - Signal`

#### ATR

- `TR = max(high-low, |high-prev_close|, |low-prev_close|)`
- `ATR = rolling_mean(TR, 14)`

#### Stochastic

- `LL = rolling_min(low, 14)`
- `HH = rolling_max(high, 14)`
- `%K = 100 * (Close - LL) / (HH - LL)`
- `%D = rolling_mean(%K, 3)`

#### CCI

- `TP = (High + Low + Close) / 3`
- `SMA = rolling_mean(TP, 20)`
- `MAD = rolling_mean(|TP - mean(TP)|, 20)`
- `CCI = (TP - SMA) / (0.015 * MAD)`

#### OBV

- `direction = sign(close.diff())`
- `OBV = cumulative_sum(direction * volume)`
- `obv_change_5d = OBV.pct_change(5)`

---

## 3. 수급 / 뉴스 / 이벤트 기반 공식

### 3.1 수급 신호

- `foreign_buy_signal = foreign_net_buy > 0`
- `institution_buy_signal = institution_net_buy > 0`
- `smart_money_buy_signal = (foreign_net_buy + institution_net_buy) > 0`

### 3.2 뉴스 신호

- `news_positive_signal = news_relevance_score * max(news_sentiment - 0.5, 0) * 2`
- `news_negative_signal = news_relevance_score * max(0.5 - news_sentiment, 0) * 2`

### 3.3 52주 고점 관련

- `rolling_high_252 = 252일 rolling max(Close)`
- `prev_rolling_high_252 = 1일 shift 후 252일 rolling max(Close)`
- `close_to_52w_high = Close / rolling_high_252`
- `near_52w_high_flag = close_to_52w_high >= 0.95`
- `breakout_52w_flag = Close >= prev_rolling_high_252`

### 3.4 investor_event_score

이 프로젝트에서 명시적으로 가중합을 사용하는 대표 공식이다.

```text
investor_event_score =
    0.35 * is_top_turnover_10
  + 0.20 * disclosure_score
  + 0.20 * news_positive_signal
  + 0.15 * smart_money_buy_signal
  + 0.10 * near_52w_high_flag
```

### 3.5 상한가/하한가/VI 관련

- `limit_hit_up_flag = daily_return >= 0.295`
- `limit_hit_down_flag = daily_return <= -0.295`
- `limit_event_flag = limit_hit_up_flag or limit_hit_down_flag`
- `vi_after_return = daily_return * vi_flag`
- `vi_after_volume_spike = vol_ratio_20 * vi_flag`

### 3.6 공매도 이벤트 점수

```text
short_sell_event_score =
    0.5 * short_sell_overheat_flag
  + 0.3 * short_sell_flag
  + 0.2 * (short_sell_ratio > 0)
```

### 3.7 주주환원 점수

```text
shareholder_return_score =
    0.4 * buyback_flag
  + 0.3 * share_cancellation_flag
  + 0.3 * value_up_disclosure_flag
```

---

## 4. 공시 / 뉴스 원천 점수 공식

### 4.1 disclosure_score

각 공시 제목에 대해:

- 기본 점수 `0.2`
- 긍정 키워드(예: 수주, 계약, 증가, 실적, 합병 등)가 포함되면 `+0.3`
- 날짜별 합산 후 `0~1` 범위로 clip

### 4.2 news_sentiment

룰 기반 점수는 다음과 같다.

```text
score =
    0.5
  + 0.12 * (positive_hits - negative_hits)
  + 0.18 * (strong_positive_hits - strong_negative_hits)
  - 0.05 * uncertainty_hits
```

최종값은 `0~1` 범위로 clip한다.

### 4.3 news_relevance_score

```text
score =
    0.25
  + 0.25 * min(price_impact_hits, 2)
  - 0.15 * low_signal_hits
  - 0.05 * uncertainty_hits
```

최종값은 `0~1` 범위로 clip한다.

### 4.4 news_impact_score

룰 기반일 때:

```text
weighted_sentiment = 0.5 + (sentiment - 0.5) * relevance
impact = (weighted_sentiment - 0.5) * 2.0
```

### 4.5 뉴스 집계 방식

같은 날짜의 뉴스는 아래 방식으로 집계한다.

- `news_sentiment`: 평균
- `news_relevance_score`: 평균
- `news_impact_score`: 평균
- `news_article_count`: 합계

---

## 5. 외부 시장 피처 공식

외부 심볼(예: 코스피, 코스닥, S&P500, 나스닥, 나스닥 선물, SOX, VIX, 환율, 미국금리 등)에 대해:

- `*_ret_1d = close.pct_change()`
- `*_ret_5d = close.pct_change(5)`
- `*_vol_20 = rolling_std(ret_1d, 20)`

이 피처는 `use_external=True`일 때만 예측에 포함된다.

---

## 6. 시장 국면 공식

시장 국면은 다음처럼 만든다.

- `trend = close_to_ma_20`
- `trend > 0.01` → `uptrend`
- `trend < -0.01` → `downtrend`
- 나머지 → `sideways`
- `vol_20 > vol_20의 75% 분위수` → `high_vol`
- 나머지 → `low_vol`

최종적으로 `market_regime = trend_state + "_" + vol_state`

예: `uptrend_high_vol`, `sideways_low_vol`

---

## 7. 모델이 직접 학습하는 타깃 공식

모델이 예측하는 핵심 타깃은 다음과 같다.

- `target_log_return = log(next_close / close)`
- `target_up = target_log_return > 0`
- `target_close = close * exp(target_log_return)`

즉, 프로젝트의 핵심 예측 대상은 **다음날 로그수익률**이며,  
다음날 종가 예측은 이 로그수익률을 다시 가격으로 환산한 값이다.

---

## 8. 모델 구조 자체

프로젝트는 `MultiHeadStockModel`을 사용한다.

- 회귀 헤드: `predicted_log_return`
- 분류 헤드: `up_probability`
- 분위수 회귀 헤드: `quantile_low`, `quantile_mid`, `quantile_high`

중요한 점은, 이 부분은 선형 회귀처럼 사람이 읽을 수 있는 **하나의 고정 수식**이 아니라  
**LightGBM / GradientBoosting 트리 앙상블이 학습한 함수**라는 점이다.

즉,

- “피처 A 10% + 피처 B 20%”처럼 단일 선형 계수표가 있는 구조가 아니고
- 피처별 영향도는 데이터에 따라 비선형으로 달라진다.

---

## 9. 예측 후처리 공식

### 9.1 예측 로그수익률 → 수익률 / 종가 변환

- `predicted_return = expm1(predicted_log_return) * 100`
- `predicted_close = Close * exp(predicted_log_return)`

### 9.2 불확실성 계산

- `uncertainty_width = quantile_high - quantile_low`
- `uncertainty_score = uncertainty_width의 percentile score`

### 9.3 정규화된 상대 강도

- `norm_return = zscore(predicted_log_return)`
- `rel_strength = zscore(predicted_log_return)`

---

## 10. 시그널 점수 공식

최종 랭킹에 쓰는 `signal_score`는 다음 공식이다.

```text
signal_score =
    return_weight * norm_return
  + up_prob_weight * up_probability
  + rel_strength_weight * rel_strength
  - uncertainty_penalty * uncertainty_score
```

기본값은:

- `return_weight = 0.45`
- `up_prob_weight = 0.35`
- `rel_strength_weight = 0.20`
- `uncertainty_penalty = 0.25`

그리고 `signal_label`은 `signal_score`를 구간으로 나눠:

- `strong_negative`
- `weak_negative`
- `neutral`
- `weak_positive`
- `strong_positive`

로 분류한다.

---

## 11. 시그널 가중치 튜닝 공식

기본 가중치는 고정이 아니라 OOF 예측 결과를 사용해 다시 튜닝된다.

탐색 방식은:

- `rw ∈ {0.3, 0.45, 0.6}`
- `uw ∈ {0.15, 0.25, 0.35}`
- `w_prob = 0.30`
- `w_rel = 1.0 - rw - w_prob`

각 조합에 대해:

```text
score =
    rw * norm_return
  + 0.30 * up_probability
  + w_rel * rel_strength
  - uw * uncertainty_score
```

그리고 상위 10% 종목의 평균 `target_log_return`이 최대인 조합을 선택한다.

즉, 실제 운영 시 `signal_score`의 비중은 **항상 고정값이 아니다.**

---

## 12. 확률 보정 공식

OOF 데이터가 충분할 경우 `IsotonicRegression`을 써서 `up_probability`를 재보정한다.

- 입력: 원래 모델의 `up_probability`
- 정답: `target_log_return > 0`
- 출력: 보정된 상승확률

이 보정은 OOF 분석과 최신 예측 모두에 적용된다.

---

## 13. 확률 override 공식

최종 예측 후 일부 조건에서 `up_probability`를 강제로 올리는 규칙이 있다.

- 거래대금 상위 종목이면 최소 `0.65`
- 외국인/기관 각각 1000억 이상 순매수면 최소 `0.70`
- 둘 다 만족하면 최소 `0.78`
- 나스닥 선물 상승이면 최소 `0.55`

이 부분은 모델이 학습한 확률을 **규칙 기반으로 덮어쓰는 후처리**다.

---

## 14. 권고 / 신뢰도 / 포지션 힌트 공식

### 14.1 권고(매수/매도/관망)

권고는 아래 기준으로 결정된다.

- `predicted_return > 1.0` → `매수`
- `predicted_return <= -1.0` → `매도`
- 그 외 → `관망`

즉, 현재 코드에서는 `signal_score`보다 **예상 수익률 임계값**이 권고 결정에 직접 쓰인다.

### 14.2 confidence_score

- 기본 `confidence_score = 1 - uncertainty_score`

### 14.3 최종 표시용 예측 신뢰도

```text
display_confidence =
    0.5 * confidence_score
  + 0.5 * history_direction_accuracy
```

### 14.4 risk_flag

아래 조건을 조합한다.

- `uncertainty_score >= 0.75`
- `up_probability < 0.5`
- `history_direction_accuracy < 0.45`

---

## 15. 백테스트에서 사용하는 공식

백테스트는 `signal_score` 상위 종목을 뽑아서 성과를 본다.

### 종목 선택 조건

- `up_probability >= min_up_probability`
- `signal_score >= min_signal_score`
- 그중 상위 `top_k`

### 일별 수익률

- `gross = 선택 종목 target_log_return 평균`
- `dyn_penalty = mean(uncertainty_score) + mean(max(vol_ratio_20 - 1, 0))`
- `cost = (fee_bps + slippage_bps + dynamic_slippage_bps * dyn_penalty) / 10000`
- `net = gross - cost`

즉, 랭킹엔 `signal_score`, 실제 백테스트 손익엔 `target_log_return`과 비용 패널티가 함께 쓰인다.

---

## 16. 공식들의 비중이 모두 같은가?

## 결론: **아니다.**

### 16.1 명시적 가중치가 서로 다르다

이미 코드에 박혀 있는 공식만 봐도:

- `investor_event_score`: `0.35 / 0.20 / 0.20 / 0.15 / 0.10`
- `short_sell_event_score`: `0.5 / 0.3 / 0.2`
- `shareholder_return_score`: `0.4 / 0.3 / 0.3`
- 기본 `signal_score`: `0.45 / 0.35 / 0.20 / -0.25`

처럼 **가중치가 모두 다르다.**

### 16.2 시그널 가중치는 튜닝으로 바뀐다

OOF 결과 기준으로 최적 가중치를 다시 찾기 때문에, 실행 데이터에 따라 `signal_score`의 각 비중도 바뀐다.

### 16.3 모델 내부는 동일 가중치 구조가 아니다

트리 기반 모델은:

- 어떤 피처는 자주 분기되고
- 어떤 피처는 거의 안 쓰이며
- 같은 피처도 값 구간에 따라 영향이 달라진다.

즉, 실제 예측 영향도는 **비선형 + 비균등**이다.

### 16.4 공식마다 역할도 다르다

- 어떤 공식은 피처 생성용
- 어떤 공식은 모델 학습 타깃용
- 어떤 공식은 랭킹용
- 어떤 공식은 설명/신뢰도/권고 생성용

이므로 “모든 공식이 동일 비중으로 예측에 들어간다”는 식으로 이해하면 맞지 않는다.

---

## 17. 한 줄 결론

이 프로젝트의 예측은 하나의 수식이 아니라,  
**다수의 피처 공식 + 트리 모델 + 확률 보정 + 시그널 가중합 + 규칙 기반 후처리**가 결합된 구조다.

따라서:

- 공식은 여러 개이며
- 역할도 서로 다르고
- 비중도 동일하지 않다.

