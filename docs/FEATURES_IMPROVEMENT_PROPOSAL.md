# FEATURES_GUIDE.md 기반 피처 개선 및 수정 제안서

## 0. 검토 관점

본 문서는 `FEATURES_GUIDE.md`의 현재 피처 설계와 처리 흐름을 바탕으로, **주식 투자 전문가의 관점**에서 예측력, 실전 매매 가능성, 리스크 통제, 데이터 누수 방지, 백테스트 신뢰도를 높이기 위한 개선 및 수정사항을 제안한다.

핵심 판단 기준은 다음과 같다.

1. **시점 정합성**: 투자 판단 시점에 실제로 관측 가능한 데이터만 사용해야 한다.
2. **알파와 리스크의 분리**: 단순 기대수익률뿐 아니라 변동성, 유동성, 거래비용, 시장 노출을 함께 평가해야 한다.
3. **상대성과 횡단면 비교**: 개별 종목의 절대 수익률보다 시장·섹터·동일 유니버스 대비 초과성과가 더 실전적인 투자 신호가 된다.
4. **실행 가능성**: 예측값이 실제 주문, 체결, 포지션 사이징, 손절·익절, 리밸런싱 규칙으로 연결되어야 한다.
5. **검증 가능성**: 새 피처는 백테스트, 워크포워드 검증, 피처 중요도 안정성, 누수 테스트를 통과해야 한다.

---

## 1. 우선순위 요약

| 우선순위 | 개선 영역 | 현재 상태 | 권장 수정 | 기대 효과 |
|---|---|---|---|---|
| P0 | 외부시장 시차 및 결측 처리 | 외부 데이터를 날짜 기준 left join하고 `ffill()` 후 `bfill()` 처리 | `bfill()` 제거, 한국 장 기준 관측 가능 시점으로 1거래일 지연 또는 `available_at` 관리 | 미래값 유입 방지, 실전 신호 신뢰도 개선 |
| P0 | 목표값 정의 | `target_log_return = log(Close_{t+1}/Close_t)` 중심 | 거래비용 차감 수익률, 벤치마크 초과수익률, 진입가 기준 목표값 추가 | 모델 예측값과 실제 매매 성과 간 괴리 축소 |
| P0 | 피처 선택 규칙 | 접두사 기반으로 `ma_`, 외부 `*_close` 등 원시 레벨값이 선택될 수 있음 | 원시 가격·지수 레벨 제외, 정규화·비율·수익률·z-score 피처 중심 allowlist 적용 | 종목 가격대·지수 레벨에 따른 왜곡 완화 |
| P0 | 당일 OHLCV 사용 정책 | 종가 기준 피처와 다음 거래일 수익률 예측 구조 | 신호 생성 시점을 `장마감 후`, `익일 시가 전`, `장중`으로 명시 | 운영 시점별 누수 및 실행 오류 방지 |
| P1 | 유동성·거래비용 피처 | 거래대금 순위와 거래량 비율 중심 | 예상 슬리피지, 회전율, 거래대금 안정성, 체결 가능 금액 피처 추가 | 소형주·급등주 과최적화 방지 |
| P1 | 시장·섹터 상대 피처 | 시장 지수 수익률은 있으나 종목 상대성과는 제한적 | 시장 대비 초과수익률, 베타, 잔차 변동성, 섹터 상대 모멘텀 추가 | 알파와 단순 시장 노출 분리 |
| P1 | 수급 피처 스케일링 | 외국인·기관 순매수 금액과 고정 임계값 중심 | 거래대금·시가총액·유동주식수 대비 비율과 분위수 기준으로 변경 | 대형주 편향 완화 |
| P1 | 시장 국면 라벨 | 문자열 라벨이며 모델 입력 제외 | 수치형 regime score, 추세·변동성·위험회피 환경 피처 추가 | 국면별 모델 적응력 개선 |
| P2 | 기술적 지표 확장 | RSI, MACD, ATR, Stochastic, CCI, OBV 중심 | ADX, Bollinger Band, Donchian breakout, realized skew, reversal 피처 추가 | 추세·역추세·변동성 압축/확장 포착 |
| P2 | 뉴스·공시 활용 | 표시 전용으로 제외 | 기본 정책은 유지하되, 별도 실험 플래그와 시점 통제된 이벤트 피처 연구 | 정성 이벤트의 통제된 알파 검증 |

---

## 2. 핵심 수정 제안

## 2.1 외부시장 피처의 시점 정합성 강화

### 문제

현재 외부 시장 피처는 날짜 기준으로 병합되며, 외부 프레임 내부 결측값은 `ffill()` 후 `bfill()`된다. 또한 미국 시장, 선물, 환율, 금리 등은 한국 주식시장과 거래 시간대가 다르다. 이 구조에서는 다음 문제가 발생할 수 있다.

- `bfill()`은 과거 날짜의 결측값을 미래 데이터로 채울 수 있어 명백한 누수 위험이 있다.
- 같은 `Date`의 미국 시장 종가는 한국 시장 장마감 시점에 이미 관측 가능한 값이 아닐 수 있다.
- 야간선물, 환율, 금리 데이터는 관측 가능 시점이 서로 다르므로 단순 날짜 병합만으로는 실전 신호와 다를 수 있다.

### 수정 권장

1. 외부시장 데이터 보정에서 `bfill()`을 제거한다.
2. 한국 주식 `Date=t`의 장마감 후 신호라면, 미국 현물 지수의 같은 날짜 종가는 원칙적으로 사용할 수 없도록 한다.
3. 외부 피처는 기본적으로 `lag1` 버전을 모델 입력으로 사용한다.
4. 외부 데이터별로 `source_market`, `timezone`, `available_at`, `is_stale` 컬럼 또는 메타데이터를 관리한다.
5. 원본 외부 피처와 모델 입력용 외부 피처를 분리한다.

### 권장 피처명

| 기존 또는 신규 피처 | 권장 처리 |
|---|---|
| `gspc_ret_1d` | 원본 보관용. 모델 입력 제외 |
| `gspc_ret_1d_lag1_kr` | 한국 시장 기준 관측 가능하도록 1거래일 지연. 모델 입력 허용 |
| `ixic_ret_1d_lag1_kr` | 동일 |
| `sox_ret_1d_lag1_kr` | 반도체 민감 종목용 핵심 외부 피처 |
| `vix_ret_1d_lag1_kr` | 위험회피 국면 피처 |
| `krw_x_ret_1d_lag1_kr` | 원/달러 환율 변화 피처 |
| `external_stale_flag` | 휴장·결측·대체 심볼 사용으로 최신성이 떨어지는 경우 1 |
| `external_fallback_used_flag` | 원 심볼 대신 ETF 또는 대체 후보를 사용한 경우 1 |

### 구현 예시

```python
external = external.sort_values("Date")
external = external.ffill()
# bfill 금지: 미래 데이터로 과거 결측값을 채우지 않는다.

for col in external_feature_cols:
    if col.endswith(("_ret_1d", "_ret_5d", "_vol_20")):
        external[f"{col}_lag1_kr"] = external[col].shift(1)
```

`feature_selection.py`에서는 `gspc_ret_1d`, `ixic_ret_1d` 같은 원본 피처가 아니라 `*_lag1_kr` 피처만 선택하도록 제한한다.

---

## 2.2 목표값을 실전 매매 기준으로 확장

### 문제

현재 목표값은 다음 거래일 종가 기준 로그수익률이다. 이는 연구용으로 단순하고 명확하지만, 실제 매매에서는 다음 질문이 남는다.

- 신호는 장마감 후 생성되는가, 장중 생성되는가?
- 실제 진입가는 익일 시가인가, 종가인가, VWAP인가?
- 거래비용과 슬리피지를 반영하면 기대수익이 양수인가?
- 종목 수익률이 시장 상승 덕분인지, 개별 알파인지 구분되는가?

### 수정 권장

기존 `target_log_return`은 유지하되, 아래 목표값을 추가한다.

| 신규 목표값 | 계산 개념 | 사용 목적 |
|---|---|---|
| `target_log_return_1d_cc` | `log(Close_{t+1}/Close_t)` | 기존 close-to-close 연구 유지 |
| `target_log_return_1d_co` | `log(Open_{t+1}/Close_t)` | 장마감 후 신호, 익일 시가 진입 가능성 평가 |
| `target_log_return_1d_oc` | `log(Close_{t+1}/Open_{t+1})` | 익일 장중 보유 성과 평가 |
| `target_excess_return_1d` | 종목 1일 수익률 - KOSPI/KOSDAQ 또는 섹터 수익률 | 시장 노출 제거 후 알파 학습 |
| `target_net_return_1d` | 목표수익률 - 추정 거래비용 - 추정 슬리피지 | 실제 투자 가능 기대수익률 |
| `target_up_after_cost` | `target_net_return_1d > 0` | 비용 차감 후 상승 분류 |
| `target_return_5d` | 5거래일 forward return | 스윙 전략 또는 저회전 전략 검토 |

### 권장 정책

- 기본 모델 학습 목표: `target_excess_return_1d` 또는 `target_net_return_1d`
- 랭킹 모델 목표: `target_excess_return_1d`의 일자별 횡단면 순위
- 매수·매도·관망 판단: `predicted_net_return`, `predicted_risk`, `liquidity_score`를 함께 사용

---

## 2.3 모델 입력 피처 선택 규칙 수정

### 문제

현재 선택 규칙은 접두사 기반이다. 이 경우 다음 문제가 생길 수 있다.

- `ma_5`, `ma_20` 등 원시 가격 레벨이 모델에 들어가면 종목 가격대 차이가 신호처럼 학습될 수 있다.
- `{alias}_close` 같은 외부 지수 레벨은 장기 추세나 단위 차이에 민감하다.
- 접두사 허용 방식은 새 컬럼 추가 시 의도하지 않은 피처가 자동 유입될 가능성이 있다.

### 수정 권장

1. 원시 레벨값은 기본 모델 입력에서 제외한다.
2. 모델 피처는 수익률, 비율, 거리, z-score, rank, quantile, regime score 중심으로 제한한다.
3. `allow_prefixes`보다 명시적 allowlist 또는 feature manifest를 우선한다.
4. `selected_feature_columns.json` 또는 `feature_manifest.csv`를 학습 산출물로 저장한다.

### 제외 권장

| 피처 패턴 | 사유 |
|---|---|
| `ma_{N}` | 원시 가격 레벨. 종목 간 가격 단위 왜곡 가능 |
| `{external_alias}_close` | 외부 지수·환율·금리의 원시 레벨. 비정상성 위험 |
| `Close`, `Open`, `High`, `Low` | 원시 가격. 조정 여부와 가격대 영향 큼 |
| `value_traded` 단독 | 대형주 편향. 로그 변환 또는 rank로 대체 권장 |

### 허용 권장

| 피처 패턴 | 사유 |
|---|---|
| `ret_{N}d` | 종목 자체 모멘텀 |
| `excess_ret_{N}d` | 시장 대비 상대 모멘텀 |
| `close_to_ma_{N}` | 가격의 이동평균 대비 상대 위치 |
| `vol_{N}` | 변동성 |
| `atr_pct` | 가격 대비 ATR |
| `volume_z20`, `turnover_z20` | 거래량·거래대금 이상치 |
| `*_ret_*_lag1_kr` | 시점 통제된 외부시장 수익률 |
| `*_vol_*_lag1_kr` | 시점 통제된 외부시장 변동성 |
| `rank_*`, `z_*`, `q_*` | 횡단면 상대 위치 |

---

## 2.4 유동성·거래비용 피처 추가

### 문제

거래대금 순위와 거래량 비율은 유용하지만, 실제 매매 가능성을 판단하기에는 부족하다. 예측수익률이 높아도 호가 스프레드, 슬리피지, 거래대금 부족, 급등락 후 체결 불리함이 크면 실전 수익으로 연결되지 않는다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 투자적 의미 |
|---|---|---|
| `log_value_traded` | `log1p(Close * Volume)` | 거래대금 규모의 안정적 표현 |
| `value_traded_ma20` | 20일 평균 거래대금 | 평균 체결 가능 규모 |
| `value_traded_z20` | 거래대금 20일 z-score | 수급 이벤트 또는 과열 탐지 |
| `turnover_stability_20` | 20일 거래대금 표준편차 / 평균 | 유동성 안정성 |
| `amihud_illiq_20` | `mean(abs(return) / value_traded)` | 가격충격 비용 추정 |
| `range_to_value_20` | `range_pct / log1p(value_traded)`의 20일 평균 | 변동성 대비 유동성 취약도 |
| `liquidity_filter_flag` | 20일 평균 거래대금이 최소 기준 이상이면 1 | 실전 편입 가능 여부 |
| `estimated_slippage_bps` | 유동성·변동성 기반 추정 슬리피지 | 비용 차감 기대수익 산출 |

### 신호 적용 권장

`predicted_return`만으로 종목을 선택하지 말고 아래 조건을 함께 적용한다.

```text
매수 후보 = predicted_net_return > threshold
        and liquidity_filter_flag == 1
        and estimated_slippage_bps < max_slippage_bps
        and turnover_stability_20 < max_turnover_instability
```

---

## 2.5 시장·섹터 상대 피처 추가

### 문제

현재 종목 자체 수익률과 외부시장 수익률은 존재하지만, 종목이 시장 대비 강한지 약한지를 직접 표현하는 피처가 부족하다. 투자 판단에서는 “올랐다”보다 “시장보다 강하게 올랐다”가 더 중요하다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 설명 |
|---|---|---|
| `benchmark_ret_1d` | 종목 소속 시장의 1일 수익률 | KOSPI/KOSDAQ 구분 필요 |
| `excess_ret_1d` | `ret_1d - benchmark_ret_1d` | 단기 상대강도 |
| `excess_ret_5d` | `ret_5d - benchmark_ret_5d` | 주간 상대강도 |
| `excess_ret_20d` | `ret_20d - benchmark_ret_20d` | 월간 상대강도 |
| `beta_60d` | 60일 종목 수익률과 벤치마크 수익률의 회귀 베타 | 시장 민감도 |
| `idiosyncratic_vol_60d` | 시장 회귀 잔차의 60일 변동성 | 개별 리스크 |
| `residual_momentum_20d` | 시장 회귀 잔차 누적 수익률 | 순수 개별 모멘텀 |
| `corr_to_market_60d` | 60일 시장 수익률 상관계수 | 분산효과와 시장 동조화 |
| `sector_excess_ret_20d` | 종목 수익률 - 섹터 수익률 | 섹터 내 강자 선별 |

### 권장 사용법

- 전체 시장 랠리 국면에서는 `ret_20d`보다 `excess_ret_20d`를 우선한다.
- 고베타 종목은 상승장에서 예측수익이 높게 나올 수 있으므로 `beta_60d`와 함께 해석한다.
- 포트폴리오 구성 시 `corr_to_market_60d`가 과도하게 높은 종목만 몰리지 않도록 제한한다.

---

## 2.6 횡단면 랭킹·분위수 피처 추가

### 문제

일봉 기반 주식 모델은 개별 종목 시계열 패턴뿐 아니라 같은 날짜의 종목 간 상대순위가 중요하다. 특히 매일 상위 N개 종목을 선택하는 전략이라면 횡단면 피처가 성능에 큰 영향을 준다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 설명 |
|---|---|---|
| `rank_ret_5d_daily` | 같은 날짜 내 `ret_5d` 순위 | 단기 모멘텀 상대순위 |
| `rank_excess_ret_20d_daily` | 같은 날짜 내 `excess_ret_20d` 순위 | 시장 대비 강도 |
| `rank_vol_20_daily` | 같은 날짜 내 `vol_20` 순위 | 위험 수준 상대비교 |
| `rank_value_traded_daily` | 거래대금 순위 | 기존 `turnover_rank_daily`의 명확한 명칭 |
| `q_ret_20d_daily` | 같은 날짜 내 수익률 분위수 | 상·중·하 그룹 구분 |
| `q_liquidity_daily` | 유동성 분위수 | 체결 가능성 그룹 |
| `cross_sectional_momentum_score` | 모멘텀·수급·거래대금 순위 조합 | 랭킹 전략용 종합점수 |

### 주의사항

- 횡단면 순위는 반드시 같은 `Date` 내에서만 계산한다.
- 학습·검증 분할 이후의 전체 기간 분위수를 사용하면 누수 가능성이 있으므로, 날짜별 순위 또는 학습 구간 기준 변환을 사용한다.
- 상장폐지 종목과 거래정지 종목이 빠진 유니버스로만 계산하면 생존편향이 발생할 수 있다.

---

## 2.7 수급 피처 스케일링 개선

### 문제

외국인·기관 순매수 피처는 실전적으로 중요하지만, 현재 구조에는 다음 한계가 있다.

- 선택 입력이 없으면 대부분 `0.0`으로 보정되므로 “데이터 없음”과 “순매수 없음”이 구분되지 않는다.
- `foreign_high_conviction_buy_flag`와 `institution_high_conviction_buy_flag`의 1,000억 원 기준은 대형주에 유리하고 중소형주에는 과도하게 높다.
- 순매수 금액 자체보다 거래대금, 시가총액, 유동주식수 대비 강도가 더 중요하다.

### 수정 권장

1. 수급 데이터 부재 여부를 나타내는 `flow_missing_flag`를 추가한다.
2. 입력이 없을 때 `0.0` 보정값을 모델이 실제 0으로 오해하지 않도록 결측 플래그를 함께 사용한다.
3. 고확신 매수 기준은 고정 금액이 아니라 비율·z-score·분위수 기준으로 변경한다.
4. 누적 순매수는 3일·5일뿐 아니라 20일 누적과 지속성 지표를 추가한다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 설명 |
|---|---|---|
| `flow_missing_flag` | 수급 원천 데이터 부재 여부 | 결측과 0을 구분 |
| `foreign_buy_ratio_z20` | `foreign_buy_ratio`의 20일 z-score | 종목별 이례적 외국인 수급 |
| `institution_buy_ratio_z20` | `institution_buy_ratio`의 20일 z-score | 종목별 이례적 기관 수급 |
| `smart_money_strength_z20` | 스마트머니 강도의 20일 z-score | 외국인+기관 합산 강도 |
| `foreign_buy_persistence_5d` | 최근 5일 중 외국인 순매수 일수 / 5 | 지속성 |
| `institution_buy_persistence_5d` | 최근 5일 중 기관 순매수 일수 / 5 | 지속성 |
| `dual_accumulation_5d_flag` | 최근 5일 외국인·기관 동시 순매수 지속 | 쌍끌이 매수 |
| `flow_reversal_flag` | 강한 순매도 후 순매수 전환 | 수급 반전 |

### 고확신 플래그 대체안

```text
foreign_high_conviction_buy_flag =
    foreign_buy_ratio_z20 > 2.0
    and foreign_buy_persistence_5d >= 0.6
    and liquidity_filter_flag == 1
```

이 방식은 대형주와 중소형주를 같은 절대 금액 기준으로 비교하는 문제를 완화한다.

---

## 2.8 시장 국면 피처를 모델 입력 가능 형태로 확장

### 문제

현재 `market_regime`은 문자열 라벨이며 모델 입력에는 포함되지 않는다. 또한 `vol_20`의 75% 분위수를 전체 데이터 기준으로 계산하면 검증·테스트 구간 정보를 학습 구간에 반영할 위험이 있다.

### 수정 권장

1. 문자열 라벨은 표시용으로 유지한다.
2. 모델 입력용 수치 피처를 별도로 생성한다.
3. 분위수 기준은 전체 데이터가 아니라 학습 구간 또는 rolling 기준으로 계산한다.
4. 국면별 모델 성능 리포트를 생성한다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 설명 |
|---|---|---|
| `trend_score_20` | `close_to_ma_20`를 clipping 또는 z-score 처리 | 단기 추세 강도 |
| `trend_score_60` | `close_to_ma_60` 기반 | 중기 추세 강도 |
| `vol_regime_score` | `vol_20`의 rolling percentile | 변동성 국면 |
| `market_risk_on_score` | KOSPI, NASDAQ, SOX 상승 + VIX 하락 조합 | 위험선호 환경 |
| `market_risk_off_score` | VIX 상승 + 지수 하락 + 환율 상승 조합 | 위험회피 환경 |
| `regime_uptrend_flag` | 추세 상승 여부 | 모델 입력 가능 boolean |
| `regime_high_vol_flag` | 고변동성 여부 | 모델 입력 가능 boolean |

### 검증 권장

- `uptrend_low_vol`, `uptrend_high_vol`, `downtrend_high_vol` 등 국면별 IC와 수익률을 별도 측정한다.
- 특정 국면에서만 작동하는 피처는 전체 모델보다 국면별 가중치 또는 앙상블에 반영한다.

---

## 2.9 기술적 지표 확장 및 중복 관리

### 문제

현재 기술적 지표는 기본적인 모멘텀·오실레이터 중심이다. 그러나 기술적 지표는 서로 상관이 높기 때문에 단순히 추가만 하면 과최적화 위험이 커진다.

### 신규 피처 제안

| 신규 피처 | 계산 개념 | 투자적 의미 |
|---|---|---|
| `atr_pct` | `atr_14 / Close` | 가격 대비 변동성 |
| `adx_14` | Average Directional Index | 추세 강도 |
| `bb_width_20` | Bollinger Band 폭 / 중심선 | 변동성 압축·확장 |
| `bb_position_20` | 종가의 밴드 내 위치 | 과열·침체 위치 |
| `donchian_high_20_breakout_flag` | 직전 20일 고가 돌파 | 단기 돌파 |
| `donchian_low_20_breakdown_flag` | 직전 20일 저가 이탈 | 추세 훼손 |
| `reversal_1d_after_gap` | 큰 갭 후 장중 반전 여부 | 단기 과열·실망 매물 |
| `realized_skew_20` | 20일 수익률 왜도 | 급등락 비대칭성 |
| `downside_vol_20` | 음수 수익률만의 변동성 | 하방 위험 |

### 중복 관리

- RSI, Stochastic, CCI는 모두 과매수·과매도 계열이므로 상관을 점검한다.
- MACD, 이동평균 괴리, Donchian breakout은 추세 계열로 묶어 ablation test를 수행한다.
- 피처 그룹별로 “추가 전/후 IC, turnover, MDD, Sharpe 변화”를 비교한다.

---

## 2.10 뉴스·공시 피처 정책

### 현재 정책 평가

뉴스·공시를 표시 전용으로 유지하고 모델 입력, 기대수익률 순위, 자동 신호 결정에서 제외하는 정책은 보수적이고 타당하다. 특히 뉴스 수집 시점, 기사 수정 시점, 공시 반영 시점이 불명확하면 강한 데이터 누수가 발생할 수 있다.

### 수정 권장

기본 정책은 유지하되, 별도 실험 모드에서만 아래를 검토한다.

| 항목 | 권장 정책 |
|---|---|
| 기본 운영 | 뉴스·공시 피처는 표시 전용 유지 |
| 실험 모드 | `--enable-event-features-experiment` 같은 명시적 옵션 필요 |
| 시점 통제 | 기사 발행시각·공시 접수시각이 `decision_time` 이전인 경우만 사용 |
| 검증 | 뉴스·공시 값을 변경해도 기본 모델 피처가 변하지 않는 테스트 유지 |
| 출력 | 모델 신호와 별도로 “정성 리스크 코멘트”로 제공 |

---

## 3. 권장 신규 피처 카탈로그

## 3.1 즉시 추가 권장 피처

| 피처명 | 공식 또는 계산 개념 | 입력 필요 | 모델 입력 여부 | 우선순위 |
|---|---|---|---|---|
| `external_stale_flag` | 외부 데이터가 최신 관측치가 아닌 경우 1 | 외부 데이터 | 허용 | P0 |
| `gspc_ret_1d_lag1_kr` | 한국 시장 기준 사용 가능하도록 지연한 S&P 500 수익률 | 외부 데이터 | 허용 | P0 |
| `ixic_ret_1d_lag1_kr` | 지연 NASDAQ 수익률 | 외부 데이터 | 허용 | P0 |
| `sox_ret_1d_lag1_kr` | 지연 SOX 수익률 | 외부 데이터 | 허용 | P0 |
| `vix_ret_1d_lag1_kr` | 지연 VIX 변화율 | 외부 데이터 | 허용 | P0 |
| `target_excess_return_1d` | 종목 수익률 - 벤치마크 수익률 | OHLCV + 시장지수 | 목표값 | P0 |
| `target_net_return_1d` | 수익률 - 비용 - 슬리피지 | OHLCV + 비용모델 | 목표값 | P0 |
| `liquidity_filter_flag` | 평균 거래대금 기준 통과 여부 | OHLCV | 허용 | P0 |
| `estimated_slippage_bps` | 유동성·변동성 기반 슬리피지 추정 | OHLCV | 허용 | P0 |
| `flow_missing_flag` | 수급 데이터 부재 여부 | 수급 데이터 | 허용 | P0 |

## 3.2 성능 개선 목적 피처

| 피처명 | 공식 또는 계산 개념 | 투자적 의미 | 우선순위 |
|---|---|---|---|
| `excess_ret_5d` | `ret_5d - benchmark_ret_5d` | 시장 대비 단기 상대강도 | P1 |
| `excess_ret_20d` | `ret_20d - benchmark_ret_20d` | 월간 상대강도 | P1 |
| `beta_60d` | 60일 시장 베타 | 시장 노출 | P1 |
| `idiosyncratic_vol_60d` | 시장 회귀 잔차 변동성 | 개별 위험 | P1 |
| `residual_momentum_20d` | 시장 회귀 잔차 누적수익 | 순수 알파 모멘텀 | P1 |
| `rank_excess_ret_20d_daily` | 일자별 초과수익률 순위 | 횡단면 선별 | P1 |
| `volume_z20` | 종목별 거래량 z-score | 거래량 이벤트 | P1 |
| `value_traded_z20` | 종목별 거래대금 z-score | 자금 유입 이벤트 | P1 |
| `smart_money_strength_z20` | 스마트머니 강도 z-score | 이례적 수급 | P1 |
| `foreign_buy_persistence_5d` | 최근 5일 외국인 순매수 지속도 | 수급 지속성 | P1 |
| `market_risk_on_score` | 지수 상승·VIX 하락 조합 | 위험선호 환경 | P1 |
| `vol_regime_score` | 변동성 rolling percentile | 국면 인식 | P1 |

## 3.3 추가 연구 후보 피처

| 피처명 | 설명 | 우선순위 |
|---|---|---|
| `adx_14` | 추세 강도 | P2 |
| `bb_width_20` | 변동성 압축·확장 | P2 |
| `bb_position_20` | 밴드 내 가격 위치 | P2 |
| `downside_vol_20` | 하방 변동성 | P2 |
| `realized_skew_20` | 수익률 왜도 | P2 |
| `donchian_high_20_breakout_flag` | 20일 고가 돌파 | P2 |
| `sector_excess_ret_20d` | 섹터 대비 상대성과 | P2, 섹터 데이터 필요 |
| `market_cap_bucket` | 시가총액 그룹 | P2, 시총 데이터 필요 |
| `free_float_turnover` | 유동주식수 대비 거래량 | P2, 유동주식수 필요 |

---

## 4. 파일별 수정 권장사항

| 파일 | 수정 권장사항 | 우선순위 |
|---|---|---|
| `src/features/external_features.py` | `bfill()` 제거, 외부 피처 lag 처리, 대체 심볼 사용 플래그 추가, `available_at` 메타데이터 관리 | P0 |
| `src/features/price_features.py` | adjusted OHLCV 지원, 비용 차감 목표값, 초과수익률 목표값, 유동성·슬리피지 피처 추가 | P0 |
| `src/features/feature_selection.py` | 원시 가격·원시 외부 레벨 제외, lag된 외부 피처만 허용, 명시적 allowlist 또는 manifest 도입 | P0 |
| `src/features/investment_signals.py` | `predicted_return` 단독 신호에서 `predicted_net_return`, 유동성, 리스크 조건을 함께 사용하도록 변경 | P0 |
| `src/features/regime_features.py` | 문자열 regime 외에 수치형 regime score와 rolling percentile 방식 추가 | P1 |
| `src/features/technical_indicators.py` | `atr_pct`, ADX, Bollinger, Donchian, downside volatility 추가 | P2 |
| `src/config/settings.py` | 거래비용, 슬리피지, 최소 거래대금, 외부 피처 지연 정책, 목표값 종류 설정 추가 | P0 |
| `tests/` | 외부 피처 지연 테스트, bfill 금지 테스트, 원시 레벨 피처 제외 테스트, 비용 차감 target 테스트 추가 | P0 |

---

## 5. 권장 설정 구조 예시

```json
{
  "feature": {
    "lookback_windows": [1, 2, 3, 5, 10, 20, 60],
    "moving_average_windows": [5, 10, 20, 60, 120],
    "volatility_windows": [5, 20, 60],
    "use_adjusted_prices": true,
    "add_cross_sectional_features": true,
    "add_liquidity_features": true,
    "add_market_relative_features": true
  },
  "target": {
    "primary": "target_net_return_1d",
    "also_build": [
      "target_log_return_1d_cc",
      "target_excess_return_1d",
      "target_return_5d"
    ],
    "execution_price": "next_open",
    "benchmark": "auto_by_market"
  },
  "external": {
    "enabled": true,
    "lag_for_korean_equities": 1,
    "allow_bfill": false,
    "keep_raw_external_columns": true,
    "select_only_lagged_external_features": true,
    "market_symbols": ["^KS11", "^KQ11", "^GSPC", "^IXIC", "NQ=F", "^SOX", "^VIX", "KRW=X", "^TNX"]
  },
  "cost": {
    "commission_bps": 1.5,
    "tax_bps": 0.0,
    "base_slippage_bps": 5.0,
    "slippage_model": "liquidity_volatility_proxy"
  },
  "liquidity": {
    "min_value_traded_ma20": 1000000000,
    "max_position_pct_of_adv20": 0.05,
    "exclude_suspended_or_zero_volume": true
  }
}
```

---

## 6. 투자 신호 산출 방식 개선안

### 현재 구조

```text
predicted_return 기준으로 매수·매도·관망 판단
```

### 권장 구조

```text
predicted_net_return
= predicted_raw_return
- estimated_transaction_cost
- estimated_slippage
- risk_penalty
```

최종 신호는 아래처럼 다중 조건으로 결정한다.

| 조건 | 매수 후보 기준 예시 |
|---|---|
| 기대수익 | `predicted_net_return > buy_threshold` |
| 리스크 | `predicted_vol_20 < max_allowed_vol` |
| 유동성 | `liquidity_filter_flag == 1` |
| 시장 국면 | `market_risk_off_score`가 과도하지 않음 |
| 수급 | `smart_money_strength_z20 > 0` 또는 수급 결측 시 중립 처리 |
| 거래비용 | `estimated_slippage_bps < max_slippage_bps` |
| 과열 방지 | `rank_vol_20_daily` 또는 `rsi_overbought_sell_flag` 확인 |

### 신호 등급 예시

| 등급 | 기준 |
|---|---|
| Strong Buy | 기대수익, 유동성, 수급, 시장국면이 모두 우호적 |
| Buy | 기대수익과 유동성은 충족하나 일부 보조 신호 중립 |
| Watch | 기대수익은 양수지만 비용·리스크 차감 후 매력 부족 |
| Avoid | 유동성 부족, 고변동성, 위험회피 국면, 비용 과다 |
| Sell / Reduce | 예측 순수익률 음수 또는 추세 훼손·수급 악화 |

---

## 7. 테스트 및 검증 체크리스트

## 7.1 데이터 누수 테스트

| 테스트 | 검증 내용 |
|---|---|
| `test_external_features_do_not_bfill` | 외부 피처가 미래값으로 과거 결측을 채우지 않는지 확인 |
| `test_external_features_are_lagged_for_kr_market` | 한국 주식 모델 입력에는 lag된 외부 피처만 선택되는지 확인 |
| `test_raw_external_close_not_selected` | `{alias}_close` 원시 레벨이 모델 입력에서 제외되는지 확인 |
| `test_raw_moving_average_not_selected` | `ma_{N}` 원시 가격 이동평균이 제외되는지 확인 |
| `test_news_disclosure_still_display_only` | 뉴스·공시 값 변경이 모델 피처에 영향을 주지 않는지 확인 |
| `test_missing_flow_has_flag` | 수급 데이터 부재와 실제 0이 구분되는지 확인 |

## 7.2 투자 성과 검증

| 검증 항목 | 권장 지표 |
|---|---|
| 예측력 | IC, Rank IC, 방향성 hit ratio |
| 랭킹 성능 | 일자별 상위 decile과 하위 decile 수익률 차이 |
| 실전 성과 | 비용 차감 CAGR, Sharpe, Sortino, MDD |
| 회전율 | 월간 turnover, 평균 보유기간 |
| 유동성 | ADV 대비 주문 비중, 예상 슬리피지 |
| 안정성 | 연도별·국면별·시장별 성능 |
| 피처 안정성 | permutation importance, SHAP, feature importance rank stability |
| 중복성 | 피처 상관계수, VIF, 그룹별 ablation |

## 7.3 백테스트 프로토콜

1. 무작위 분할 대신 시간 순서 기반 walk-forward validation을 사용한다.
2. 학습 기간, 검증 기간, 테스트 기간을 명확히 분리한다.
3. 종목 유니버스는 해당 시점에 실제 투자 가능했던 종목으로 구성한다.
4. 거래정지, 상장폐지, 액면분할, 배당, 권리락을 조정한다.
5. 거래비용과 슬리피지를 기본값이 아니라 종목별 유동성에 따라 다르게 적용한다.
6. 상위 N개 매수 전략이라면 예측값 자체보다 일자별 랭킹 품질을 우선 평가한다.
7. 피처 추가 전후의 성능 개선이 특정 기간이나 특정 종목군에만 의존하는지 확인한다.

---

## 8. 단계별 적용 로드맵

## Phase 1: 누수 방지 및 모델 입력 정리

- 외부시장 `bfill()` 제거
- 외부시장 lag 피처 추가
- 원시 `ma_`, 외부 `*_close` 모델 입력 제외
- feature manifest 저장
- 수급 결측 플래그 추가
- 관련 단위 테스트 추가

## Phase 2: 실전 투자 가능성 반영

- `target_net_return_1d`, `target_excess_return_1d` 추가
- 거래비용·슬리피지 설정 추가
- 유동성 필터와 예상 슬리피지 피처 추가
- 신호 판단 로직을 `predicted_return` 단독 기준에서 `predicted_net_return + risk + liquidity` 기준으로 변경

## Phase 3: 알파 품질 개선

- 시장 대비 초과수익률, 베타, 잔차 모멘텀 추가
- 횡단면 rank·quantile 피처 추가
- 수급 지속성·z-score 피처 추가
- 국면별 수치형 피처 추가

## Phase 4: 고급 연구 피처 및 운영 리포트

- ADX, Bollinger, Donchian, downside volatility 추가
- 섹터 상대성과, 시가총액 그룹, 유동주식수 회전율 추가
- 국면별 성과 리포트와 피처 중요도 안정성 리포트 자동화
- 뉴스·공시 이벤트 피처는 별도 실험 모드에서만 검증

---

## 9. 최종 권고

현재 피처 구조는 기본적인 가격·기술적 지표·외부시장·수급 피처를 갖추고 있어 연구 출발점으로는 적절하다. 다만 실전 투자 모델로 사용하려면 단순히 지표를 더 추가하기보다 다음 세 가지 수정이 먼저 필요하다.

1. **외부시장과 결측 처리의 누수 가능성을 제거한다.**  
   `bfill()` 제거와 외부 피처 지연 처리는 최우선 수정사항이다.

2. **예측 목표를 실제 매매 성과에 맞춘다.**  
   `target_log_return`만으로는 부족하며, 비용 차감 수익률과 시장 대비 초과수익률을 함께 학습·평가해야 한다.

3. **모델 입력을 원시 레벨값이 아닌 투자적으로 해석 가능한 상대값으로 제한한다.**  
   가격 레벨, 지수 레벨, 단순 이동평균 레벨은 제외하고 수익률, 거리, 비율, z-score, rank, regime score 중심으로 재구성하는 것이 바람직하다.

이 세 가지를 반영한 뒤 유동성, 수급 지속성, 시장 상대강도, 국면 피처를 단계적으로 추가하면 모델의 백테스트 신뢰도와 실제 투자 적용 가능성이 모두 개선될 가능성이 높다.
