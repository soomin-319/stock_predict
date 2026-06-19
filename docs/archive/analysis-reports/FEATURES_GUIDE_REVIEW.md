# 피처 가이드 전문가 검토 및 개선 제안서

> **검토 기준**: 한국 주식시장(KOSPI·KOSDAQ) 퀀트 투자 실무 관점  
> **원본 문서**: `FEATURES_GUIDE.md`  
> **검토 목적**: 예측 정확도 향상, 룩어헤드 바이어스 차단, 한국 시장 특수성 반영

---

## 목차

1. [전체 구조 평가](#1-전체-구조-평가)
2. [가격·거래량 피처 개선](#2-가격거래량-피처-개선)
3. [기술적 지표 개선](#3-기술적-지표-개선)
4. [외부 시장 피처 개선](#4-외부-시장-피처-개선)
5. [수급 피처 개선](#5-수급-피처-개선)
6. [시장 국면 라벨 개선](#6-시장-국면-라벨-개선)
7. [목표값 설계 개선](#7-목표값-설계-개선)
8. [한국 시장 특화 피처 추가](#8-한국-시장-특화-피처-추가)
9. [데이터 품질 및 룩어헤드 바이어스 관리](#9-데이터-품질-및-룩어헤드-바이어스-관리)
10. [우선순위별 개선 로드맵](#10-우선순위별-개선-로드맵)

---

## 1. 전체 구조 평가

### 1.1 잘 설계된 부분

- 뉴스·공시를 모델 입력에서 명시적으로 분리한 설계는 **재현 가능성과 과적합 방지** 측면에서 올바른 방향이다.
- `DISPLAY_ONLY_CONTEXT_COLUMNS`와 테스트 강제화로 경계를 코드 수준에서 보장하는 구조가 견고하다.
- 외부 심볼 다운로드 실패 시 대체 심볼을 시도하는 fallback 로직은 실무 안정성을 높인다.

### 1.2 구조적 리스크 요약

| 분류 | 주요 리스크 |
|---|---|
| 룩어헤드 바이어스 | 외부 시장 피처의 시차 미보정, ffill 방향 문제 |
| 한국 시장 특수성 부재 | 상한가·하한가, 공매도, 프로그램 매매 미반영 |
| 피처 포화 가능성 | 유사 피처 중복, 다중공선성 미검증 |
| 목표값 단일성 | 단기 로그수익률만 사용, 리스크 조정 미실시 |
| 수급 데이터 공백 | `_fetch_flow()` 자리표시자로 수급 피처 전체가 `0.0`으로 대체될 가능성 |

---

## 2. 가격·거래량 피처 개선

### 2.1 누락된 핵심 가격 피처

#### VWAP (거래량 가중 평균가)

기관 투자자의 체결 기준가로 활용되며, VWAP 대비 종가 위치는 당일 매수·매도 강도를 나타낸다.

```
vwap = sum(price * volume) / sum(volume)  # 일중 분봉 기준
close_to_vwap = Close / vwap - 1
```

> **실무 중요도**: 높음. 기관은 VWAP 알고리즘을 체결 기준으로 사용하므로, 종가가 VWAP 위에 있으면 기관 매수 강도가 강하다고 해석된다.

#### 피벗 포인트 (Pivot Point)

전일 고·저·종가 기반의 당일 지지·저항 레벨. 단기 트레이딩 모델에서 매우 유효하다.

```
pivot = (High_{t-1} + Low_{t-1} + Close_{t-1}) / 3
r1    = 2 * pivot - Low_{t-1}
s1    = 2 * pivot - High_{t-1}
close_to_pivot = Close / pivot - 1
```

#### 갭 지속성 피처

갭 발생 후 당일 종가가 갭을 메우는지(갭 채움)를 나타내는 지표. 갭 방향과 당일 인트라데이 방향의 일치 여부는 모멘텀 신뢰도를 나타낸다.

```
gap_filled_flag = (gap_return > 0) & (intraday_return < -gap_return * 0.5)
# 혹은 반대 방향
```

### 2.2 거래량 피처 보강

현재 `vol_ratio_20`(20일 평균 대비 당일 거래량)은 있으나, 아래 피처가 추가로 필요하다.

| 신규 피처 | 계산 개념 | 투자 의미 |
|---|---|---|
| `vol_surge_flag` | `vol_ratio_20 > 2.0` | 비정상적 거래량 폭발, 세력 개입 or 뉴스 반응 신호 |
| `vol_trend_5d` | 5일 거래량 추세 기울기 | 거래량 증가 추세 = 관심 집중 |
| `price_vol_divergence` | 가격 상승 + 거래량 감소 | 추세 약화의 조기 신호 |
| `up_vol_ratio` | 상승일 평균 거래량 / 하락일 평균 거래량 (20일) | OBV보다 직관적인 매수 강도 지표 |

### 2.3 이동평균 피처 보강

현재 `close_to_ma_N`만 있으나, 이동평균 **간의 관계**가 더 중요하다.

```
# 골든크로스 / 데스크로스
ma_cross_5_20   = ma_5 / ma_20 - 1   # 양수: 단기 MA가 중기 MA 위
ma_cross_20_60  = ma_20 / ma_60 - 1
ma_alignment    = (ma_5 > ma_20) & (ma_20 > ma_60)  # 정배열 여부 (강세 구조 필터)
```

> **실무 중요도**: 높음. `ma_alignment` 정배열 조건은 상승 추세 확인의 1차 필터로 실전에서 광범위하게 사용된다.

---

## 3. 기술적 지표 개선

### 3.1 추가해야 할 핵심 지표

#### 볼린저 밴드 (Bollinger Bands)

현재 완전히 누락되어 있다. 변동성 기반의 가격 채널로, 한국 시장에서 매우 높은 실용도를 가진다.

| 피처 | 계산 개념 |
|---|---|
| `bb_upper`, `bb_lower` | MA(20) ± 2 × std(20) |
| `bb_pct` | (Close - bb_lower) / (bb_upper - bb_lower) |
| `bb_squeeze` | 밴드 폭이 6개월 최저치인지 여부 |

`bb_squeeze`는 변동성 압축 후 폭발적 상승/하락 전 구조를 포착하는 데 유효하다.

#### ADX (Average Directional Index)

추세의 **강도**를 측정. RSI와 달리 방향이 아닌 강도만을 나타내므로 필터로 유용하다.

```
adx_14 = ADX(14)
strong_trend_flag = adx_14 > 25   # 추세 강도 충분
```

#### Williams %R

Stochastic과 유사하나 역산 방식으로 계산되며, 과매수·과매도 구간에서 빠른 반응을 보여 단기 신호 생성에 유효하다.

```
williams_r_14 = (Highest_High_14 - Close) / (Highest_High_14 - Lowest_Low_14) * -100
```

#### 가격 모멘텀 가속도

단순 수익률이 아닌 수익률의 **변화율**로, 모멘텀 전환점 포착에 핵심적이다.

```
momentum_accel = ret_5d - ret_20d   # 단기 모멘텀이 중기 대비 가속 여부
```

### 3.2 현재 지표의 개선 사항

#### RSI 개선

현재 RSI 30~35를 매수 관찰 구간으로 정의하고 있으나, 실전에서는 다음이 더 유효하다.

```
rsi_divergence_bull = (Close < Close_{-5}) & (rsi_14 > rsi_14_{-5})  # 가격 하락 + RSI 상승 = 강세 다이버전스
rsi_divergence_bear = (Close > Close_{-5}) & (rsi_14 < rsi_14_{-5})  # 약세 다이버전스
```

> 단순 레벨보다 **다이버전스**가 실제 반전 신호로 훨씬 강하다.

#### MACD 개선

현재 `macd_hist`만 있으나, 히스토그램의 **방향 전환**이 핵심이다.

```
macd_hist_cross_up   = (macd_hist > 0) & (macd_hist_{-1} <= 0)  # 히스토그램 0선 상향 돌파
macd_hist_cross_down = (macd_hist < 0) & (macd_hist_{-1} >= 0)  # 하향 이탈
```

---

## 4. 외부 시장 피처 개선

### 4.1 추가해야 할 외부 심볼

| 추가 심볼 | 별칭 | 이유 |
|---|---|---|
| `000001.SS` (상해종합) | `sse` | 한국 시장과 중국 시장의 상관관계는 0.4~0.6으로 높음 |
| `^HSI` (항생지수) | `hsi` | 홍콩 경유 외국인 자금의 방향 선행 지표 |
| `CL=F` (WTI 원유) | `wti` | 정유·화학·항공 섹터에 직접 영향 |
| `GC=F` (금) | `gold` | 안전자산 선호 지표, 리스크오프 신호 |
| `DX-Y.NYB` (달러 인덱스) | `dxy` | 원/달러보다 글로벌 달러 강도를 직접 반영 |
| `^SOX` + `3711.KS` (삼성전자) 상관 | `sox_krw_corr` | 반도체 섹터 국내 반응 lag 추정 |

> **한국 시장 특성**: KOSPI는 반도체(삼성전자·SK하이닉스) 비중이 40%를 초과하므로 SOX 지수와의 연동이 다른 아시아 시장보다 훨씬 강하다.

### 4.2 시차(Lag) 처리 — 가장 중요한 개선 사항

현재 문서에 다음 경고가 있으나 코드 레벨에서 강제되지 않는다.

> "실전 연구에서는 한국 시장 시점에 실제 관측 가능했던 값인지 확인하고, 필요하면 외부 피처를 1거래일 지연해야 한다."

**반드시 자동화가 필요한 시차 분류**:

| 외부 시장 | 관측 가능 시점 | 처리 방법 |
|---|---|---|
| 미국 지수 (당일 종가) | 한국 다음 거래일 오전 | **반드시 1거래일 lag** |
| 미국 지수 (전일 종가 기준) | 한국 당일 장 전 | lag 없이 사용 가능 |
| VIX | 미국 장마감 후 | 1거래일 lag |
| 원/달러 환율 | 한국 당일 실시간 | lag 없이 사용 가능 |
| 나스닥 선물 (NQ=F) | 한국 장 중 실시간 | 당일 사용 가능하나 종가는 장마감 후 확정 |

**코드 수준 강제 방법**:

```python
# external_features.py에 추가 권고
US_MARKET_SYMBOLS = ['^GSPC', '^IXIC', 'NQ=F', '^SOX', '^VIX', '^TNX']
KR_MARKET_SYMBOLS = ['KRW=X', '^KS11', '^KQ11']

def add_external_market_features_with_coverage(df, config):
    for symbol in US_MARKET_SYMBOLS:
        # 미국 지수는 반드시 1거래일 lag 적용
        df[f'{alias}_ret_1d'] = df[f'{alias}_ret_1d'].shift(1)
```

### 4.3 외부 피처 결측 처리 개선

현재 `ffill()` 후 `bfill()`을 적용하는데, `bfill()`은 **미래 데이터를 과거로 역충전**하므로 학습 데이터에서 룩어헤드 바이어스가 된다.

```python
# 현재 (위험)
df.fillna(method='ffill').fillna(method='bfill')

# 권고 (bfill 제거 또는 학습/예측 단계 분리)
df.fillna(method='ffill')  # ffill만 사용
# 초기 결측값은 0 또는 섹션 평균으로 대체
df.fillna(0)
```

---

## 5. 수급 피처 개선

### 5.1 현재 수급 피처의 구조적 문제

현재 `_fetch_flow()`가 자리표시자이므로 수급 피처 전체가 `0.0`으로 채워질 가능성이 높다. 이 경우 수급 피처는 모델에 아무런 정보를 제공하지 않는다.

**단기 해결 방법**: 수급 피처를 입력 CSV로 받을 때 데이터 존재 여부를 반드시 검증한다.

```python
def validate_flow_data(df):
    flow_cols = ['foreign_net_buy', 'institution_net_buy']
    for col in flow_cols:
        if col in df.columns:
            non_zero_ratio = (df[col] != 0).mean()
            if non_zero_ratio < 0.05:
                warnings.warn(f"{col}의 95% 이상이 0. 수급 피처가 placeholder일 가능성 높음.")
```

### 5.2 누락된 핵심 수급 피처

#### 공매도 관련 피처

한국 시장에서 공매도 비율과 잔고는 기관·헤지펀드의 방향성 뷰를 반영하는 중요한 역지표다.

| 피처 | 계산 개념 | 출처 |
|---|---|---|
| `short_sell_ratio` | 당일 공매도 거래량 / 총 거래량 | KRX 공시 |
| `short_balance` | 공매도 잔고 수량 | KRX 공시 |
| `short_balance_ratio` | 공매도 잔고 / 상장주식수 | 계산 |
| `short_squeeze_flag` | 공매도 비율 급감 + 가격 급등 동시 발생 | 복합 조건 |

#### 프로그램 매매 피처

차익·비차익 프로그램 매매는 지수 선물 기저(basis) 변동에 따라 자동으로 발생하며, 개별 종목 가격에 기계적 영향을 준다.

| 피처 | 투자 의미 |
|---|---|
| `program_buy_net` | 차익+비차익 프로그램 순매수 금액 |
| `basis_spread` | KOSPI200 선물 기저 (선물 - 현물) |
| `program_buy_ratio` | 프로그램 매수 / 총 매수대금 |

#### 신용잔고 피처

신용잔고 급증은 개인 투자자 레버리지 과열의 선행 지표이며, 이후 반대매매 리스크를 높인다.

```
margin_balance_change_5d = (margin_balance - margin_balance_{-5}) / margin_balance_{-5}
margin_balance_to_cap = margin_balance / market_cap  # 시총 대비 신용잔고 비율
```

### 5.3 고확신 임계값 개선

현재 외국인·기관 고확신 기준이 **1,000억 원**으로 고정되어 있어 시총이 작은 중소형주에는 적용이 불가능하다.

**개선 방향**: 절대금액 대신 상대 기준으로 변경

```python
# 현재 (문제)
foreign_high_conviction_buy_flag = foreign_net_buy > 1_000_00_000_000  # 고정 1,000억

# 권고 (상대 기준)
foreign_high_conviction_buy_flag = foreign_net_buy > market_cap * 0.005  # 시총의 0.5% 이상
# 또는
foreign_high_conviction_buy_flag = foreign_net_buy_z20 > 2.0  # z-score 2 이상
```

---

## 6. 시장 국면 라벨 개선

### 6.1 현재 시장 국면의 한계

현재 `market_regime`은 추세(uptrend/downtrend/sideways) × 변동성(high_vol/low_vol) 2차원 조합이다. 실제 투자 환경은 이보다 복잡하다.

### 6.2 추가 국면 차원

#### 유동성 국면

```
liquidity_regime = 'high_liq' if vol_ratio_20 > 1.3 else 'low_liq'
```

#### 섹터 로테이션 국면

반도체 섹터와 금융 섹터의 상대 강도를 비교하여 성장주/가치주 선호 국면을 구분한다.

```
growth_regime_flag   = (sox_ret_5d > kospi_ret_5d) & (gspc_ret_5d > tnx_ret_5d * -1)
defensive_regime_flag = (vix_ret_1d > 0.05) & (gold_ret_1d > 0.01)
```

#### 개선된 임계값

현재 `close_to_ma_20 > 0.01`을 추세 기준으로 사용하는데, 이는 코스닥 변동성에 비해 너무 작다.

| 시장 | 권고 추세 임계값 |
|---|---|
| KOSPI 대형주 | ±0.015 |
| KOSPI 중형주 | ±0.020 |
| KOSDAQ | ±0.030 |

```python
trend_threshold = config.get('trend_threshold_by_market', {
    'KOSPI_LARGE': 0.015,
    'KOSPI_MID': 0.020,
    'KOSDAQ': 0.030
})
```

### 6.3 시장 국면의 모델 입력 포함 검토

현재 `market_regime`은 표시 전용이다. 이를 원-핫 인코딩하여 모델 입력에 포함하면 국면별 수익률 패턴 차이를 학습에 활용할 수 있다.

```python
# feature_selection.py에 추가
regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
# 예: regime_uptrend_high_vol, regime_sideways_low_vol ...
```

---

## 7. 목표값 설계 개선

### 7.1 단일 목표값의 한계

현재 `target_log_return`(익일 로그수익률)만을 목표값으로 사용하는데, 이는 다음 문제를 가진다.

1. **거래비용 미반영**: 0.1~0.3% 수준의 매수·매도 수수료 + 증권거래세가 수익 예측을 크게 왜곡할 수 있다.
2. **리스크 무시**: 동일한 기대수익률이어도 변동성이 다르면 실질 투자 가치가 다르다.
3. **단기 과적합 위험**: 1일 수익률은 노이즈가 매우 높아 예측 자체가 어렵다.

### 7.2 목표값 개선 방안

#### 거래비용 조정 수익률

```python
TRANSACTION_COST = 0.003  # 수수료 0.15% × 2 + 증권거래세 0.18% ≈ 0.48%, 여유분 포함

target_net_return = target_log_return - TRANSACTION_COST
target_net_positive = target_net_return > 0  # 실질 매수 가치 여부
```

#### 다중 기간 목표값

```python
target_log_return_3d  = log(Close_{t+3} / Close_t)
target_log_return_5d  = log(Close_{t+5} / Close_t)
target_log_return_10d = log(Close_{t+10} / Close_t)
```

단기 목표값과 중기 목표값의 앙상블은 단기 노이즈를 줄이는 데 효과적이다.

#### 샤프 기반 목표값

```python
# 5거래일 기간 기준 rolling sharpe
target_sharpe_5d = (target_log_return_5d - rf_rate_5d) / rolling_vol_5d
```

### 7.3 분류 목표값 개선

현재 `target_up`은 단순 양수 여부인데, 임계값을 추가하면 노이즈성 신호를 걸러낼 수 있다.

```python
target_strong_up   = target_log_return > TRANSACTION_COST + vol_20 * 0.5  # 유의미한 상승
target_strong_down = target_log_return < -(TRANSACTION_COST + vol_20 * 0.5)
target_neutral     = ~(target_strong_up | target_strong_down)
```

---

## 8. 한국 시장 특화 피처 추가

### 8.1 가격 제한폭 관련 피처

한국은 상한가(+30%)·하한가(-30%)의 일일 가격 제한이 있어, 이에 근접하는 종목은 다음날 가격 형성 방식이 다르다.

| 피처 | 계산 개념 | 투자 의미 |
|---|---|---|
| `near_upper_limit` | `daily_return > 0.27` | 상한가 근접, 다음날 연속 상한가 가능성 |
| `upper_limit_flag` | `daily_return >= 0.295` | 상한가 달성 (물량 잠김) |
| `near_lower_limit` | `daily_return < -0.27` | 하한가 근접 |
| `lower_limit_flag` | `daily_return <= -0.295` | 하한가 달성 |
| `consecutive_up_limit` | 연속 상한가 횟수 | 테마주 과열 신호 |

> **실무 중요도**: 매우 높음. 상한가·하한가 종목은 일반 종목과 전혀 다른 메커니즘으로 움직이므로, 모델 학습 시 별도 처리하거나 필터링이 필요하다.

### 8.2 결산·배당 이벤트 피처

```python
ex_dividend_flag     = 배당락일 여부  # 배당락일 당일 인위적 가격 하락
rights_offering_flag = 유상증자 권리락일 여부
stock_split_flag     = 액면분할 여부
fiscal_year_end_flag = 결산월(12월, 3월) 여부  # 기관 포트폴리오 리밸런싱 수요
```

### 8.3 코스피200 편입·편출 효과

코스피200 구성 종목은 패시브 펀드 리밸런싱의 기계적 수요가 발생한다.

```python
kospi200_member_flag      = 코스피200 구성 여부
kospi200_rebalancing_flag = 반기 리밸런싱 시즌(6월·12월 선물 만기 주간) 여부
etf_tracking_demand       = 코스피200 내 종목의 시가총액 비중 변화율
```

### 8.4 옵션 만기일 효과 (매월 둘째 목요일)

선물·옵션 만기일 직전에는 프로그램 매매가 급증하여 가격 변동성이 커진다.

```python
days_to_expiry      = 선물·옵션 만기일까지 남은 거래일 수
expiry_week_flag    = 만기일까지 5거래일 이내 여부
triple_witch_flag   = 선물·옵션·ETF 동시 만기일(3월·6월·9월·12월) 여부
```

### 8.5 섹터 상대강도

한국 시장은 반도체·이차전지·바이오·금융 섹터의 로테이션이 뚜렷하다.

```python
sector_rel_strength_5d  = 해당 섹터 5일 수익률 - KOSPI 5일 수익률
sector_rank_in_kospi    = 해당 날짜 기준 섹터 수익률 순위
sector_leadership_flag  = sector_rank_in_kospi <= 3  # 상위 3개 섹터 여부
```

---

## 9. 데이터 품질 및 룩어헤드 바이어스 관리

### 9.1 룩어헤드 바이어스 위험 체크리스트

| 위험 항목 | 현재 상태 | 권고 조치 |
|---|---|---|
| 외부 시장 피처 시차 | 문서 경고는 있으나 미강제 | 코드 레벨 자동 lag 적용 |
| `bfill()` 사용 | 학습 데이터에 미래값 역충전 가능 | `ffill()`만 사용 |
| 52주 최고가 초기 구간 | 20개 관측치부터 계산 (정확도 낮음) | 최소 60개 관측치로 상향 |
| 목표값 생성 후 shift | `shift(-1)`로 정확히 내일 종가 사용 | 현행 유지, 마지막 행 제거 로직 확인 |
| 거래량 순위 | 당일 전체 종목 데이터 필요 | 당일 데이터 완결 후 계산 확인 |

### 9.2 피처 다중공선성 관리

현재 `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d` 등 유사 피처가 많아 다중공선성 위험이 있다.

```python
# feature_selection.py에 추가
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df, feature_cols, threshold=10):
    """VIF > 10인 피처는 다중공선성 위험 경고"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_cols
    vif_data["VIF"] = [variance_inflation_factor(df[feature_cols].values, i)
                       for i in range(len(feature_cols))]
    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        warnings.warn(f"다중공선성 위험 피처: {high_vif['feature'].tolist()}")
    return vif_data
```

### 9.3 이상치 처리 강화

현재 이상치 처리 로직에 대한 언급이 없다. 실제 OHLCV 데이터에는 다음 이상치가 발생한다.

- 공매도 재개·금지에 따른 거래량 급변
- 합병·분할·이전상장에 따른 가격 불연속
- 데이터 오류(High < Low 등)

```python
def validate_ohlcv(df):
    """OHLCV 기본 일관성 검증"""
    assert (df['High'] >= df['Low']).all(), "High < Low 오류"
    assert (df['High'] >= df['Close']).all(), "High < Close 오류"
    assert (df['Low'] <= df['Open']).all(), "Low > Open 오류"
    assert (df['Volume'] >= 0).all(), "음수 거래량 오류"
```

---

## 10. 우선순위별 개선 로드맵

### Phase 1 — 즉시 적용 (리스크 차단)

| 개선 항목 | 예상 소요 | 기대 효과 |
|---|---|---|
| 미국 외부 시장 피처 1거래일 lag 강제 | 0.5일 | 룩어헤드 바이어스 완전 차단 |
| `bfill()` 제거 → `ffill()` + `fillna(0)` | 0.5일 | 결측 처리 안전화 |
| OHLCV 기본 검증 함수 추가 | 1일 | 데이터 품질 보장 |
| 수급 데이터 유효성 검증 경고 추가 | 0.5일 | 수급 피처 신뢰성 확인 |
| 상한가·하한가 피처 추가 | 1일 | 한국 시장 특수 메커니즘 반영 |

### Phase 2 — 단기 적용 (예측력 향상)

| 개선 항목 | 예상 소요 | 기대 효과 |
|---|---|---|
| 볼린저 밴드 피처 추가 | 1일 | 변동성 채널 기반 신호 추가 |
| MA 정배열 피처 추가 | 0.5일 | 추세 구조 필터 추가 |
| RSI 다이버전스 피처 추가 | 1일 | 반전 신호 품질 향상 |
| 중국·홍콩 외부 시장 추가 | 1일 | 아시아 시장 영향 반영 |
| 고확신 수급 기준 상대화 | 0.5일 | 중소형주 수급 신호 정확도 향상 |
| 거래비용 조정 목표값 추가 | 1일 | 실질 수익성 기준 학습 |

### Phase 3 — 중기 적용 (정밀화)

| 개선 항목 | 예상 소요 | 기대 효과 |
|---|---|---|
| 공매도 피처 추가 | 2일 | 기관 방향성 뷰 반영 |
| 섹터 상대강도 피처 추가 | 2일 | 섹터 로테이션 포착 |
| 다중기간 목표값 앙상블 | 3일 | 단기 노이즈 감소 |
| VIF 기반 다중공선성 관리 | 1일 | 모델 안정성 향상 |
| 옵션 만기일·코스피200 이벤트 피처 | 2일 | 이벤트 드리븐 효과 반영 |
| 시장 국면 원-핫 인코딩 → 모델 입력 | 1일 | 국면별 패턴 학습 |

---

## 요약 및 결론

현재 `FEATURES_GUIDE.md`는 퀀트 투자 파이프라인의 기본 골격으로서 합리적으로 설계되어 있다. 그러나 **한국 시장 특수성 반영 부족**, **미국 외부 시장 피처의 룩어헤드 바이어스 위험**, **수급 데이터 공백 리스크**, **단일 목표값의 한계**는 모델 성능과 실전 적용 가능성에 직접적인 제약이 된다.

가장 즉각적으로 시행해야 할 조치는 **외부 시장 피처의 lag 보정**과 **`bfill()` 제거**이며, 이 두 가지만으로도 과거 백테스트의 실제 예측력 과대평가 문제를 상당 부분 해소할 수 있다.

> 추가 개선 사항을 적용할 때마다 `tests/test_pipeline_smoke.py`와 별도 워크포워드 검증을 반드시 병행한다.
