# FEATURES_GUIDE 통합 피처 개선 및 수정 제안서 v2

> **검토 관점**: 한국 주식시장(KOSPI·KOSDAQ) 퀀트 투자 및 실전 운용 관점  
> **기반 문서**: `FEATURES_GUIDE.md`, `FEATURES_IMPROVEMENT_PROPOSAL.md`, `FEATURES_GUIDE_REVIEW.md`  
> **목적**: 예측 정확도 향상, 룩어헤드 바이어스 차단, 한국 시장 특수성 반영, 실제 매매 성과와 모델 목적함수의 정합성 개선

---

## 0. 핵심 결론

현재 피처 설계는 가격·거래량·기술적 지표·외부 시장·수급·시장 국면·표시 전용 뉴스/공시를 명확히 분리한다는 점에서 기본 구조가 좋다. 특히 뉴스·공시를 모델 입력에서 제외하고 표시 문맥으로만 사용하는 정책은 재현성과 과적합 방지 측면에서 적절하다.

다만 실전 투자 모델로 사용하려면 다음 5가지를 우선 수정해야 한다.

| 우선순위 | 개선 영역 | 핵심 수정 | 이유 |
|---|---|---|---|
| P0 | 외부 시장 시차 | 미국 지수·VIX·SOX·미국 금리 등은 한국 시장 기준 `lag1` 모델 입력만 허용 | 동일 날짜 병합은 실전 관측 가능 시점과 어긋나 룩어헤드 바이어스 발생 가능 |
| P0 | 결측 처리 | 외부 피처의 `bfill()` 제거, `ffill()` 후 초기 결측은 별도 플래그와 안전값 처리 | `bfill()`은 미래 데이터를 과거로 역충전할 수 있음 |
| P0 | 목표값 | `target_log_return` 외에 거래비용 차감 수익률, 벤치마크 초과수익률, 다중 기간 목표값 추가 | 예측값과 실제 매매 가능 수익률 간 괴리 축소 |
| P0 | 피처 선택 | 원시 가격·원시 지수 레벨·원시 MA는 모델 입력 제외, 정규화·비율·수익률·z-score 중심 allowlist 적용 | 가격대·지수 레벨 차이로 인한 왜곡 방지 |
| P0 | 한국 시장 특수성 | 상한가·하한가, 공매도, 프로그램 매매, 신용잔고, 옵션만기, 섹터 로테이션 피처 추가 | 한국 시장의 구조적 수급·이벤트 효과 반영 |

---

## 1. 현재 설계 평가

### 1.1 강점

- `target_log_return`을 다음 거래일 로그수익률로 명시해 목표값 정의가 단순하고 재현 가능하다.
- 뉴스·공시 관련 열을 `DISPLAY_ONLY_CONTEXT_COLUMNS`로 분리해 모델 입력과 표시 문맥을 구분한 점은 매우 좋다.
- `src/features/feature_selection.py`가 모델 입력 열을 최종 결정하도록 설계되어 있어, 피처 생성과 피처 사용의 책임이 분리되어 있다.
- 외부 심볼 다운로드 실패 시 대체 심볼을 시도하고 coverage를 기록하는 설계는 운영 안정성에 유리하다.
- 가격, 기술적 지표, 외부 시장, 수급, 시장 국면, 투자 보조 플래그가 파일 단위로 분리되어 있어 개선 범위가 명확하다.

### 1.2 구조적 리스크

| 리스크 | 현재 상태 | 영향 |
|---|---|---|
| 외부시장 룩어헤드 | 외부 피처를 `Date` 기준 left join하고 내부 결측을 `ffill()` 후 `bfill()` | 백테스트 성과 과대평가 가능 |
| 시점 정책 불명확 | 장마감 후 예측인지, 익일 시가 진입인지, 장중 진입인지 문서상 분리 부족 | 실제 운용 시 피처 가용성 혼선 |
| 수급 데이터 공백 | `_fetch_flow()`가 자리표시자이면 수급 피처가 0으로 대체될 수 있음 | 수급 피처가 정보 없는 노이즈 또는 상수열이 될 수 있음 |
| 목표값 단일성 | 익일 종가 기준 로그수익률 중심 | 거래비용, 슬리피지, 보유기간, 리스크 미반영 |
| 피처 중복 | `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ma_N`, `close_to_ma_N` 등 유사 피처 다수 | 다중공선성 및 중요도 불안정 가능 |
| 한국시장 이벤트 부재 | 가격제한폭, 옵션만기, KOSPI200 리밸런싱, 공매도, 신용잔고 미반영 | 국내 주식 특유의 비선형 가격 형성 미포착 |

---

## 2. 최우선 수정: 외부 시장 피처의 시점 정합성

### 2.1 문제

현재 외부 시장 데이터는 날짜 기준으로 원본 데이터에 병합된다. 그러나 한국 주식시장과 미국 주식시장, 미국 선물, VIX, 미국 금리, 원/달러 환율은 관측 가능 시점이 다르다.

예를 들어 한국 시장 `Date=t` 장마감 후 신호를 생성한다면, 같은 `Date=t`의 미국 S&P 500, NASDAQ, SOX, VIX 종가는 아직 확정되지 않았을 수 있다. 따라서 같은 날짜의 미국 종가 수익률을 한국 종목의 `t -> t+1` 수익률 예측에 사용하면 실전에서는 사용할 수 없는 정보를 학습하는 문제가 생긴다.

### 2.2 권장 정책

| 외부 자산군 | 예시 | 모델 입력 정책 |
|---|---|---|
| 미국 현물 지수 | `^GSPC`, `^IXIC`, `^SOX` | 한국 시장 기준 `lag1_kr`만 허용 |
| 미국 변동성·금리 | `^VIX`, `^TNX` | `lag1_kr`만 허용 |
| 나스닥 선물 | `NQ=F` | 장중 사용 여부를 명시. 일봉 종가 기준이면 `lag1_kr` 권장 |
| 원/달러 환율 | `KRW=X`, `USDKRW=X` | 데이터 타임스탬프가 한국 장중 관측 가능하면 당일 사용 가능. 불명확하면 `lag1_kr` |
| 한국 지수 | `^KS11`, `^KQ11` | 장마감 후 신호라면 당일 수익률 사용 가능. 장중 예측이면 사용 금지 또는 지연 |
| 중국·홍콩 지수 | `000001.SS`, `^HSI` | 장마감 시각 기준으로 `available_at` 관리. 보수적으로 `lag1_kr` 우선 |

### 2.3 권장 피처명

| 원본 피처 | 모델 입력 권장 피처 | 비고 |
|---|---|---|
| `gspc_ret_1d` | `gspc_ret_1d_lag1_kr` | 미국 주식시장 피처는 원칙적으로 lag 입력 |
| `ixic_ret_1d` | `ixic_ret_1d_lag1_kr` | NASDAQ 영향 반영 |
| `sox_ret_1d` | `sox_ret_1d_lag1_kr` | 반도체 종목 핵심 외부 피처 |
| `vix_ret_1d` | `vix_ret_1d_lag1_kr` | 위험회피 국면 |
| `tnx_ret_1d` | `tnx_ret_1d_lag1_kr` | 금리 민감주 |
| `nq_f_ret_1d` | `nq_f_ret_1d_lag1_kr`, `nq_f_intraday_ret_kr` | 데이터 수집 시점에 따라 분리 |
| `krw_x_ret_1d` | `krw_x_ret_1d_lag1_kr` 또는 `krw_x_ret_intraday_kr` | 환율 데이터 가용성 검증 필요 |

### 2.4 구현 예시

```python
# src/features/external_features.py
US_LAG_REQUIRED_ALIASES = {"gspc", "ixic", "sox", "vix", "tnx"}
CONSERVATIVE_LAG_ALIASES = {"nq_f", "dxy", "gold", "wti", "hsi", "sse"}

for alias in external_aliases:
    ret_cols = [
        f"{alias}_ret_1d",
        f"{alias}_ret_5d",
        f"{alias}_vol_20",
    ]
    for col in ret_cols:
        if col in external.columns:
            if alias in US_LAG_REQUIRED_ALIASES or alias in CONSERVATIVE_LAG_ALIASES:
                external[f"{col}_lag1_kr"] = external[col].shift(1)
```

### 2.5 결측 처리 수정

```python
# 현재 위험한 방식
external = external.ffill().bfill()

# 권장 방식
external = external.sort_values("Date")
external = external.ffill()

for col in external_feature_cols:
    external[f"{col}_missing_flag"] = external[col].isna().astype(int)

# 초기 구간만 안전값 처리. 단, missing_flag를 함께 모델에 제공하거나 해당 행 제외 검토.
external[external_feature_cols] = external[external_feature_cols].fillna(0.0)
```

`bfill()`은 미래 데이터를 과거로 주입할 수 있으므로 학습·검증·실전 예측 모든 단계에서 기본 금지해야 한다.

---

## 3. 목표값 설계 개선

### 3.1 현재 목표값의 한계

현재 핵심 목표값은 다음 거래일 로그수익률 `target_log_return = log(Close_{t+1} / Close_t)`이다. 이는 연구 초기에는 적절하지만, 실전 매매 관점에서는 다음 한계가 있다.

1. 거래비용과 슬리피지를 반영하지 않는다.
2. 시장 전체 상승으로 인한 수익과 종목 고유 알파를 구분하지 않는다.
3. 종가 기준 예측과 실제 진입가 기준 수익률이 다를 수 있다.
4. 1일 수익률은 노이즈가 커서 과적합 가능성이 높다.
5. 동일 기대수익률이라도 변동성·유동성·하방위험이 다르면 투자 가치가 다르다.

### 3.2 신규 목표값 권장

| 목표값 | 계산 개념 | 목적 |
|---|---|---|
| `target_log_return_1d` | `log(Close_{t+1}/Close_t)` | 기존 목표값 유지 |
| `target_open_to_close_1d` | `log(Close_{t+1}/Open_{t+1})` | 익일 시가 진입·종가 청산 전략 검증 |
| `target_close_to_open_1d` | `log(Open_{t+1}/Close_t)` | 오버나이트 갭 예측 |
| `target_log_return_3d` | `log(Close_{t+3}/Close_t)` | 단기 노이즈 완화 |
| `target_log_return_5d` | `log(Close_{t+5}/Close_t)` | 스윙 관점 목표 |
| `target_excess_1d_ks11` | 종목 1일 수익률 - KOSPI 1일 수익률 | 시장 대비 알파 학습 |
| `target_excess_1d_sector` | 종목 1일 수익률 - 섹터 1일 수익률 | 섹터 효과 제거 |
| `target_net_return_1d` | `target_log_return_1d - estimated_cost` | 실질 매매 가능 수익률 |
| `target_strong_up` | 비용+변동성 임계값 이상 상승 | 노이즈성 상승 제거 |
| `target_strong_down` | 비용+변동성 임계값 이상 하락 | 하방 리스크 분류 |

### 3.3 거래비용 차감 목표값

```python
# 보수적 기본값 예시. 실제 수수료, 세금, 슬리피지는 계좌·종목 유동성별로 별도 관리.
base_commission_bps = 3       # 왕복 수수료 가정
sell_tax_bps = 18             # 한국 증권거래세 예시값은 시점별로 변동 가능하므로 설정화 필요
slippage_bps = estimated_slippage_bps
estimated_cost = (base_commission_bps + sell_tax_bps + slippage_bps) / 10_000

target_net_return_1d = target_log_return_1d - estimated_cost
```

### 3.4 분류 목표값 개선

단순 `target_up = target_log_return > 0`은 노이즈가 많다. 다음과 같이 중립 구간을 명시해야 한다.

```python
threshold = estimated_cost + vol_20 * 0.5

target_strong_up = target_log_return_1d > threshold
target_strong_down = target_log_return_1d < -threshold
target_neutral = ~(target_strong_up | target_strong_down)
```

---

## 4. 모델 입력 피처 선택 규칙 수정

### 4.1 문제

현재 `select_feature_columns()`는 접두사 기반으로 모델 입력을 선택한다. 이 경우 다음 피처가 의도치 않게 들어갈 수 있다.

- `ma_5`, `ma_20` 같은 원시 가격 레벨 기반 이동평균
- `{alias}_close` 같은 외부 지수 원시 종가
- lag 처리되지 않은 외부 수익률
- 스케일이 큰 원시 순매수 금액

원시 레벨값은 종목 가격대, 지수 레벨, 시계열 추세에 따라 모델이 잘못된 패턴을 학습할 수 있다.

### 4.2 제외 권장

| 제외 피처 | 이유 |
|---|---|
| `ma_{N}` | 종목 가격대 자체를 반영하므로 cross-section 모델에서 왜곡 가능 |
| `{alias}_close` | 외부 지수 레벨은 비정상 시계열이며 직접 입력 부적절 |
| `Close`, `Open`, `High`, `Low` | 가격 스케일 차이 큼 |
| `value_traded` 원시값 | 시가총액·종목 규모 편향 |
| `foreign_net_buy`, `institution_net_buy` 원시값 | 대형주 편향 |
| lag 미적용 미국 외부 피처 | 룩어헤드 위험 |

### 4.3 허용 권장

| 허용 피처 유형 | 예시 |
|---|---|
| 수익률 | `ret_1d`, `ret_5d`, `target 제외 과거 수익률` |
| 비율 | `close_to_ma_20`, `foreign_buy_ratio`, `turnover_ratio` |
| z-score | `foreign_net_buy_z20`, `volume_z20`, `value_traded_z60` |
| 횡단면 랭크 | `ret_5d_rank_daily`, `value_traded_rank_pct_daily` |
| lag 외부 피처 | `gspc_ret_1d_lag1_kr`, `sox_ret_1d_lag1_kr` |
| 국면 수치 피처 | `regime_trend_score`, `regime_vol_score`, `risk_off_score` |
| 유동성·비용 피처 | `estimated_slippage_bps`, `tradable_amount_ratio` |

### 4.4 권장 구현 방향

접두사 기반 선택보다 명시적 allowlist + blocklist 조합이 안전하다.

```python
MODEL_FEATURE_ALLOW_PATTERNS = [
    r"^ret_\\d+d$",
    r"^close_to_ma_\\d+$",
    r"^vol_\\d+$",
    r"^.*_z\\d+$",
    r"^.*_rank_pct_daily$",
    r"^.*_ret_\\d+d_lag1_kr$",
    r"^foreign_.*_ratio$",
    r"^institution_.*_ratio$",
    r"^smart_money_.*$",
    r"^bb_.*$",
    r"^adx_.*$",
    r"^ma_cross_.*$",
]

MODEL_FEATURE_DENY_PATTERNS = [
    r"^ma_\\d+$",
    r"^.*_close$",
    r"^(Open|High|Low|Close|Volume)$",
    r"^target_.*$",
    r"^.*news.*$",
    r"^.*disclosure.*$",
]
```

---

## 5. 가격·거래량 피처 개선

### 5.1 신규 가격 피처

| 피처 | 계산 개념 | 투자 의미 | 우선순위 |
|---|---|---|---|
| `close_to_vwap` | 종가 / VWAP - 1 | 기관 체결 기준 대비 매수·매도 강도 | P1 |
| `pivot`, `r1`, `s1` | 전일 고·저·종 기반 지지·저항 | 단기 지지·저항 레벨 | P2 |
| `close_to_pivot` | 종가 / 피벗 - 1 | 당일 종가의 지지·저항 대비 위치 | P2 |
| `gap_filled_flag` | 갭 발생 후 되돌림 여부 | 갭 모멘텀 신뢰도 | P1 |
| `gap_follow_through_flag` | 갭 방향과 당일 방향 일치 | 추세 지속성 | P1 |
| `overnight_return` | 시가 / 전일 종가 - 1 | 장전 수급·뉴스 반응 | P1 |
| `intraday_reversal` | 갭 방향과 종가 방향 반대 | 단기 과열·실망 매물 | P1 |

VWAP은 일봉 OHLCV만으로는 정확히 계산할 수 없다. 분봉 데이터가 없다면 `typical_price = (High + Low + Close) / 3` 기반의 근사 VWAP을 별도 이름으로 사용해야 한다.

```python
typical_price = (High + Low + Close) / 3
approx_vwap_proxy = typical_price
close_to_approx_vwap = Close / approx_vwap_proxy - 1
```

### 5.2 거래량·유동성 피처

| 피처 | 계산 개념 | 투자 의미 | 우선순위 |
|---|---|---|---|
| `vol_surge_flag` | `vol_ratio_20 > 2.0` | 비정상 거래량 폭발 | P1 |
| `vol_trend_5d` | 최근 5일 거래량 회귀 기울기 | 관심 증가·감소 추세 | P1 |
| `volume_z20` | 거래량 20일 z-score | 종목별 평소 대비 거래량 | P1 |
| `value_traded_z60` | 거래대금 60일 z-score | 거래대금 이벤트성 변화 | P1 |
| `price_vol_divergence` | 가격 상승 + 거래량 감소 | 추세 약화 경고 | P2 |
| `up_vol_ratio_20` | 상승일 평균 거래량 / 하락일 평균 거래량 | 매수 강도 | P2 |
| `turnover_ratio` | 거래량 / 상장주식수 | 회전율 | P1 |
| `tradable_amount_ratio` | 목표 주문금액 / 20일 평균 거래대금 | 실제 체결 가능성 | P0/P1 |
| `estimated_slippage_bps` | 거래대금·스프레드 기반 추정 | 수익률 현실화 | P0/P1 |

### 5.3 이동평균 구조 피처

현재 `close_to_ma_N`은 유효하지만, 실전에서는 이동평균 간 관계가 더 중요하다.

| 피처 | 계산 개념 | 투자 의미 |
|---|---|---|
| `ma_cross_5_20` | `ma_5 / ma_20 - 1` | 단기·중기 추세 차이 |
| `ma_cross_20_60` | `ma_20 / ma_60 - 1` | 중기·장기 추세 차이 |
| `ma_alignment_bull` | `ma_5 > ma_20 > ma_60` | 정배열 강세 구조 |
| `ma_alignment_bear` | `ma_5 < ma_20 < ma_60` | 역배열 약세 구조 |
| `ma_slope_20` | 20일 이동평균 기울기 | 추세 방향과 속도 |
| `ma_slope_accel` | 단기 MA 기울기 - 장기 MA 기울기 | 추세 가속도 |

---

## 6. 기술적 지표 개선

### 6.1 신규 지표 우선순위

| 피처 | 설명 | 사용 목적 | 우선순위 |
|---|---|---|---|
| `bb_pct` | 볼린저 밴드 내 종가 위치 | 과열·침체·평균회귀 | P1 |
| `bb_width` | 밴드 폭 / MA | 변동성 확장·수축 | P1 |
| `bb_squeeze_flag` | 밴드 폭이 장기 저점권 | 변동성 압축 후 돌파 후보 | P1 |
| `adx_14` | 추세 강도 | 추세장/횡보장 필터 | P1 |
| `plus_di_14`, `minus_di_14` | 방향성 지표 | 상승·하락 추세 방향 | P2 |
| `williams_r_14` | 단기 과매수·과매도 | 반전·단기 타이밍 | P2 |
| `donchian_high_20_breakout` | 20일 고가 돌파 | 추세 추종 | P2 |
| `momentum_accel_5_20` | `ret_5d - ret_20d` | 모멘텀 가속 | P1 |
| `realized_skew_20` | 20일 수익률 왜도 | 급락·급등 비대칭성 | P2 |
| `downside_vol_20` | 음의 수익률 변동성 | 하방 리스크 | P1 |

### 6.2 RSI 개선

단순 RSI 레벨보다 다이버전스와 회복 신호가 더 실전적이다.

```python
rsi_rebound_from_oversold = (rsi_14.shift(1) < 30) & (rsi_14 >= 30)
rsi_divergence_bull = (Close < Close.shift(5)) & (rsi_14 > rsi_14.shift(5))
rsi_divergence_bear = (Close > Close.shift(5)) & (rsi_14 < rsi_14.shift(5))
```

### 6.3 MACD 개선

`macd_hist`의 절대값보다 방향 전환과 기울기 변화가 중요하다.

```python
macd_hist_cross_up = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
macd_hist_cross_down = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
macd_hist_slope = macd_hist - macd_hist.shift(1)
macd_hist_accel = macd_hist_slope - macd_hist_slope.shift(1)
```

### 6.4 중복 관리

기술적 지표를 많이 추가할수록 유사 신호가 중복된다. 다음 기준으로 관리한다.

- `RSI`, `Stochastic`, `Williams %R`은 모두 오실레이터 계열이므로 모델 중요도와 상관관계를 함께 확인한다.
- `ATR`, `bb_width`, `vol_20`은 모두 변동성 계열이므로 VIF 또는 feature clustering으로 중복 제거를 검토한다.
- `MACD`, `ma_cross`, `ma_slope`는 추세 계열이므로 동일 방향 중복에 주의한다.

---

## 7. 외부 시장 피처 확장

### 7.1 추가 외부 심볼

| 추가 심볼 | 별칭 | 투자 의미 | 적용 우선순위 |
|---|---|---|---|
| `000001.SS` | `sse` | 중국 경기·소비·소재 민감주 영향 | P2 |
| `^HSI` | `hsi` | 홍콩·중국 리스크 및 외국인 자금 심리 | P2 |
| `CL=F` | `wti` | 정유·화학·항공·운송 섹터 영향 | P2 |
| `GC=F` | `gold` | 안전자산 선호, 리스크오프 | P2 |
| `DX-Y.NYB` | `dxy` | 글로벌 달러 강도 | P2 |
| `^RUT` | `rut` | 미국 중소형 위험선호 | P3 |
| `^MOVE` | `move` | 채권 변동성, 금리 리스크 | P3 |

### 7.2 파생 외부 피처

| 피처 | 계산 개념 | 목적 |
|---|---|---|
| `risk_off_score` | VIX 상승, 금 상승, 주가 하락, DXY 상승 조합 | 위험회피 국면 측정 |
| `semiconductor_tailwind_score` | SOX·NASDAQ·KRW 조합 | 반도체주 우호 환경 |
| `china_sensitive_tailwind` | SSE·HSI·KRW 조합 | 중국 민감주 환경 |
| `rate_sensitive_headwind` | TNX 상승 + 성장주 약세 | 성장주 할인율 부담 |
| `oil_sensitive_score` | WTI 급등락 | 정유·화학·항공 섹터 필터 |

---

## 8. 수급 피처 개선

### 8.1 수급 데이터 유효성 검증

현재 수급 열이 없으면 `0.0`으로 보정될 수 있다. 이는 모든 종목의 수급 신호가 상수열이 되는 문제를 만든다.

```python
def validate_flow_data(df):
    flow_cols = ["foreign_net_buy", "institution_net_buy"]
    for col in flow_cols:
        if col in df.columns:
            non_zero_ratio = (df[col] != 0).mean()
            if non_zero_ratio < 0.05:
                warnings.warn(
                    f"{col}의 95% 이상이 0입니다. placeholder 또는 수집 실패 가능성이 높습니다."
                )
```

### 8.2 스케일링 개선

고정 금액 기준은 대형주 편향을 만든다. 모든 수급 피처는 거래대금, 시가총액, 유동주식수, 종목별 rolling 분포 기준으로 정규화해야 한다.

| 기존 방식 | 문제 | 권장 방식 |
|---|---|---|
| `foreign_net_buy > 1000억` | 중소형주 신호 무시 | `foreign_net_buy / market_cap` 또는 z-score |
| 원시 순매수 금액 | 대형주 편향 | `foreign_buy_ratio`, `foreign_net_buy_z20` |
| 단일일 순매수 | 일회성 노이즈 | 3일·5일·20일 누적 및 지속성 |

### 8.3 신규 수급 피처

| 피처 | 계산 개념 | 투자 의미 | 우선순위 |
|---|---|---|---|
| `foreign_buy_to_mcap` | 외국인 순매수 / 시가총액 | 규모 중립 수급 강도 | P1 |
| `institution_buy_to_mcap` | 기관 순매수 / 시가총액 | 규모 중립 기관 수급 | P1 |
| `foreign_buy_persistence_5d` | 최근 5일 중 외국인 순매수 일수 | 지속성 | P1 |
| `institution_buy_persistence_5d` | 최근 5일 중 기관 순매수 일수 | 지속성 | P1 |
| `dual_accumulation_5d` | 외국인·기관 동시 순매수 지속 | 고품질 수급 | P1 |
| `retail_contrarian_signal` | 개인 대량 순매수 + 가격 하락 | 개인 물림 가능성 | P2 |
| `short_sell_ratio` | 공매도 거래량 / 총 거래량 | 하방 베팅 강도 | P2 |
| `short_balance_ratio` | 공매도 잔고 / 상장주식수 | 누적 숏 포지션 | P2 |
| `short_squeeze_flag` | 공매도 감소 + 가격 급등 | 숏커버링 가능성 | P2 |
| `program_buy_net` | 차익+비차익 프로그램 순매수 | 지수·선물 연계 수급 | P2 |
| `basis_spread` | KOSPI200 선물 - 현물 | 프로그램 매매 압력 | P2 |
| `margin_balance_change_5d` | 신용잔고 5일 변화율 | 개인 레버리지 과열 | P2 |
| `margin_balance_to_cap` | 신용잔고 / 시가총액 | 반대매매 리스크 | P2 |

### 8.4 고확신 수급 플래그 수정

```python
foreign_high_conviction_buy_flag = (
    (foreign_buy_to_mcap > 0.005) |
    (foreign_net_buy_z20 > 2.0) |
    (foreign_buy_ratio > foreign_buy_ratio.rolling(60).quantile(0.9))
)

institution_high_conviction_buy_flag = (
    (institution_buy_to_mcap > 0.005) |
    (institution_net_buy_z20 > 2.0) |
    (institution_buy_ratio > institution_buy_ratio.rolling(60).quantile(0.9))
)
```

---

## 9. 시장·섹터 상대 피처

### 9.1 시장 대비 초과성과

절대 수익률은 시장 베타를 많이 포함한다. 종목 고유 알파를 학습하려면 시장·섹터 대비 상대성과를 추가해야 한다.

| 피처 | 계산 개념 |
|---|---|
| `excess_ret_1d_ks11` | 종목 `ret_1d` - KOSPI `ks11_ret_1d` |
| `excess_ret_5d_ks11` | 종목 `ret_5d` - KOSPI `ks11_ret_5d` |
| `excess_ret_20d_ks11` | 종목 `ret_20d` - KOSPI `ks11_ret_20d` |
| `excess_ret_5d_sector` | 종목 5일 수익률 - 섹터 5일 수익률 |
| `sector_rel_strength_5d` | 섹터 5일 수익률 - KOSPI 5일 수익률 |
| `sector_rank_daily` | 날짜별 섹터 수익률 순위 |
| `sector_leadership_flag` | 섹터 순위 상위 3개 여부 |

### 9.2 베타·잔차 피처

| 피처 | 계산 개념 | 투자 의미 |
|---|---|---|
| `beta_60_ks11` | 최근 60일 종목 수익률의 KOSPI 베타 | 시장 민감도 |
| `residual_ret_1d` | 종목 수익률 - 베타×시장 수익률 | 종목 고유 움직임 |
| `residual_vol_20` | 잔차 수익률 20일 변동성 | 고유 리스크 |
| `idiosyncratic_momentum_20d` | 시장 효과 제거 후 20일 모멘텀 | 순수 알파 모멘텀 |

### 9.3 횡단면 랭킹 피처

같은 날짜 유니버스 내 상대 순위는 주식 선택 모델에서 매우 중요하다.

| 피처 | 계산 개념 |
|---|---|
| `ret_5d_rank_pct_daily` | 날짜별 5일 수익률 백분위 |
| `ret_20d_rank_pct_daily` | 날짜별 20일 수익률 백분위 |
| `vol_20_rank_pct_daily` | 날짜별 변동성 백분위 |
| `value_traded_rank_pct_daily` | 날짜별 거래대금 백분위 |
| `smart_money_strength_rank_pct_daily` | 날짜별 스마트머니 강도 백분위 |
| `close_to_52w_high_rank_pct_daily` | 날짜별 52주 고점 근접도 백분위 |

주의: 횡단면 랭킹은 해당 날짜의 전체 유니버스 데이터가 완성된 뒤 계산해야 한다. 장중 실시간 신호에 사용하려면 장중 유니버스 동기화 정책이 필요하다.

---

## 10. 시장 국면 라벨 개선

### 10.1 현재 한계

현재 `market_regime`은 `close_to_ma_20`과 `vol_20`의 단순 조합이며 문자열 라벨로 모델 입력에서 제외된다. 실전에서는 추세, 변동성, 유동성, 위험선호, 섹터 로테이션을 분리해 수치형 피처로 제공하는 편이 유용하다.

### 10.2 신규 국면 피처

| 피처 | 계산 개념 | 목적 |
|---|---|---|
| `regime_trend_score` | `close_to_ma_20`, `ma_slope_20`, `ks11_ret_20d` 조합 | 추세 강도 |
| `regime_vol_score` | `vol_20`의 rolling percentile | 변동성 국면 |
| `liquidity_regime_score` | 거래대금·거래량의 시장 전체 수준 | 유동성 환경 |
| `risk_off_score` | VIX, 금, DXY, 지수 하락 조합 | 위험회피 환경 |
| `growth_regime_flag` | SOX/NASDAQ 강세 + 금리 안정 | 성장주 우호 환경 |
| `defensive_regime_flag` | VIX 상승 + 금 상승 + 지수 약세 | 방어주 우호 환경 |
| `regime_uptrend_high_vol` | 기존 문자열 라벨 원-핫 | 모델 입력 후보 |

### 10.3 시장별 추세 임계값

현재 `close_to_ma_20 > 0.01`은 모든 종목에 동일 적용된다. KOSDAQ과 중소형주는 변동성이 커서 임계값을 달리해야 한다.

| 시장/종목군 | 권장 추세 임계값 |
|---|---:|
| KOSPI 대형주 | ±0.015 |
| KOSPI 중형주 | ±0.020 |
| KOSDAQ | ±0.030 |
| 고변동성 테마주 | ±0.040 이상 검토 |

```python
trend_threshold_by_market = {
    "KOSPI_LARGE": 0.015,
    "KOSPI_MID": 0.020,
    "KOSDAQ": 0.030,
}
```

---

## 11. 한국 시장 특화 피처

### 11.1 가격 제한폭 피처

한국 주식은 일일 가격 제한폭이 있어, 상한가·하한가 근접 종목은 일반 종목과 다른 가격 형성 메커니즘을 보인다.

| 피처 | 계산 개념 | 투자 의미 | 우선순위 |
|---|---|---|---|
| `near_upper_limit` | `daily_return > 0.27` | 상한가 근접 | P0/P1 |
| `upper_limit_flag` | `daily_return >= 0.295` | 상한가 달성 | P0/P1 |
| `near_lower_limit` | `daily_return < -0.27` | 하한가 근접 | P0/P1 |
| `lower_limit_flag` | `daily_return <= -0.295` | 하한가 달성 | P0/P1 |
| `consecutive_upper_limit` | 연속 상한가 횟수 | 테마 과열·유동성 잠김 | P1 |
| `limit_unlock_risk` | 상한가 이후 거래량 급증·음봉 | 물량 출회 위험 | P2 |

### 11.2 배당·권리락·기업 이벤트

| 피처 | 투자 의미 |
|---|---|
| `ex_dividend_flag` | 배당락에 따른 인위적 가격 하락 보정 |
| `rights_offering_flag` | 유상증자 권리락 이벤트 |
| `stock_split_flag` | 액면분할·병합에 따른 가격 불연속 |
| `fiscal_year_end_flag` | 결산·윈도드레싱·기관 리밸런싱 가능성 |
| `trading_halt_resume_flag` | 거래정지 해제 이후 변동성 확대 |
| `management_issue_flag` | 관리종목·투자주의·투자경고 등 리스크 |

### 11.3 KOSPI200·패시브 리밸런싱

| 피처 | 투자 의미 |
|---|---|
| `kospi200_member_flag` | 패시브 자금 수요 대상 여부 |
| `kospi200_rebalancing_flag` | 6월·12월 정기변경 시즌 효과 |
| `index_inclusion_candidate_score` | 편입 후보 추정 점수 |
| `etf_tracking_demand` | 지수 내 비중 변화에 따른 패시브 매수·매도 수요 |

### 11.4 옵션 만기일 효과

| 피처 | 계산 개념 |
|---|---|
| `days_to_expiry` | 옵션 만기일까지 남은 거래일 수 |
| `expiry_week_flag` | 만기일까지 5거래일 이내 여부 |
| `triple_witch_flag` | 3·6·9·12월 선물·옵션 동시만기 주간 |
| `program_pressure_expiry` | 만기주 프로그램 순매수 압력 |

### 11.5 섹터 로테이션

한국 시장에서는 반도체, 이차전지, 바이오, 금융, 자동차, 조선, 방산 등 섹터 순환매가 뚜렷하다.

| 피처 | 계산 개념 |
|---|---|
| `sector_rel_strength_5d` | 섹터 5일 수익률 - KOSPI 5일 수익률 |
| `sector_rel_strength_20d` | 섹터 20일 수익률 - KOSPI 20일 수익률 |
| `sector_rank_in_universe` | 날짜별 섹터 수익률 순위 |
| `sector_leadership_flag` | 상위 3개 섹터 여부 |
| `theme_crowding_score` | 동일 테마 종목 동반 급등·거래량 폭증 정도 |

---

## 12. 뉴스·공시 정책

### 12.1 현행 정책 유지 권장

뉴스·공시는 현재 표시 전용으로 분리되어 있다. 이 정책은 기본적으로 유지하는 것이 좋다. 뉴스·공시는 다음 문제가 있기 때문이다.

- 수집 시점과 발행 시점의 차이로 룩어헤드가 생기기 쉽다.
- 자연어 감성 점수는 모델·프롬프트·데이터 공급자 변경에 따라 재현성이 낮을 수 있다.
- 공시 정정, 장중 공시, 장마감 후 공시 등 가용 시점 통제가 어렵다.

### 12.2 예외적 연구 방향

뉴스·공시를 모델 입력으로 연구하려면 기본 피처 파이프라인이 아니라 별도 실험 플래그에서만 허용한다.

| 실험 피처 | 조건 |
|---|---|
| `event_known_before_signal_flag` | 신호 생성 시점 이전에 확인된 이벤트만 1 |
| `disclosure_after_close_flag` | 장마감 후 공시는 익일 이후에만 사용 |
| `earnings_surprise_lagged` | 실적 발표 확정 시점 기준 lag 적용 |
| `news_count_lag1` | 수집·발행 시점 검증 후 lag 적용 |

`DISPLAY_ONLY_CONTEXT_COLUMNS` 경계를 깨는 경우에는 별도 테스트를 반드시 추가한다.

---

## 13. 투자 신호 산출 방식 개선

### 13.1 현재 구조

현재 매수·매도·관망 판단은 최종 `predicted_return`을 기준으로 한다. 이 구조는 간결하지만, 실전에서는 기대수익률만으로는 부족하다.

### 13.2 권장 구조

최종 신호는 다음 요소를 결합해야 한다.

```text
expected_alpha
- estimated_transaction_cost
- estimated_slippage
- liquidity_penalty
- risk_penalty
+ signal_quality_bonus
= investable_score
```

| 구성 요소 | 예시 |
|---|---|
| 기대 알파 | `predicted_excess_return`, `predicted_net_return` |
| 비용 | 수수료, 세금, 슬리피지 |
| 유동성 패널티 | 거래대금 부족, 상한가 잠김, 체결 가능 금액 부족 |
| 리스크 패널티 | 고변동성, 하방 변동성, 이벤트 리스크, 신용잔고 과열 |
| 품질 보너스 | 수급 지속성, 섹터 리더십, 외부시장 우호, 정배열 |

### 13.3 신호 등급 예시

| 등급 | 조건 예시 | 해석 |
|---|---|---|
| `STRONG_BUY` | 비용 차감 기대수익률 상위 5%, 유동성 충분, 수급·추세 동시 양호 | 적극 매수 후보 |
| `BUY` | 비용 차감 기대수익률 양수, 리스크 허용 | 일반 매수 후보 |
| `WATCH` | 기대수익률은 양호하나 유동성·국면·이벤트 리스크 존재 | 관찰 |
| `HOLD` | 기대수익률과 비용이 유사 | 관망 |
| `SELL_AVOID` | 기대수익률 음수 또는 하방 리스크 큼 | 회피·매도 |

---

## 14. 데이터 품질 및 검증

### 14.1 OHLCV 검증

```python
def validate_ohlcv(df):
    errors = []
    errors.append((df["High"] < df["Low"], "High < Low"))
    errors.append((df["High"] < df["Close"], "High < Close"))
    errors.append((df["Low"] > df["Open"], "Low > Open"))
    errors.append((df["Volume"] < 0, "Negative Volume"))

    for mask, message in errors:
        if mask.any():
            raise ValueError(f"OHLCV validation failed: {message}")
```

추가로 액면분할, 병합, 거래정지, 이전상장, 상장폐지, 권리락으로 인한 가격 불연속을 별도 이벤트로 처리해야 한다.

### 14.2 룩어헤드 바이어스 체크리스트

| 항목 | 테스트 방법 |
|---|---|
| 외부시장 lag | 미국 지수 원본 피처가 모델 입력에 포함되지 않는지 검사 |
| `bfill()` 금지 | 외부 피처 생성 코드에서 backward fill 사용 여부 검사 |
| 목표값 제외 | `target_` 접두사 컬럼이 입력 피처에 없는지 검사 |
| 뉴스·공시 제외 | `DISPLAY_ONLY_CONTEXT_COLUMNS`가 입력 피처에 없는지 검사 |
| 종목별 rolling | rolling, shift, pct_change가 `Symbol` groupby 내부에서 수행되는지 검사 |
| 마지막 행 제거 | `target_log_return` 결측 행이 학습 데이터에서 제거되는지 검사 |
| 당일 순위 계산 | 동일 날짜 전체 유니버스 기준 계산 후 실전 시점에 사용 가능한지 검사 |

### 14.3 다중공선성 관리

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor


def check_vif(df, feature_cols, threshold=10):
    x = df[feature_cols].dropna()
    vif_data = []
    for i, col in enumerate(feature_cols):
        vif = variance_inflation_factor(x.values, i)
        vif_data.append({"feature": col, "vif": vif})
    result = pd.DataFrame(vif_data)
    return result[result["vif"] > threshold]
```

VIF가 높다고 무조건 제거할 필요는 없지만, 선형 모델·로지스틱 모델·해석 가능한 모델에서는 중요하게 관리해야 한다. 트리 계열 모델에서도 유사 피처가 많으면 중요도 해석이 불안정해질 수 있다.

### 14.4 워크포워드 검증

무작위 train/test split은 시계열 투자 모델에 부적절하다. 다음 프로토콜을 권장한다.

```text
Train: 2018-01-01 ~ 2021-12-31
Validation: 2022-01-01 ~ 2022-12-31
Test: 2023-01-01 ~ 2023-12-31
Forward test: 2024-01-01 ~ 현재
```

평가 지표는 RMSE나 방향 정확도만으로 부족하다.

| 평가 영역 | 권장 지표 |
|---|---|
| 예측 정확도 | IC, Rank IC, direction accuracy, calibration |
| 투자 성과 | CAGR, Sharpe, Sortino, MDD, hit ratio |
| 실행 가능성 | turnover, average holding period, capacity, slippage sensitivity |
| 안정성 | 연도별 성과, 국면별 성과, 섹터별 성과, 피처 중요도 안정성 |

---

## 15. 파일별 수정 권장사항

| 파일 | 수정 내용 | 우선순위 |
|---|---|---|
| `src/features/external_features.py` | `bfill()` 제거, `lag1_kr` 생성, `available_at`, `stale_flag`, fallback flag 추가 | P0 |
| `src/features/price_features.py` | 상한가·하한가, gap follow-through, MA cross, volume z-score, target 다중화 | P0/P1 |
| `src/features/technical_indicators.py` | Bollinger Band, ADX, Williams %R, Donchian, downside volatility 추가 | P1/P2 |
| `src/features/investment_signals.py` | 비용·유동성·리스크 패널티 기반 investable score 추가 | P1 |
| `src/features/regime_features.py` | 문자열 regime 외 수치형 regime score 및 원-핫 후보 생성 | P1 |
| `src/features/feature_selection.py` | allowlist/denylist 정교화, 원시 레벨값 제외, lag 외부 피처만 허용 | P0 |
| `src/config/settings.py` | 거래비용, lag 정책, 시장별 threshold, 신규 외부 심볼 설정 추가 | P0/P1 |
| `tests/` | 누수, 결측, 피처 선택, target 제외, display-only 경계 테스트 추가 | P0 |

---

## 16. 권장 설정 구조 예시

```json
{
  "feature": {
    "lookback_windows": [1, 2, 3, 5, 10, 20, 60],
    "moving_average_windows": [5, 10, 20, 60, 120],
    "volatility_windows": [5, 20, 60],
    "enable_bollinger": true,
    "enable_adx": true,
    "enable_market_specific_limits": true,
    "trend_threshold_by_market": {
      "KOSPI_LARGE": 0.015,
      "KOSPI_MID": 0.020,
      "KOSDAQ": 0.030
    }
  },
  "external": {
    "enabled": true,
    "disable_bfill": true,
    "default_lag_policy": "conservative_lag1_kr",
    "market_symbols": [
      "^KS11", "^KQ11", "^GSPC", "^IXIC", "NQ=F", "^SOX", "^VIX",
      "KRW=X", "^TNX", "000001.SS", "^HSI", "CL=F", "GC=F", "DX-Y.NYB"
    ],
    "lag_required_aliases": ["gspc", "ixic", "sox", "vix", "tnx"],
    "fallback_flag_enabled": true
  },
  "trading": {
    "signal_timing": "after_close_for_next_open",
    "base_commission_bps": 3,
    "sell_tax_bps": 18,
    "default_slippage_bps": 10,
    "max_order_to_adv20_ratio": 0.05,
    "min_value_traded_20d": 1000000000
  },
  "target": {
    "primary": "target_net_excess_return_1d",
    "horizons": [1, 3, 5, 10],
    "include_open_to_close": true,
    "include_excess_return": true,
    "classification_threshold_mode": "cost_plus_volatility"
  }
}
```

---

## 17. 우선순위별 로드맵

### Phase 1 — 즉시 적용: 누수 차단 및 입력 정리

| 작업 | 기대 효과 |
|---|---|
| 외부 피처 `bfill()` 제거 | 미래값 역충전 제거 |
| 미국 외부시장 `lag1_kr` 강제 | 룩어헤드 바이어스 차단 |
| lag 미적용 외부 피처 모델 입력 차단 | 실전 시점 정합성 강화 |
| 원시 가격·원시 지수 레벨 입력 제외 | 스케일 왜곡 완화 |
| OHLCV 검증 함수 추가 | 데이터 오류 조기 발견 |
| 수급 데이터 유효성 경고 추가 | placeholder 수급 데이터 차단 |
| 상한가·하한가 피처 추가 | 한국 시장 가격제한 효과 반영 |

### Phase 2 — 단기 적용: 실전 매매 가능성 반영

| 작업 | 기대 효과 |
|---|---|
| 거래비용 차감 목표값 추가 | 실제 수익률과 모델 목적 정합화 |
| 유동성·슬리피지 피처 추가 | 체결 불가능 종목 과대평가 방지 |
| MA cross·정배열 피처 추가 | 추세 구조 반영 |
| Bollinger Band·ADX 추가 | 변동성 압축·추세 강도 반영 |
| 외국인·기관 고확신 기준 상대화 | 중소형주 수급 신호 개선 |
| 거래량 z-score·거래대금 안정성 피처 추가 | 이벤트성 관심 포착 |

### Phase 3 — 중기 적용: 알파 품질 개선

| 작업 | 기대 효과 |
|---|---|
| 시장·섹터 대비 초과성과 피처 추가 | 베타와 알파 분리 |
| 횡단면 rank percentile 피처 추가 | 종목 선택력 향상 |
| 공매도·신용잔고·프로그램 매매 피처 추가 | 국내 수급 구조 반영 |
| 섹터 로테이션 피처 추가 | 순환매 포착 |
| 다중 기간 목표값 앙상블 | 1일 노이즈 완화 |
| 시장 국면 수치형 피처 및 원-핫 추가 | 국면별 패턴 학습 |

### Phase 4 — 운영 고도화

| 작업 | 기대 효과 |
|---|---|
| 피처 registry 도입 | 피처별 시점·소스·lag·입력 여부 추적 |
| 워크포워드 자동 리포트 | 성과 안정성 검증 |
| 피처 중요도 드리프트 모니터링 | 모델 열화 감지 |
| capacity·slippage 민감도 리포트 | 실전 운용 가능 금액 추정 |
| 뉴스·공시 별도 실험 파이프라인 | 정성 이벤트 알파 검증 |

---

## 18. 최종 권고

가장 먼저 수정해야 할 것은 신규 지표 추가가 아니라 **데이터 시점 정합성**이다. 특히 외부 시장 피처의 `bfill()` 제거와 미국 시장 피처의 `lag1_kr` 강제는 백테스트 신뢰도를 지키기 위한 필수 조치다.

그 다음으로 목표값을 실제 매매 기준으로 확장해야 한다. 단순 익일 로그수익률 예측은 연구용으로는 충분하지만, 투자 의사결정에서는 거래비용, 슬리피지, 유동성, 시장 대비 초과수익률을 반영한 `target_net_excess_return` 계열이 더 적합하다.

마지막으로 한국 시장 특화 피처를 단계적으로 추가한다. 상한가·하한가, 공매도, 프로그램 매매, 신용잔고, 옵션만기, KOSPI200 리밸런싱, 섹터 로테이션은 국내 주식 모델에서 구조적 알파 또는 리스크 필터로 작동할 가능성이 높다.

**권장 적용 순서**는 다음과 같다.

1. 외부 피처 lag 및 결측 처리 수정
2. 피처 선택 allowlist/denylist 정리
3. 거래비용·유동성 반영 목표값 추가
4. 상한가·하한가 및 데이터 품질 검증 추가
5. 수급 스케일링과 고확신 기준 상대화
6. Bollinger, ADX, MA 구조, 거래량 z-score 추가
7. 시장·섹터 상대성과와 횡단면 랭킹 추가
8. 공매도·신용잔고·프로그램 매매·옵션만기 이벤트 확장

이 순서를 따르면 피처 수를 무리하게 늘리기 전에, 먼저 백테스트의 신뢰성과 실전 매매 가능성을 확보할 수 있다.
