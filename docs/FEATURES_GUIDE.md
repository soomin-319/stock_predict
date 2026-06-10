# 피처 구성 및 처리 가이드

## 1. 목적과 핵심 원칙

이 문서는 예측 파이프라인에서 생성·사용하는 가격, 기술적 지표, 외부 시장,
투자자 수급 및 보조 신호 피처를 정리한다.

- 모델의 목표값은 **다음 거래일 로그수익률**인 `target_log_return`이다.
- 매수·매도·관망 판단은 최종 `predicted_return`을 기준으로 한다.
- 뉴스와 공시는 표시·검토용 문맥이다. 모델 입력, 기대수익률 순위 및 자동
  신호 결정에 사용하지 않는다.
- 피처는 종목별 시계열을 기준으로 계산하며, 모델 입력 여부는
  `src/features/feature_selection.py`가 최종 결정한다.

## 2. 피처 처리 흐름

`src/pipeline.py`의 `_build_pipeline_feature_matrix()`는 다음 순서로 처리한다.

1. `build_features()`: 가격·거래량·기술적 지표·수급 파생 피처 및 목표값 생성
2. `add_external_market_features_with_coverage()`: 외부 시장 피처 병합
3. `annotate_market_regime()`: 시장 국면 표시용 라벨 생성
4. `add_investment_signal_features()`: 투자 판단 보조 플래그 생성
5. 목표값이 없는 마지막 행 제거
6. `select_feature_columns()`: 실제 모델 입력 열 선택

주요 구현 파일:

| 영역 | 파일 |
|---|---|
| 가격·수급 피처와 목표값 | `src/features/price_features.py` |
| 기술적 지표 계산 함수 | `src/features/technical_indicators.py` |
| 외부 시장 피처 | `src/features/external_features.py` |
| 시장 국면 라벨 | `src/features/regime_features.py` |
| 투자 판단 보조 피처 | `src/features/investment_signals.py` |
| 모델 입력 열 선택 | `src/features/feature_selection.py` |
| 기본 피처 설정 | `src/config/settings.py` |

## 3. 입력 데이터

기본 입력은 종목별 일봉 OHLCV이다.

| 필수 열 | 의미 |
|---|---|
| `Date` | 거래일 |
| `Symbol` | 종목 식별자 |
| `Open`, `High`, `Low`, `Close` | 시가, 고가, 저가, 종가 |
| `Volume` | 거래량 |

선택 입력:

- `foreign_net_buy`: 외국인 순매수 금액
- `institution_net_buy`: 기관 순매수 금액
- 뉴스·공시 관련 열: 표시 문맥 생성용이며 모델에서는 제외

선택 입력이 없으면 대부분 `0.0`으로 보정된다. 현재 내장 수급 수집 함수
`_fetch_flow()`는 실제 데이터를 가져오지 않는 자리표시자이므로, 수급 피처를
활용하려면 입력 CSV 또는 별도 수집 경로에서 수급 열을 제공해야 한다.

## 4. 가격·거래량 피처

### 4.1 수익률과 일중 움직임

| 피처 | 계산 개념 |
|---|---|
| `log_return` | `log(Close_t / Close_{t-1})`; 변동성 계산용 중간 열이며 모델 입력에서는 제외 |
| `daily_return` | 전일 종가 대비 당일 종가 수익률 |
| `gap_return` | 전일 종가 대비 당일 시가 갭 |
| `intraday_return` | 당일 시가 대비 종가 수익률 |
| `range_pct` | `(High - Low) / Close` |
| `ret_{N}d` | N거래일 종가 수익률 |

기본 수익률 기간 `N`: `1, 2, 3, 5, 10, 20, 60`

### 4.2 이동평균과 추세

| 피처 | 계산 개념 |
|---|---|
| `ma_{N}` | N일 단순이동평균 |
| `close_to_ma_{N}` | `Close / ma_N - 1` |
| `close_to_52w_high` | 종가 / 최근 252거래일 최고 종가 |
| `near_52w_high_flag` | 52주 최고가의 95% 이상 여부 |
| `breakout_52w_flag` | 직전일까지의 252일 최고가 돌파 여부 |

기본 이동평균 기간 `N`: `5, 10, 20, 60, 120`

52주 최고가 계산은 최소 20개 관측치부터 값을 만들므로, 초기 구간의
`close_to_52w_high`는 엄밀한 52주 비교가 아닐 수 있다.

### 4.3 변동성과 거래량

| 피처 | 계산 개념 |
|---|---|
| `vol_{N}` | 일별 로그수익률의 N일 표준편차 |
| `vol_ratio_20` | 당일 거래량 / 20일 평균 거래량 |
| `value_traded` | `Close * Volume` |
| `turnover_rank_daily` | 같은 날짜 내 거래대금 내림차순 순위 |
| `is_top_turnover_3` | 거래대금 순위 3위 이내 |
| `is_top_turnover_10` | 거래대금 순위 10위 이내 |
| `is_top_turnover_15` | 설정된 상위 거래대금 순위 이내; 기본 15위 |

기본 변동성 기간 `N`: `5, 20, 60`

## 5. 기술적 지표 피처

| 피처 | 설명 |
|---|---|
| `rsi_14` | RSI. 기본 기간 14일 |
| `rsi_pullback_buy_flag` | RSI가 30~35 구간인지 여부 |
| `rsi_buy_watch_flag` | 설정값 기준 RSI 매수 관찰 구간 여부 |
| `rsi_overbought_sell_flag` | RSI가 기본 70 이상인지 여부 |
| `macd` | EMA(12) - EMA(26) |
| `macd_signal` | MACD의 EMA(9) |
| `macd_hist` | MACD - Signal |
| `atr_14` | True Range의 14일 평균 |
| `stoch_k`, `stoch_d` | 기본 14일 Stochastic K와 3일 평균 D |
| `cci_20` | 기본 20일 Commodity Channel Index |
| `obv` | 종가 방향을 반영한 누적 거래량 |
| `obv_change_5d` | OBV의 5일 변화율 |

RSI, CCI, Stochastic 기간은 `FeatureConfig`에서 변경할 수 있다.

## 6. 외부 시장 피처

외부 시장 데이터는 `yfinance`에서 종가를 받아 날짜 기준으로 원본 데이터에
병합한다. 각 외부 심볼마다 다음 열을 만든다.

- `{alias}_close`: 외부 자산 종가
- `{alias}_ret_1d`: 1일 수익률
- `{alias}_ret_5d`: 5일 수익률
- `{alias}_vol_20`: 1일 수익률의 20일 표준편차

### 6.1 기본 외부 심볼과 별칭

| 요청 심볼 | 별칭 | 의미 | 다운로드 대체 후보 |
|---|---|---|---|
| `^KS11` | `ks11` | KOSPI | `EWY` |
| `^KQ11` | `kq11` | KOSDAQ | `KORU` |
| `^GSPC` | `gspc` | S&P 500 | `SPY` |
| `^IXIC` | `ixic` | NASDAQ Composite | `QQQ` |
| `NQ=F` | `nq_f` | Nasdaq 100 선물 | `QQQ` |
| `^SOX` | `sox` | 필라델피아 반도체 지수 | `SOXX` |
| `^VIX` | `vix` | 변동성 지수 | `VIXY` |
| `KRW=X` | `krw_x` | 원/달러 환율 | `USDKRW=X` |
| `^TNX` | `tnx` | 미국 10년물 금리 | `IEF` |

원 심볼 다운로드가 실패하면 대체 후보를 시도한다. 모든 후보가 실패한
심볼은 건너뛰며 파이프라인 실행은 계속된다. 요청·성공·실패·대체 사용
건수는 파이프라인 리포트의 외부 피처 coverage에 기록된다.

### 6.2 병합과 결측 처리 주의사항

- 외부 데이터는 `Date`로 left join된다.
- 외부 프레임 내부 결측값은 현재 `ffill()` 후 `bfill()`된다.
- 서로 다른 시장의 휴장일과 시차가 별도로 보정되거나 지연 적용되지는 않는다.
- 따라서 실전 연구에서는 한국 시장 시점에 실제 관측 가능했던 값인지 확인하고,
  필요하면 외부 피처를 1거래일 지연해야 한다.
- `--disable-external`을 사용하면 외부 시장 피처를 생성하지 않는다.

## 7. 수급 및 투자 판단 보조 피처

### 7.1 외국인·기관 수급

| 피처 | 설명 |
|---|---|
| `foreign_buy_signal` | 외국인 순매수 > 0 |
| `institution_buy_signal` | 기관 순매수 > 0 |
| `smart_money_buy_signal` | 외국인 + 기관 순매수 합계 > 0 |
| `foreign_buy_ratio` | 외국인 순매수 / 거래대금 |
| `institution_buy_ratio` | 기관 순매수 / 거래대금 |
| `smart_money_strength` | 외국인·기관 순매수 합계 / 거래대금 |
| `foreign_net_buy_z20` | 외국인 순매수의 종목별 20일 z-score |
| `institution_net_buy_z20` | 기관 순매수의 종목별 20일 z-score |
| `foreign_net_buy_3d`, `foreign_net_buy_5d` | 외국인 순매수 3일·5일 합계 |
| `institution_net_buy_3d`, `institution_net_buy_5d` | 기관 순매수 3일·5일 합계 |
| `foreign_high_conviction_buy_flag` | 외국인 순매수가 기본 1,000억 원 이상 |
| `institution_high_conviction_buy_flag` | 기관 순매수가 기본 1,000억 원 이상 |
| `dual_high_conviction_buy_flag` | 외국인과 기관이 모두 고확신 순매수 조건 충족 |

### 7.2 시장 리더와 외부 시장 보조 플래그

| 피처 | 설명 |
|---|---|
| `leader_confirmation_flag` | 거래대금 상위 종목들의 동반 상승 조건 충족 여부 |
| `distance_to_52w_high` | `max(1 - close_to_52w_high, 0)` |
| `nasdaq_tailwind_flag` | `nq_f_ret_1d`가 기본 +1% 이상 |
| `nasdaq_headwind_flag` | `nq_f_ret_1d`가 기본 -1% 이하 |

`leader_confirmation_flag`는 `add_investment_signal_features()` 단계에서 설정값에
따라 다시 계산된다. 기본값은 거래대금 상위 3개 중 리더가 상승하고, 상승한
종목이 2개 이상인 경우다.

## 8. 시장 국면 라벨

`annotate_market_regime()`는 다음 조건으로 `market_regime` 문자열을 만든다.

- 추세:
  - `close_to_ma_20 > 0.01`: `uptrend`
  - `close_to_ma_20 < -0.01`: `downtrend`
  - 그 외: `sideways`
- 변동성:
  - `vol_20`이 전체 데이터의 75% 분위수 초과: `high_vol`
  - 그 외: `low_vol`

예: `uptrend_high_vol`, `sideways_low_vol`

현재 `market_regime`은 문자열 라벨이며 `select_feature_columns()`의 모델 입력
목록에는 포함되지 않는다.

## 9. 뉴스·공시: 표시 전용 피처

다음 열은 데이터프레임에 존재할 수 있지만 모델 입력에서 명시적으로 제외된다.

- `disclosure_score`
- `news_sentiment`
- `news_relevance_score`
- `news_impact_score`
- `news_article_count`
- `news_positive_signal`
- `news_negative_signal`
- `news_same_day_signal`
- `disclosure_same_day_signal`
- `investor_event_score`

`investor_event_score`는 거래대금, 공시, 긍정 뉴스, 수급, 52주 최고가 근접을
조합한 문맥 점수이지만 뉴스·공시 입력을 포함하므로 표시 전용이다.

`DISPLAY_ONLY_CONTEXT_COLUMNS`와 관련 테스트가 이 원칙을 강제한다. 뉴스·공시
값을 바꾸어도 선택된 모델 피처 값은 달라지지 않아야 한다.

## 10. 목표값

| 열 | 계산 |
|---|---|
| `target_log_return` | `log(Close_{t+1} / Close_t)` |
| `target_up` | `target_log_return > 0` |
| `target_close` | `Close_t * exp(target_log_return)` |

목표값은 모델 입력 피처에 포함되지 않는다. 다음 거래일 종가가 없는 종목별
마지막 행은 학습·검증용 피처 행에서 제거된다.

## 11. 실제 모델 피처 선택 규칙

`select_feature_columns()`는 다음 중 하나에 해당하는 열만 모델 입력으로 선택한다.

1. 허용된 접두사로 시작하는 열:
   `ret_`, `ma_`, `close_to_ma_`, `vol_`, `ks`, `kq`, `gspc`, `ixic`,
   `nq_f`, `sox`, `vix`, `krw`, `tnx`
2. `MODEL_FEATURE_COLUMN_BASE`에 명시된 열
3. 단, `DISPLAY_ONLY_CONTEXT_COLUMNS`에 속하면 항상 제외

새 피처를 만들기만 해서는 모델에 자동 사용되지 않을 수 있다. 신규 피처 추가 시
`feature_selection.py`의 허용 목록 또는 접두사를 확인하고, 뉴스·공시 표시 전용
경계를 침범하지 않는 테스트를 추가해야 한다.

## 12. 기본 설정 변경 예시

`--config-json`으로 다음과 같이 기간과 외부 심볼을 변경할 수 있다.

```json
{
  "feature": {
    "lookback_windows": [1, 5, 20],
    "moving_average_windows": [5, 20, 60],
    "volatility_windows": [5, 20],
    "rsi_period": 14,
    "cci_period": 20,
    "stochastic_period": 14
  },
  "external": {
    "enabled": true,
    "market_symbols": ["^KS11", "^GSPC", "NQ=F", "^VIX", "KRW=X"]
  }
}
```

## 13. 피처 추가·변경 체크리스트

1. 해당 시점에 실제 관측 가능한 데이터만 사용한다.
2. 종목별 rolling, shift, pct-change 계산인지 확인한다.
3. 외부 시장 시차와 휴장일을 고려한다.
4. 결측값 보정이 미래값을 유입하지 않는지 확인한다.
5. 모델 사용 피처라면 `feature_selection.py`에 선택 규칙을 추가한다.
6. 뉴스·공시는 표시 전용으로 유지한다.
7. 관련 단위 테스트와 최소 `tests/test_pipeline_smoke.py`를 실행한다.
