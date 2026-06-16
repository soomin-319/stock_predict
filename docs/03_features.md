# 03. 피처 엔지니어링

`src/features/`는 OHLCV, 외부 시장, 투자자 수급 데이터를 모델 입력 피처로 변환한다.
뉴스와 공시는 표시·검토용 문맥이며 모델 입력, 예상 수익률, 순위, 추천 결정에 사용하지 않는다.

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `price_features.py` | 가격·수익률·거래량 피처 조립, 결측 진단 |
| `technical_indicators.py` | RSI, MACD, ATR, Stochastic, CCI, OBV의 단일 구현 |
| `external_features.py` | 시장 지수·환율·금리 피처 수집과 시점 정렬 |
| `regime_features.py` | 시장 레짐 피처 |
| `investment_signals.py` | 설정 기반 투자 보조 플래그 |
| `feature_selection.py` | 모델 입력과 표시 전용 컬럼 분리 |

## 파이프라인 순서

```text
OHLCV
  -> build_features()
  -> add_external_market_features_with_coverage()
  -> annotate_market_regime()
  -> add_investment_signal_features()
  -> target_log_return 결측 행 제거
  -> select_feature_columns()
  -> Walk-Forward 검증
```

`build_features()`는 내부 계산 전에 `Symbol`, `Date` 순으로 안정 정렬하고 반환 전에 입력 행 순서를 복원한다.

## 가격·거래량 피처

주요 컬럼:

- `daily_return`, `gap_return`, `intraday_return`, `range_pct`
- `ret_1d` ~ `ret_60d`
- `ma_5` ~ `ma_120`, `close_to_ma_*`
- `vol_5` ~ `vol_60`
- `value_traded`, `turnover_rank_daily`

`vol_ratio_20`은 변동성 비율이 아니다. 다음 거래량 급증 지표다.

```text
당일 거래량 / 최근 20거래일 평균 거래량
```

## 기술지표

`technical_indicators.py`가 RSI, MACD, ATR, Stochastic, CCI, OBV의 단일 정의를 제공한다.
`price_features.py`는 종목별 기술지표 블록을 한 번 계산하여 합친다. 같은 공식을 별도로 재구현하지 않는다.

모든 생성 피처의 `inf`, `-inf`는 `NaN`으로 변환한다. 특히 OBV가 0을 통과하더라도
`obv_change_5d`가 무한대가 되지 않는다.

## 외부 시장 피처와 시점 정렬

기본 대상은 KOSPI, KOSDAQ, S&P 500, NASDAQ, NASDAQ 선물, SOX, VIX, 원/달러 환율,
미국 10년물 금리다.

- `^KS11`, `^KQ11`: 한국 장 마감 시점에 확정되므로 같은 날짜에 결합한다.
- 그 외 해외 지수·선물·환율·금리: yfinance 일봉만으로 한국 15:30 시점 값을 판별할 수
  없으므로 한 관측치를 지연하여 결합한다.
- 날짜 정렬 후 과거 값을 사용하는 `ffill`만 허용한다.
- 미래 값을 과거로 복사하는 `bfill`은 금지한다.
- 최초 관측 이전 구간은 `NaN`으로 유지한다.

이 정책은 보수적으로 look-ahead 누수를 방지한다. 장중 시점 피처가 필요하면 시각 정보가 있는
별도 데이터 공급자와 명시적 시점 정렬이 필요하다.

## 52주 고가 기준

`build_features()`는 연속값 `close_to_52w_high`와 돌파 여부 `breakout_52w_flag`를 생성한다.
`near_52w_high_flag`의 최종 기준은 `investment_signals.py`가
`InvestmentCriteriaConfig.near_52w_distance_threshold`를 사용하여 계산한다.

따라서 52주 고가 근접 기준을 코드에 `0.95` 등으로 하드코딩하지 않는다.

## 가격제한폭 피처

`limit_hit_up_flag`, `limit_hit_down_flag`, `limit_event_flag`는 다음 우선순위를 사용한다.

1. 행에 `price_limit_pct` 또는 `PriceLimitPct`가 있으면 해당 값을 사용한다.
2. 명시값이 없으면 일반 한국 주식 기본값을 사용한다.
   - 2015-06-15 이전: ±15%
   - 2015-06-15 이후: ±30%

ETF, 신규상장, 재상장, 정리매매 등 특수 예외는 필요한 메타데이터가 없으면 추측하지 않는다.

## 결측값 정책과 진단

중립 의미가 명확한 피처만 명시적으로 채운다.

| 피처 | 중립값 |
|---|---:|
| `rsi_14` | 50 |
| `stoch_k`, `stoch_d` | 50 |
| `macd`, `macd_signal`, `macd_hist` | 0 |

이력이 부족하여 계산할 수 없는 절대 수준 피처는 임의의 0으로 위장하지 않는다.
`ma_120`, `vol_60`, `atr_14`, `cci_20`, `obv_change_5d`에는 `<피처>_missing`
진단 플래그가 추가된다.

`feature_missing_rate_summary()`는 선택 피처별 결측률을 계산한다. 파이프라인 보고서의
`diagnostics.feature_missing_rates`에 기록되며 신호와 추천에는 사용하지 않는다.

## 피처 선택과 표시 전용 문맥

`select_feature_columns()`는 허용된 기본 컬럼, 접두사 컬럼, `_missing` 진단 플래그만 모델에 전달한다.

`DISPLAY_ONLY_CONTEXT_COLUMNS`에는 뉴스·공시와 이를 포함하는 합성 문맥 컬럼이 들어간다.
이 컬럼들은 결과 표시에는 사용할 수 있지만 모델 학습, 예상 수익률, 순위, 매수·매도·관망 결정에
사용할 수 없다.

## 타깃

- `target_log_return`: 다음 거래일 로그 수익률
- `target_up`: 다음 거래일 상승 여부
- `target_close`: 다음 거래일 종가

타깃은 다음 날 값을 사용하므로 학습·검증 입력 피처에 포함하지 않는다.
