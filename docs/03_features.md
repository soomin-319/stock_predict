# 03. 피처 — 가격·기술적·외부·투자자 이벤트

`src/features/`는 정제된 OHLCV(+선택 컨텍스트)에서 모델 입력 피처와 표시용 컨텍스트 컬럼,
학습 타깃을 생성한다. 모든 피처는 종목별로 시간 정렬 후 계산되며, 다음날을 보는 누수가 없도록
타깃만 미래 시프트를 사용한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `price_features.py` | 메인 피처 빌더(`build_features`), 타깃 생성, 결측/중립값 처리 |
| `technical_indicators.py` | RSI/MACD/ATR/Stochastic/CCI/OBV 등 지표 계산 |
| `external_features.py` | yfinance 외부 시장 지수 피처(가용성 래그 적용) |
| `regime_features.py` | 시장 국면(추세×변동성) 라벨 (표시용) |
| `investment_signals.py` | 투자 지원 플래그(고확신 순매수, 52주 고점, 섹터 리더 등) |
| `feature_selection.py` | 모델 피처 컬럼 선택 및 표시 전용 컬럼 분리 |

---

## 메인 빌더 (`build_features`)

```python
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame
```

처리 순서:

1. 입력 순서를 `_feature_input_order`로 보존하고 `[Symbol, Date]` 안정 정렬.
2. 컨텍스트 수치 컬럼을 한글/영문 별칭에서 정규화(`foreign_net_buy`, `disclosure_score`, `news_*` 등).
3. **수익률/거래대금 피처**: `log_return`, `daily_return`, `gap_return`, `intraday_return`, `range_pct`,
   `value_traded`, 일별 `turnover_rank_daily`, `is_top_turnover_3/10`.
4. **룩백/이동평균/변동성**: `ret_{w}d`, `ma_{w}` + `close_to_ma_{w}`, `vol_{w}`, `vol_ratio_20` (`FeatureConfig` 윈도우 기반).
5. **기술적 지표 블록**: 종목별 `compute_technical_indicator_block` → `rsi_14`, `macd(_signal/_hist)`,
   `atr_14`, `stoch_k/d`, `cci_20`, `obv`, `obv_change_5d`. RSI 풀백/과매수 플래그 파생.
6. **투자자/이벤트 피처**: `foreign/institution_buy_signal`, `smart_money_*`, z-score(20일), 3/5일 합,
   `news_positive/negative_signal`, `close_to_52w_high`, `breakout_52w_flag`, `leader_confirmation_flag`,
   `investor_event_score`, 상/하한가 플래그, 단기과열/공매도/주주환원 점수 등.
7. **레거시 컬럼 정리**: 하위호환용 기본값 컬럼과 원본 한글/영문 소스 컬럼을 마지막에 제거.
8. **타깃 생성**: `target_log_return = log(Close[t+1]/Close[t])`, `target_up = (target_log_return > 0)`,
   `target_close = Close * exp(target_log_return)`.
9. **결측 표시/중립값**: `MISSING_INDICATOR_SOURCE_COLUMNS`에 대해 `{col}_missing` 플래그 생성,
   `NEUTRAL_FEATURE_VALUES`(rsi=50, stoch=50, macd=0 등)로 지표 중립값 채움.

> 가격 상/하한가 임계값은 날짜 기준이다. `KRX_PRICE_LIMIT_CHANGE_DATE`(2015-06-15) 이전은 15%,
> 이후는 30%를 기본으로 사용하며, 입력에 `price_limit_pct`가 있으면 우선한다.

## 기술적 지표 (`technical_indicators.py`)

| 지표 | 함수 | 비고 |
|------|------|------|
| RSI | `compute_rsi` | Wilder식 EWM, 결측은 50으로 채움 |
| MACD | `compute_macd` | EMA(12,26), 시그널 EMA(9), 히스토그램 |
| ATR | `compute_atr` | True Range 14일 평균 |
| Stochastic | `compute_stochastic` | %K, %D(3일) |
| CCI | `compute_cci` | 20일, MAD 기반 |
| OBV | `compute_obv` | 누적 거래량 방향 |
| z-score | `rolling_zscore` | 투자자 수급 표준화에 사용 |

모든 블록은 `inf/-inf`를 `NaN`으로 치환해 반환한다.

## 외부 시장 피처 (`external_features.py`)

```python
def add_external_market_features_with_coverage(df, symbols) -> tuple[pd.DataFrame, dict]
```

- 기본 심볼: `^KS11, ^KQ11, ^GSPC, ^IXIC, NQ=F, ^SOX, ^VIX, KRW=X, ^TNX`(`ExternalFeatureConfig`).
- 각 심볼마다 `{base}_close`, `{base}_ret_1d`, `{base}_ret_5d`, `{base}_vol_20`을 만든다.
- **폴백 심볼**: 실패 시 ETF 대체(예: `^SOX→SOXX`, `^VIX→VIXY`, `^GSPC→SPY`).
- **가용성 래그**: 국내 지수(`^KS11`, `^KQ11`)를 제외한 해외 심볼은 1일 시프트(`_apply_availability_lag`)하여
  당일 종가 예측 시점에 사용 불가능한 미래 정보를 막는다.
- **커버리지**: 요청/성공/실패/폴백 사용 수를 반환하며, 0건 성공이면 외부 피처 없이 진행.
- 최대 4개 스레드 다운로드, 머지 후 `ffill`.

## 시장 국면 (`regime_features.py`)

```python
def annotate_market_regime(df) -> pd.DataFrame
```

`close_to_ma_20`(추세)과 `vol_20`(변동성)으로 `{uptrend|downtrend|sideways}_{high_vol|low_vol}` 라벨을 만든다.
이 컬럼(`market_regime`)은 표시·진단용이며 모델 피처가 아니다.

## 투자 지원 플래그 (`investment_signals.py`)

```python
def add_investment_signal_features(df, cfg: InvestmentCriteriaConfig) -> pd.DataFrame
```

`InvestmentCriteriaConfig` 임계값을 사용해 플래그를 생성한다.

| 플래그 | 의미 |
|--------|------|
| `is_top_turnover_15` | 거래대금 상위 N위 이내(`top_turnover_rank`) |
| `foreign/institution_high_conviction_buy_flag` | 고확신 순매수(`high_conviction_net_buy_krw` 이상) |
| `dual_high_conviction_buy_flag` | 외국인·기관 동시 고확신 |
| `near_52w_high_flag`, `breakout_52w_flag` | 52주 고점 근접/돌파 (`near_52w_distance_threshold`) |
| `nasdaq_tailwind/headwind_flag` | NASDAQ 선물 수익률 임계 |
| `rsi_buy_watch_flag`, `rsi_overbought_sell_flag` | RSI 관찰/과매수 |
| `leader_confirmation_flag` | 거래대금 상위 종목 동반 상승(섹터 리더) |
| `news/disclosure_same_day_signal` | 당일 뉴스/공시 존재(표시용) |

## 피처 선택 (`feature_selection.py`)

```python
def select_feature_columns(df) -> list[str]
def display_context_columns(df) -> list[str]
```

- `FEATURE_COLUMN_PREFIXES`(`ret_`, `ma_`, `close_to_ma_`, `vol_`, `ks`, `kq`, `gspc`, `ixic`, `nq_f`,
  `sox`, `vix`, `krw`, `tnx`) 또는 `MODEL_FEATURE_COLUMN_BASE`에 속하거나 `_missing`으로 끝나는 컬럼이
  모델 입력 후보다.
- **표시 전용 제외**: `DISPLAY_ONLY_CONTEXT_COLUMNS`(`disclosure_score`, `news_*`, 합성 `investor_event_score`)와
  `DISPLAY_ONLY_CONTEXT_PREFIXES`(`news_impact_`)는 모델 입력에서 항상 제외된다.
- `MODEL_FEATURE_COLUMN_BASE = FEATURE_COLUMN_BASE - DISPLAY_ONLY_CONTEXT_COLUMNS`로 정의되어,
  뉴스/공시 컨텍스트가 피처로 새어들지 않도록 한다.

> 이 분리가 "뉴스/공시는 표시용이며 기대수익률·추천을 바꾸지 않는다"는 핵심 가드레일의 구현 지점이다.

---

## 개선 및 수정 제안

> 우선순위: **P2(성능/명확성)**. 정확성·누수 관련 항목은 현재 코드에서 발견되지 않았다.

### P2 — `_leader_confirmation` 날짜별 파이썬 루프

- **문제**: `investment_signals._leader_confirmation`은 날짜 그룹마다 정렬 후 `.loc` 행 단위 대입으로
  리더 수익률/확인 플래그를 채운다(`investment_signals.py:42-58`). 종목·일수가 많은 전체 유니버스에서
  파이썬 루프 오버헤드가 누적된다. (`price_features.build_features`의 `leader_confirmation_flag`는 이미
  벡터화되어 있어 두 경로의 정의가 중복된다.)
- **제안**: 날짜 그룹 벡터화(`groupby(...).transform`)로 통일하거나, 빌더에서 이미 만든 값을 재사용해 중복 계산을 제거한다.

### P2 — `market_regime` 변동성 분위수 기준

- **문제**: `annotate_market_regime`의 `high_vol` 경계가 **전체 프레임의 `vol_20` 75분위**를 사용한다
  (`regime_features.py:16`). 전 기간/전 종목을 한꺼번에 보는 기준이라 시점별(point-in-time) 국면이 아니다.
- **영향**: `market_regime`은 모델 피처가 아니라 표시·진단용이므로 예측 누수는 아니다. 다만 라벨 해석 시
  "현재 시점 기준"이 아님을 유념해야 한다.
- **제안**: 국면 라벨을 시점 기준으로 쓰려면 롤링 분위수 또는 일자 횡단면 분위수로 정의한다.
