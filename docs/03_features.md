# 03. 피처 엔지니어링

`src/features/` 패키지는 OHLCV 및 투자자 흐름 데이터를 모델 입력 피처로 변환한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `price_features.py` | 가격 기반 피처 통합 빌드 (메인 진입점) |
| `technical_indicators.py` | RSI, MACD, ATR, Stochastic, OBV 등 기술지표 계산 |
| `external_features.py` | 외부 시장 데이터 (지수, VIX, 환율, 금리) 추가 |
| `regime_features.py` | 마켓 레짐 어노테이션 |
| `investment_signals.py` | 투자 시그널 피처 (거래대금 순위, RSI 풀백 등) |
| `feature_selection.py` | 모델 피처 컬럼 목록 관리 |

---

## 가격 피처 빌드 (`price_features.py`)

```python
# src/features/price_features.py
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame
def select_feature_columns(df: pd.DataFrame) -> list[str]
```

`build_features()`는 종목(`Symbol`)별로 그룹화한 뒤 다음 피처들을 순차적으로 계산한다:

### 수익률 계산

| 피처 | 설명 |
|------|------|
| `daily_return` | 일간 종가 수익률 |
| `gap_return` | 전일 종가 → 당일 시가 갭 |
| `intraday_return` | 당일 시가 → 종가 수익률 |
| `range_pct` | (고가 - 저가) / 종가 |
| `ret_1d` ~ `ret_60d` | N일 전 대비 수익률 (롤링 윈도우별) |

### 이동평균 피처

| 피처 | 설명 |
|------|------|
| `ma_5` ~ `ma_120` | N일 이동평균 |
| `close_to_ma_5` ~ `close_to_ma_120` | 종가 / 이동평균 - 1 |

### 변동성 피처

| 피처 | 설명 |
|------|------|
| `vol_5` ~ `vol_60` | N일 수익률 표준편차 |
| `vol_ratio_20` | 최근 5일 변동성 / 20일 변동성 |

### 투자 경고 수준

```python
# src/features/price_features.py:22
WARNING_LEVEL_MAP = {
    "none": 0.0, "투자주의": 1.0, "투자경고": 2.0, "투자위험": 3.0, ...
}
```

`market_warning` 컬럼을 숫자(0~3)로 인코딩하여 `warning_level` 피처로 변환.

### 타겟 변수

| 변수 | 설명 |
|------|------|
| `target_log_return` | 다음 거래일 로그 수익률 (회귀 타겟) |
| `target_up` | 다음 거래일 종가 상승 여부 (분류 타겟, 0/1) |

---

## 기술지표 (`technical_indicators.py`)

```python
# src/features/technical_indicators.py
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series
def compute_macd(series: pd.Series, ...) -> tuple[pd.Series, pd.Series, pd.Series]
def rolling_zscore(series: pd.Series, window: int) -> pd.Series
```

| 지표 | 피처명 | 기본 파라미터 |
|------|--------|---------------|
| RSI | `rsi_14` | 14일 |
| MACD | `macd`, `macd_signal`, `macd_hist` | (12, 26, 9) |
| ATR | `atr_14` | 14일 |
| Stochastic | `stoch_k`, `stoch_d` | 14일 |
| CCI | `cci_20` | 20일 |
| OBV | `obv`, `obv_change_5d` | — |

---

## 외부 시장 피처 (`external_features.py`)

```python
# src/features/external_features.py
def add_external_market_features_with_coverage(
    df: pd.DataFrame,
    market_symbols: list[str],
) -> tuple[pd.DataFrame, dict]
```

기본 수집 대상 지수/자산:

| 심볼 | 의미 | 피처 접두사 |
|------|------|-------------|
| `^KS11` | KOSPI | `ks_` |
| `^KQ11` | KOSDAQ | `kq_` |
| `^GSPC` | S&P500 | `gspc_` |
| `^IXIC` | NASDAQ Composite | `ixic_` |
| `NQ=F` | NASDAQ 선물 | `nq_f_` |
| `^SOX` | 필라델피아 반도체 | `sox_` |
| `^VIX` | VIX 변동성 지수 | `vix_` |
| `KRW=X` | 원/달러 환율 | `krw_` |
| `^TNX` | 미국 10년물 금리 | `tnx_` |

- `--disable-external` 옵션으로 비활성화 가능
- 커버리지 비율(`external_coverage_ratio`) 계산 후 커버리지 게이트 평가에 사용

---

## 마켓 레짐 (`regime_features.py`)

```python
# src/features/regime_features.py
def annotate_market_regime(df: pd.DataFrame) -> pd.DataFrame
```

시장 상태를 분류하여 피처로 추가한다:

| 피처 | 설명 |
|------|------|
| `market_type_kospi` | KOSPI 종목 여부 (0/1) |
| `market_type_kosdaq` | KOSDAQ 종목 여부 (0/1) |
| `venue_krx` / `venue_nxt` | 거래소 종류 |
| `session_regular` | 정규장 여부 |
| `days_since_listing` | 상장일로부터 경과 일수 |
| `is_newly_listed` | 최근 20일 이내 신규상장 |
| `is_newly_listed_60d` | 최근 60일 이내 신규상장 |

---

## 투자 시그널 피처 (`investment_signals.py`)

```python
# src/features/investment_signals.py
def add_investment_signal_features(df: pd.DataFrame, cfg: InvestmentCriteriaConfig) -> pd.DataFrame
```

`InvestmentCriteriaConfig` 기준으로 다음 피처를 생성한다:

| 피처 | 설명 | 기준 |
|------|------|------|
| `turnover_rank_daily` | 당일 거래대금 순위 | 전체 종목 대비 |
| `is_top_turnover_3` | 거래대금 상위 3위 이내 | `top_turnover_rank=15`에서 3위 기준 |
| `is_top_turnover_10` | 거래대금 상위 10위 이내 | |
| `value_traded` | 거래대금 (Close × Volume) | |
| 투자자 흐름 피처 | `foreign_net_buy`, `institution_net_buy` 등 | 투자자 컨텍스트 활성화 시 |

---

## 피처 선택 (`feature_selection.py`)

```python
# src/features/feature_selection.py
FEATURE_COLUMN_BASE: frozenset[str]      # 고정 모델 입력 컬럼
FEATURE_COLUMN_PREFIXES: tuple[str, ...] # 동적 피처 접두사 (ret_, ma_, ks_, ...)
DISPLAY_ONLY_CONTEXT_COLUMNS: list[str]  # 표시 전용 컬럼 (모델 입력 제외)

def select_feature_columns(df: pd.DataFrame) -> list[str]
```

- `select_feature_columns()`: 실제 DataFrame에 존재하는 컬럼 중 모델 입력 컬럼만 반환
- `DISPLAY_ONLY_CONTEXT_COLUMNS`: 뉴스·공시 요약 등 예측에 사용하지 않는 컬럼 목록
- 총 피처 수: 일반적으로 50~100개 (활성화된 기능에 따라 다름)

---

## 피처 빌드 순서 (파이프라인 내)

```
OHLCV DataFrame
    │
build_features()          ← price_features.py
    │  └─ technical_indicators.py
    │
add_external_market_features_with_coverage()  ← external_features.py
    │
annotate_market_regime()  ← regime_features.py
    │
add_investment_signal_features()  ← investment_signals.py
    │
dropna(subset=["target_log_return"])
    │
select_feature_columns()  ← feature_selection.py
    │
→ Walk-Forward 검증으로 전달
```

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성/누수) > P1(견고성) > P2(성능/품질/문서)**.

### P0 — 미국 지수 "당일" 조인으로 인한 미래정보 누수(look-ahead)

- **문제**: 외부 피처는 같은 **달력 날짜**로 병합된다(`external_features.py:185`). 그러나 `^GSPC`(S&P500)·`^IXIC`·`^SOX`·`^VIX`·`^TNX`의 당일 종가는 한국장 마감(15:30 KST) **이후** 미국장에서 확정된다. 즉 한국 종목의 `target`(다음날 수익률)을 예측하는 시점에 `gspc_ret_1d`(당일) 값은 **아직 알 수 없는 정보**이며, 이는 검증 성능을 과대평가하게 만드는 누수다.
- **제안**: 미국·유럽 지수 피처는 **1거래일 시프트**(전일 미국 종가)하여 한국장 마감 시점에 가용한 정보만 사용. 24시간 거래되는 `NQ=F` 선물은 한국 마감 시점 스냅샷으로 별도 정렬.

### P0 — 외부 피처 `bfill()`이 미래값을 과거로 채움

- **문제**: 병합 후 `ext.sort_values("Date").ffill().bfill()`(`external_features.py:184`). `ffill`은 과거→현재로 안전하지만 `bfill`은 시계열 **앞쪽 결측을 미래값으로 역채움**하여 누수가 발생한다.
- **제안**: `bfill` 제거. 선행 결측은 NaN 유지 후 모델 입력 시 처리하거나, 해당 구간을 학습/검증에서 제외.

### P0 — 문서 오류: `vol_ratio_20` 정의

- **문제**: 문서는 `vol_ratio_20 = 최근 5일 변동성 / 20일 변동성`이라고 하지만, 실제 코드는 **거래량 비율** `Volume / rolling(20).mean(Volume)`이다(`price_features.py:192`). 변동성과 무관하다.
- **제안**: 문서를 "당일 거래량 / 20일 평균 거래량(거래량 급증 지표)"로 정정. 진짜 변동성비가 필요하면 `vol_5 / vol_20` 피처를 별도 추가.

### P1 — `near_52w_high_flag` 임계값이 설정과 불일치(0.95 하드코딩)

- **문제**: `near_52w_high_flag = (close_to_52w_high >= 0.95)`로 하드코딩(`price_features.py:262`). 반면 `InvestmentCriteriaConfig.near_52w_distance_threshold = 0.03`(→ 0.97 의미)은 **사용되지 않으며**, `06_signal_policy.md`는 "× 0.97"로 설명한다. 세 곳이 제각각이다.
- **제안**: 하드코딩 0.95를 `1 - cfg.near_52w_distance_threshold`로 치환해 설정·코드·문서를 일치.

### P1 — `obv_change_5d` 등 0/부호변화에 의한 inf 미처리

- **문제**: `obv_change_5d = obv.pct_change(5)`(`price_features.py:231`)는 OBV가 0을 지나거나 부호가 바뀌면 `inf/-inf`가 된다. 모델 입력 직전 `fillna(0)`만 적용되어 inf는 그대로 남을 수 있다.
- **제안**: 기술지표 계산 후 `replace([inf,-inf], nan)`을 일괄 적용. OBV 변화율은 차분/zscore 등 스케일 안정적 형태로 대체 검토.

### P1 — 상·하한가 임계값 하드코딩(±0.295)

- **문제**: `limit_hit_up/down_flag`가 ±29.5%로 고정(`price_features.py:282-283`). 한국 가격제한폭은 2015년 이전 ±15%였고 ETF/신규상장/정리매매는 예외다. 과거 데이터·특수종목에서 오탐.
- **제안**: 제한폭을 설정값/시장·기간별 테이블로 분리.

### P2 — 기술지표 로직 중복(DRY)

- **문제**: `technical_indicators.py`에 `compute_atr/compute_stochastic/compute_cci/compute_obv` 헬퍼가 있으나, `price_features.py`는 이를 호출하지 않고 **인라인으로 재구현**한다(예: ATR `price_features.py:208-213`). 한쪽만 수정되면 정의가 갈라진다.
- **제안**: `price_features`가 `groupby().apply`로 헬퍼를 재사용하도록 통합하고, 단위 테스트는 헬퍼에 집중.

### P2 — 그룹 롤링 성능

- **문제**: `groupby(...).transform(lambda x: x.rolling(...))`·`groupby.apply(_macd_group)` 패턴이 다수라(`price_features.py`) 종목 수가 많아지면 느리다. `grouped`도 `out`이 바뀔 때마다 재생성된다.
- **제안**: 종목별 정렬 후 벡터화 롤링(또는 `numba`/`bottleneck`), 그룹 객체 1회 생성 후 재사용, 한 번의 `concat`으로 컬럼 병합(이미 일부 적용됨)을 전 구간으로 확대.

### P2 — 결측 기본값 0의 의미 점검

- **문제**: 다수 피처가 결측 시 0으로 채워진다. `close_to_ma_*`(중심 0)에는 중립이지만 `rsi_14`(중립 50)·`stoch_k` 등에는 0이 왜곡된 값이다. 모델 `predict`도 `fillna(0)`(`lgbm_heads.py:183`).
- **제안**: 피처별 "중립값" 사전을 정의해 채우고, 결측 비율을 진단 지표로 노출.
